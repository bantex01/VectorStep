#!/usr/bin/env python3
"""Backfill prompt_hash and agent_version on historical pipeline_steps rows.

See SPEC-prompt-versioning.md §4i. On the day the calibration bucket key widens to
include prompt_hash/agent_version, every historical row has NULL for both — every
existing bucket de-validates at once, and every `enforce: true` step drops to its
`on_uncalibrated` behaviour with no warning. This script stamps historical rows with
the *current* prompt hash and *current* agent version, on the operator's explicit
assertion that they have not changed them since those runs happened.

Usage:
    python scripts/backfill_prompt_versions.py --config <path> --assume-unchanged [--dry-run]

This is a standalone script, not an endpoint — run it once, by hand, right after
upgrading.
"""
import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict

import httpx
from sqlalchemy import distinct, select

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.database import create_tables, get_session_factory, init_db  # noqa: E402
from src.db.models import PipelineStep  # noqa: E402
from src.models.pipeline import FanOutGroupConfig, ParallelGroupConfig, PipelineConfig, StepConfig  # noqa: E402
from src.pipeline.calibration import compute_calibration_buckets  # noqa: E402
from src.pipeline.loader import load_pipelines, load_step_library  # noqa: E402
from src.pipeline.versioning import prompt_hash, record_agent_version, record_prompt_version  # noqa: E402

logger = logging.getLogger("backfill_prompt_versions")

_ASSUME_UNCHANGED_HELP = (
    "This asserts that the prompt templates and agent definitions currently on disk "
    "are the ones that produced your historical runs. If you have edited them, this "
    "will attribute old outcomes to the current version — exactly the bug this "
    "feature exists to prevent."
)


def _build_template_index(
    pipelines: list[PipelineConfig],
) -> tuple[dict[str, str], dict[str, str]]:
    """Map runtime PipelineStep.step_name -> its currently-configured prompt_template.

    Two lookup tables, because the runtime step_name shape differs by step kind:
      - plain steps and parallel branches: an exact "name" or "group/branch" match.
      - fan-out branches: runtime names are "group/<item-index>", but every branch of
        a given fan-out group shares the SAME template (rendered differently per item
        at runtime) — so this is keyed by group name and matched by prefix.
    First pipeline to define a given name wins if the same name appears in more than
    one pipeline (this is a one-shot backfill under an explicit operator assertion,
    not a strict validator).
    """
    by_name: dict[str, str] = {}
    fanout_by_group: dict[str, str] = {}
    for pipeline in pipelines:
        for step in pipeline.steps:
            if isinstance(step, StepConfig):
                by_name.setdefault(step.name, step.prompt_template)
            elif isinstance(step, ParallelGroupConfig):
                for branch in step.parallel.steps:
                    by_name.setdefault(f"{step.parallel.name}/{branch.name}", branch.prompt_template)
            elif isinstance(step, FanOutGroupConfig):
                fanout_by_group.setdefault(step.fan_out.name, step.fan_out.prompt_template)
    return by_name, fanout_by_group


def _resolve_template(
    step_name: str, by_name: dict[str, str], fanout_by_group: dict[str, str],
) -> str | None:
    if step_name in by_name:
        return by_name[step_name]
    if "/" in step_name:
        prefix = step_name.split("/", 1)[0]
        if prefix in fanout_by_group:
            return fanout_by_group[prefix]
    return None


async def _fetch_gateway_agent_versions(rest_url: str) -> dict[str, str] | None:
    """{'gateway:<name>': version} for every agent the Gateway currently reports, or
    None if the Gateway couldn't be reached at all — the caller must not stamp a
    guess in that case, it must skip agent_version backfill entirely."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{rest_url}/agents")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Could not reach Gateway at %s: %s", rest_url, exc)
        return None
    return {
        f"gateway:{a['name']}": a["version"]
        for a in data.get("agents", [])
        if a.get("name") and a.get("version")
    }


async def _backfill_prompt_hashes(
    session_factory, by_name: dict[str, str], fanout_by_group: dict[str, str], dry_run: bool,
) -> tuple[dict[tuple[str, str], int], list[str]]:
    """Returns ({(step_name, hash): row_count}, [skipped step_names])."""
    async with session_factory() as session:
        step_names = (await session.execute(
            select(distinct(PipelineStep.step_name)).where(PipelineStep.prompt_hash.is_(None))
        )).scalars().all()

    stamped: dict[tuple[str, str], int] = {}
    skipped: list[str] = []
    for step_name in sorted(step_names):
        template = _resolve_template(step_name, by_name, fanout_by_group)
        if template is None:
            skipped.append(step_name)
            continue
        h = prompt_hash(template)
        if h is None:
            # Current template is empty/whitespace-only — nothing to stamp, and this
            # is correct: a step with no real prompt has no meaningful version.
            continue

        async with session_factory() as session:
            rows = (await session.execute(
                select(PipelineStep).where(
                    PipelineStep.step_name == step_name, PipelineStep.prompt_hash.is_(None),
                )
            )).scalars().all()
            stamped[(step_name, h)] = len(rows)
            if not dry_run:
                for row in rows:
                    row.prompt_hash = h
                await record_prompt_version(session, hash_=h, step_name=step_name, template=template)
                await session.commit()

    return stamped, skipped


async def _backfill_agent_versions(
    session_factory, rest_url: str, dry_run: bool,
) -> tuple[dict[tuple[str, str], int], list[str], bool]:
    """Returns ({(agent, version): row_count}, [skipped agent names], gateway_reachable)."""
    agent_versions = await _fetch_gateway_agent_versions(rest_url)
    if agent_versions is None:
        return {}, [], False

    async with session_factory() as session:
        agents = (await session.execute(
            select(distinct(PipelineStep.agent)).where(
                PipelineStep.agent_version.is_(None), PipelineStep.agent.is_not(None),
            )
        )).scalars().all()

    stamped: dict[tuple[str, str], int] = {}
    skipped: list[str] = []
    for agent in sorted(agents):
        if not agent.startswith("gateway:"):
            continue  # not a gateway-executed step — agent_version is legitimately N/A
        version = agent_versions.get(agent)
        if version is None:
            skipped.append(agent)
            continue

        async with session_factory() as session:
            rows = (await session.execute(
                select(PipelineStep).where(
                    PipelineStep.agent == agent, PipelineStep.agent_version.is_(None),
                )
            )).scalars().all()
            stamped[(agent, version)] = len(rows)
            if not dry_run:
                for row in rows:
                    row.agent_version = version
                await record_agent_version(session, rest_url, agent_version=version, agent=agent)
                await session.commit()

    return stamped, skipped, True


async def _validated_buckets_among(
    session_factory, n_min: int, bin_width: float, stamped_hashes: set[str], stamped_versions: set[str],
) -> list[str]:
    """After a real (non-dry-run) backfill, which newly-stamped buckets are already
    validated (>= n_min in at least one bin)? Only meaningful post-write — a dry-run
    hasn't touched the DB, so this is skipped for --dry-run (an honest gap, not a
    guess)."""
    buckets = await compute_calibration_buckets(session_factory, bin_width=bin_width, n_min=n_min)
    lines = []
    for (step_name, agent, model, provider, p_hash, a_version), bucket in buckets.items():
        if p_hash not in stamped_hashes and a_version not in stamped_versions:
            continue
        if any(b.validated for b in bucket.bins):
            lines.append(
                f"  {step_name} / agent={agent} model={model} provider={provider} "
                f"prompt_hash={p_hash} agent_version={a_version} (n={bucket.total_n})"
            )
    return lines


async def run(config_path: str, dry_run: bool) -> None:
    os.environ["CONFIG_PATH"] = config_path
    from src.main import _load_config  # same env-var-resolved config loader main.py's lifespan uses

    config = _load_config()
    db_url = config.get("database", {}).get("url", "sqlite+aiosqlite:///./runs.db")
    step_library_dir = config.get("step_library_dir", "./steps")
    pipeline_dir = config.get("pipeline_config_dir", "./pipelines")
    gateway_cfg = config.get("executors", {}).get("gateway", {})
    rest_url = gateway_cfg.get("rest_url") or os.environ.get("PORK_GATEWAY_URL", "http://localhost:18780")
    calibration_cfg = config.get("calibration", {})
    n_min = calibration_cfg.get("n_min", 20)
    bin_width = calibration_cfg.get("bin_width", 0.1)

    init_db(db_url)
    await create_tables()
    session_factory = get_session_factory()

    step_library = load_step_library(step_library_dir)
    pipelines = load_pipelines(pipeline_dir, step_library=step_library)
    by_name, fanout_by_group = _build_template_index(pipelines)

    print(f"{'DRY RUN — ' if dry_run else ''}Backfilling against {db_url}")
    print(f"Loaded {len(pipelines)} pipeline(s), {len(step_library)} library step(s)\n")

    prompt_stamped, prompt_skipped = await _backfill_prompt_hashes(
        session_factory, by_name, fanout_by_group, dry_run,
    )
    agent_stamped, agent_skipped, gateway_reachable = await _backfill_agent_versions(
        session_factory, rest_url, dry_run,
    )

    verb = "Would stamp" if dry_run else "Stamped"
    print(f"prompt_hash — {verb.lower()} {sum(prompt_stamped.values())} row(s) across "
          f"{len(prompt_stamped)} (step, hash) pair(s):")
    for (step_name, h), count in sorted(prompt_stamped.items()):
        print(f"  {step_name} -> {h}  ({count} row(s))")
    if prompt_skipped:
        print(f"  Skipped {len(prompt_skipped)} step name(s) no longer in the step "
              f"library/pipelines — rows stay NULL:")
        for name in prompt_skipped:
            print(f"    {name}")

    print()
    if not gateway_reachable:
        print("agent_version — Gateway unreachable. SKIPPED ENTIRELY. Rows stay NULL. "
              "Re-run this script once the Gateway is reachable.")
    else:
        print(f"agent_version — {verb.lower()} {sum(agent_stamped.values())} row(s) across "
              f"{len(agent_stamped)} (agent, version) pair(s):")
        for (agent, version), count in sorted(agent_stamped.items()):
            print(f"  {agent} -> {version}  ({count} row(s))")
        if agent_skipped:
            print(f"  Skipped {len(agent_skipped)} agent(s) no longer on the Gateway — rows stay NULL:")
            for name in agent_skipped:
                print(f"    {name}")

    print()
    if dry_run:
        print("(--dry-run: nothing written. Run again without --dry-run to apply, and to see "
              "which buckets that creates are already validated.)")
    else:
        stamped_hashes = {h for (_s, h) in prompt_stamped}
        stamped_versions = {v for (_a, v) in agent_stamped}
        validated_lines = await _validated_buckets_among(
            session_factory, n_min, bin_width, stamped_hashes, stamped_versions,
        )
        if validated_lines:
            print(f"Buckets now validated (>= {n_min} labelled results in at least one bin):")
            for line in validated_lines:
                print(line)
        else:
            print(f"No newly-stamped bucket has reached n_min={n_min} yet.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--assume-unchanged", action="store_true", required=True, help=_ASSUME_UNCHANGED_HELP,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print row counts per (step_name, hash) / (agent, version) it would write, and exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run(args.config, args.dry_run))


if __name__ == "__main__":
    main()
