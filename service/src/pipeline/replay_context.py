"""Rebuilds a step's Jinja2 context from persisted PipelineRun/PipelineStep rows.

Shared with main.py's `/runs/{run_id}/rerun` endpoint, which needed exactly this
— "reconstruct prior step outputs + normalised context from a completed run" —
before this module existed and had it inlined. Extracted here (per
SPEC-replay-shadow-eval.md §notes: "name it replay_context.py regardless of
which spec creates it") so both call sites share one definition and never
silently drift apart; main.py's rerun_from_step now calls into this module too.

Split pure-vs-IO, readiness.py style: reconstruct_normalised_context and
group_prior_step_outputs are pure (operate on already-fetched rows);
load_run_and_steps is the only I/O.
"""
import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import selectinload

from ..db.models import PipelineRun, PipelineStep
from ..models.context import NormalisedContext
from ..models.llm import LLMOutput
from ..models.pipeline import PipelineConfig
from .context import build_context

logger = logging.getLogger(__name__)


class ContextReconstructionError(Exception):
    """Raised when a step's Jinja context cannot be rebuilt from persisted rows —
    run/step missing, owning pipeline no longer loaded, or a prior step's
    parsed_output doesn't parse. replay.py catches this per-sample and counts
    the sample as `unreplayable`; it is never silently dropped."""


async def load_run_and_steps(
    session_factory: async_sessionmaker, run_id: str,
) -> tuple[PipelineRun, list[PipelineStep]]:
    """The only I/O in this module. Raises ContextReconstructionError if the run
    doesn't exist. Steps are returned sorted by step_index."""
    async with session_factory() as session:
        result = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.id == run_id)
            .options(selectinload(PipelineRun.steps))
        )
        run = result.scalar_one_or_none()
    if run is None:
        raise ContextReconstructionError(f"run '{run_id}' not found")
    return run, sorted(run.steps, key=lambda s: s.step_index)


def reconstruct_normalised_context(
    run: PipelineRun, *, source: str = "replay", metadata_extra: dict | None = None,
) -> NormalisedContext:
    """Pure. Rebuilds NormalisedContext from PipelineRun.normalised_context's
    persisted JSON — the same fields main.py's rerun_from_step has always
    pulled out by hand, lifted here so both call sites agree. `source` and
    `fingerprint=None` are always overridden: a reconstructed context is never
    the original webhook delivery and must never dedupe against it."""
    try:
        raw = json.loads(run.normalised_context)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContextReconstructionError(
            f"run '{run.id}': normalised_context is not valid JSON: {exc}"
        ) from exc
    return NormalisedContext(
        source=source,
        pipeline=raw.get("pipeline", ""),
        severity=raw.get("severity"),
        labels=raw.get("labels", {}),
        summary=raw.get("summary"),
        fingerprint=None,
        raw=raw.get("raw", {}),
        metadata={**raw.get("metadata", {}), **(metadata_extra or {})},
    )


def group_prior_step_outputs(
    steps: list[PipelineStep], *, before_step_index: int,
) -> dict[str, LLMOutput]:
    """Pure. Every step with step_index < before_step_index, keyed by step_name
    (a fan-out/parallel branch stored as "group/branch" registers under the
    branch name, same unwrap main.py's rerun_from_step has always done — so
    {{steps.branch_name.field}} resolves the same way it did in production).
    A step with no parsed_output (it failed before producing one) is skipped,
    not errored — that's a legitimate upstream state, distinct from the
    downstream corruption ContextReconstructionError is for."""
    outputs: dict[str, LLMOutput] = {}
    for step_row in steps:
        if step_row.step_index >= before_step_index:
            continue
        if not step_row.parsed_output:
            continue
        try:
            parsed = json.loads(step_row.parsed_output)
            output = LLMOutput.model_validate(parsed)
        except Exception as exc:
            raise ContextReconstructionError(
                f"step '{step_row.step_name}' (run '{step_row.run_id}'): "
                f"parsed_output does not parse as LLMOutput: {exc}"
            ) from exc
        name = step_row.step_name.split("/", 1)[1] if "/" in step_row.step_name else step_row.step_name
        outputs[name] = output
    return outputs


async def build_replay_context(
    session_factory: async_sessionmaker,
    pipeline_registry: dict[str, PipelineConfig],
    run_id: str,
    step_name: str,
    *,
    artifact_store=None,
) -> dict:
    """The one function replay.py's `rerender` mode calls. Loads the recorded
    run + its steps, reconstructs NormalisedContext and every prior step's
    output, then delegates to pipeline.context.build_context — the exact
    function production execution uses — so replay sees an identical context
    shape to what the recorded step actually saw.

    Raises ContextReconstructionError (never returns a partial/best-effort
    context) if the run, the target step, or the owning pipeline config can't
    be found, or if the JSON on any prior step doesn't parse.
    """
    run, steps = await load_run_and_steps(session_factory, run_id)

    target = next((s for s in steps if s.step_name == step_name), None)
    if target is None:
        raise ContextReconstructionError(f"step '{step_name}' not found on run '{run_id}'")

    pipeline = pipeline_registry.get(run.pipeline_name)
    if pipeline is None:
        raise ContextReconstructionError(
            f"pipeline '{run.pipeline_name}' is not currently loaded — cannot "
            "resolve context_template.include / vars for reconstruction"
        )

    normalised = reconstruct_normalised_context(
        run, metadata_extra={"original_run_id": run_id, "replayed_step": step_name},
    )
    step_outputs = group_prior_step_outputs(steps, before_step_index=target.step_index)

    return await build_context(
        pipeline, normalised, run_id, step_name, step_outputs,
        artifact_store=artifact_store,
    )
