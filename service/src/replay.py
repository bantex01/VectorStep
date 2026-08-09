"""Replay / shadow evaluation of recorded step executions (SPEC-replay-shadow-eval.md).

Takes the most recent K labelled, production executions of a step's calibration
bucket and re-runs them against a candidate model/agent/prompt, so a
configuration change can earn evidence before it's promoted rather than only
after. Deliberately mirrors readiness.py's split: everything that can be pure
(bucket-key resolution given rows, candidate StepConfig construction, report
rollup math) takes plain values and does no I/O; run_replay_batch and
get_report are the only places that touch the DB or the live executors.

Design decisions are recorded in the spec, not repeated here — see §2 for why
replay is allowlist-gated, why batches are ordinary stage='testing' runs, why
there's no LLM auto-judge in Phase 1, and why D-check failure is the only
auto-label (same asymmetry resolve_label already encodes for production).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from .db.models import PipelineRun, PipelineStep, RunFeedback, StepFeedback
from .models.llm import LLMOutput
from .models.pipeline import PipelineConfig, StepConfig
from .models.replay import ReplayConfig
from .pipeline.calibration import resolve_label
from .pipeline.replay_context import ContextReconstructionError, build_replay_context
from .pipeline.runner import PipelineRunner
from .pipeline.versioning import prompt_hash as _prompt_hash
from .pipeline.versioning import record_agent_version, record_prompt_version
from .utils import utc_now

logger = logging.getLogger(__name__)

# Fixed per spec §2 "Concurrency" — not configurable in Phase 1.
_CONCURRENCY = 3

_config: ReplayConfig | None = None


def configure(raw: dict | None) -> None:
    """Parse and install the replay.safe_agents allowlist from config.yaml's
    `replay:` block. None/empty means replay is unconfigured — every request
    403s (see ReplayNotConfigured)."""
    global _config
    _config = ReplayConfig.model_validate(raw) if raw else None


def get_config() -> ReplayConfig | None:
    return _config


class ReplayNotConfigured(Exception):
    """No replay.safe_agents configured at all — main.py maps this to 403."""


class AgentNotAllowlisted(Exception):
    """The recorded or candidate agent isn't in replay.safe_agents — main.py
    maps this to 403."""

    def __init__(self, agent_key: str):
        super().__init__(f"agent '{agent_key}' is not in replay.safe_agents")
        self.agent_key = agent_key


class CandidateSpec(BaseModel):
    model: str | None = None
    agent: str | None = None
    prompt_template: str | None = None


class BucketSelector(BaseModel):
    agent: str | None = None
    model: str | None = None
    provider: str | None = None
    prompt_hash: str | None = None
    agent_version: str | None = None


class ReplayRequest(BaseModel):
    bucket: BucketSelector | Literal["current"] = "current"
    candidate: CandidateSpec
    mode: Literal["rendered", "rerender"]
    k: int = Field(default=20, ge=1, le=100)


@dataclass
class BucketKey:
    agent: str | None
    model: str | None
    provider: str | None
    prompt_hash: str | None
    agent_version: str | None


@dataclass
class SampleRow:
    id: str
    run_id: str
    step_index: int
    step_name: str
    executor: str
    agent: str | None
    model: str | None
    provider: str | None
    prompt_hash: str | None
    agent_version: str | None
    prompt: str
    raw_output: str | None
    executed_at: Any
    label: float
    label_source: str


def agent_key(executor: str, agent: str | None) -> str:
    """Same "executor:agent" format PipelineStep.agent already uses."""
    return f"{executor}:{agent}" if agent else executor


def is_allowlisted(cfg: ReplayConfig | None, *keys: str) -> bool:
    if not cfg or not cfg.safe_agents:
        return False
    allowed = set(cfg.safe_agents)
    return all(k in allowed for k in keys)


def render_verbatim(text: str) -> str:
    """A Jinja2 template string that renders back to `text` exactly, regardless
    of context. Used by `rendered` mode so a recorded prompt is resent as-is
    even if it happens to contain literal {{ }} / {% %} (e.g. a JSON example in
    the prompt body) — those must not be re-interpreted as template syntax the
    second time around.

    Builds the substitution in a single linear pass (not two sequential
    str.replace calls) — the replacement text itself contains braces, and a
    second .replace() over the first's output would corrupt those too.
    """
    out = []
    for ch in text:
        if ch == "{":
            out.append("{{ '{' }}")
        elif ch == "}":
            out.append("{{ '}' }}")
        else:
            out.append(ch)
    return "".join(out)


def find_step_config(pipeline_registry: dict[str, PipelineConfig], step_name: str) -> StepConfig | None:
    """First sequential (non-parallel, non-fan-out) step named `step_name`
    across every currently loaded pipeline — same "sequential steps only"
    scope main.py's /runs/{run_id}/rerun already uses. Supplies the candidate's
    deterministic_checks/timeout/retry — the parts of the step that aren't
    "the variable under test" and so aren't something the replay request
    overrides."""
    for pipeline in pipeline_registry.values():
        for step in pipeline.steps:
            if isinstance(step, StepConfig) and step.name == step_name:
                return step
    return None


def rendered_prompt_recoverable(sample: SampleRow) -> str | None:
    """The exact rendered primary prompt, if the executor that produced this
    sample stashed it (executors/gateway.py and executors/openclaw_ws.py both
    do, via raw_response["prompt"]). None means genuinely not recoverable —
    e.g. a row recorded before that stashing existed — and `rendered` mode
    must treat the sample as unreplayable rather than guessing from the
    fallback executor_config JSON dump PipelineStep.prompt holds instead."""
    if not sample.raw_output:
        return None
    try:
        raw = json.loads(sample.raw_output)
    except (TypeError, json.JSONDecodeError):
        return None
    prompt = raw.get("prompt") if isinstance(raw, dict) else None
    return prompt if isinstance(prompt, str) and prompt else None


def minimal_ctx(batch_run_id: str, step_name: str) -> dict:
    """`rendered` mode needs no reconstructed context — the prompt text is
    resent verbatim — but the executor still renders session_key against
    *some* context, so this supplies just enough for the default
    "pipeline:{{pipeline_run_id}}:{{step_name}}"-shaped templates to resolve.
    Setting pipeline_run_id to the new batch run's id (never the original
    run's) is what keeps a replayed session from reusing — and so being
    polluted by — the recorded run's own conversation history."""
    return {
        "pipeline_run_id": batch_run_id,
        "pipeline_name": None,
        "current_step": step_name,
        "labels": {},
        "team": None,
        "_testing": True,
        "steps": {},
    }


def build_candidate_step(
    base_step: StepConfig,
    bucket: BucketKey,
    candidate: CandidateSpec,
    mode: str,
    recorded_prompt: str | None,
) -> StepConfig:
    """The candidate configuration under test = the recorded bucket's
    agent/model, overridden by whatever the candidate spec set, plus the
    *current* pipeline's deterministic_checks/timeout/retry from base_step.
    Executor itself is never overridable — only model/agent/prompt_template
    are (spec §2). verifier/grounding are deliberately never copied onto the
    candidate: replay never calls the runner methods that would use them, but
    leaving them unset here documents that "bare primary call only" is by
    construction, not by omission."""
    executor, _, recorded_agent_name = (bucket.agent or "").partition(":")
    executor = executor or base_step.executor
    resolved_agent = candidate.agent or recorded_agent_name or None
    resolved_model = candidate.model or bucket.model

    executor_config = dict(base_step.executor_config)
    if resolved_agent:
        executor_config["agent"] = resolved_agent
    if resolved_model:
        executor_config["model"] = resolved_model

    if mode == "rendered":
        prompt_template = render_verbatim(recorded_prompt or "")
    else:
        prompt_template = candidate.prompt_template or base_step.prompt_template

    return StepConfig(
        name=base_step.name,
        executor=executor,
        executor_config=executor_config,
        prompt_template=prompt_template,
        timeout_seconds=base_step.timeout_seconds,
        retry=base_step.retry,
        deterministic_checks=base_step.deterministic_checks,
    )


async def resolve_bucket_key(
    session_factory: async_sessionmaker,
    step_name: str,
    selector: BucketSelector | Literal["current"],
) -> BucketKey:
    """"current" resolves to whatever (agent, model, provider, prompt_hash,
    agent_version) the single most recent production execution of step_name
    landed in — i.e. whatever bucket production traffic is earning trust in
    right now. An explicit BucketSelector is used as-is: a None field is a
    real NULL-dimension match, never a wildcard, same rule
    calibration.py's bucket key follows."""
    if selector != "current":
        return BucketKey(
            agent=selector.agent, model=selector.model, provider=selector.provider,
            prompt_hash=selector.prompt_hash, agent_version=selector.agent_version,
        )
    async with session_factory() as session:
        row = (await session.execute(
            select(
                PipelineStep.agent, PipelineStep.model, PipelineStep.provider,
                PipelineStep.prompt_hash, PipelineStep.agent_version,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(
                PipelineRun.stage == "production",
                or_(PipelineStep.step_name == step_name, PipelineStep.step_name.like(f"{step_name}/%")),
            )
            .order_by(PipelineStep.executed_at.desc())
            .limit(1)
        )).first()
    if row is None:
        raise ValueError(
            f"no production executions of step '{step_name}' found — cannot resolve 'current' bucket"
        )
    agent, model, provider, p_hash, a_version = row
    return BucketKey(agent=agent, model=model, provider=provider, prompt_hash=p_hash, agent_version=a_version)


async def select_samples(
    session_factory: async_sessionmaker, step_name: str, bucket: BucketKey, k: int,
) -> tuple[list[SampleRow], dict[str, int]]:
    """Most recent K labelled, stage=production executions of `bucket`, most
    recent first — labelled-only because the comparison needs ground truth on
    the recorded side (spec §2). Returns (samples, distribution) where
    distribution is {"correct"|"partial"|"incorrect": n} over exactly the
    returned samples — "the bucket's existing numbers" for this batch."""
    conditions = [
        PipelineRun.stage == "production",
        or_(PipelineStep.step_name == step_name, PipelineStep.step_name.like(f"{step_name}/%")),
        PipelineStep.agent == bucket.agent if bucket.agent is not None else PipelineStep.agent.is_(None),
        PipelineStep.model == bucket.model if bucket.model is not None else PipelineStep.model.is_(None),
        PipelineStep.provider == bucket.provider if bucket.provider is not None else PipelineStep.provider.is_(None),
        PipelineStep.prompt_hash == bucket.prompt_hash if bucket.prompt_hash is not None else PipelineStep.prompt_hash.is_(None),
        PipelineStep.agent_version == bucket.agent_version if bucket.agent_version is not None else PipelineStep.agent_version.is_(None),
    ]
    async with session_factory() as session:
        rows = (await session.execute(
            select(
                PipelineStep.id, PipelineStep.run_id, PipelineStep.step_index, PipelineStep.step_name,
                PipelineStep.executor, PipelineStep.agent, PipelineStep.model, PipelineStep.provider,
                PipelineStep.prompt_hash, PipelineStep.agent_version, PipelineStep.prompt,
                PipelineStep.raw_output, PipelineStep.executed_at, PipelineStep.deterministic_passed,
                StepFeedback.outcome, RunFeedback.outcome,
            )
            .select_from(PipelineStep)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .outerjoin(StepFeedback, StepFeedback.step_id == PipelineStep.id)
            .outerjoin(RunFeedback, RunFeedback.run_id == PipelineStep.run_id)
            .where(*conditions)
            .order_by(PipelineStep.executed_at.desc())
        )).all()

    _label_to_outcome = {1.0: "correct", 0.5: "partial", 0.0: "incorrect"}
    samples: list[SampleRow] = []
    distribution = {"correct": 0, "partial": 0, "incorrect": 0}
    for (id_, run_id, step_index, s_name, executor, agent, model, provider, p_hash, a_version,
         prompt, raw_output, executed_at, det_passed, step_outcome, run_outcome) in rows:
        if len(samples) >= k:
            break
        resolved = resolve_label(step_outcome, det_passed, run_outcome)
        if resolved is None:
            continue
        label, source = resolved
        samples.append(SampleRow(
            id=id_, run_id=run_id, step_index=step_index, step_name=s_name, executor=executor,
            agent=agent, model=model, provider=provider, prompt_hash=p_hash, agent_version=a_version,
            prompt=prompt, raw_output=raw_output, executed_at=executed_at,
            label=label, label_source=source,
        ))
        distribution[_label_to_outcome.get(label, "partial")] += 1
    return samples, distribution


async def run_replay_batch(
    session_factory: async_sessionmaker,
    runner: PipelineRunner,
    step_name: str,
    request: ReplayRequest,
    gateway_rest_url: str | None = None,
) -> str:
    """Orchestrates one replay batch end to end and returns the synthetic run
    (batch) id. Raises ReplayNotConfigured / AgentNotAllowlisted (-> 403) or
    ValueError (-> 422) for request-shape problems; per-sample failures never
    raise — they're recorded in the descriptor as unreplayable instead.

    Phase 2 hook (not built): a provider-cost preview of the batch before
    launch is straightforward once `request.candidate.model` and the sample
    count are known — resolve_rate/resolve_approx_rate (pricing.py/
    live_pricing.py) already do this per-step; the only new piece would be an
    average-tokens-per-sample estimate to multiply by. Belongs right here,
    before sample execution starts.
    """
    cfg = get_config()
    if not cfg or not cfg.safe_agents:
        raise ReplayNotConfigured()

    pipeline_registry = runner._pipeline_registry or {}
    bucket = await resolve_bucket_key(session_factory, step_name, request.bucket)
    executor_from_bucket, _, recorded_agent_name = (bucket.agent or "").partition(":")

    recorded_agent_key = bucket.agent or ""
    candidate_agent_name = request.candidate.agent or recorded_agent_name or None
    candidate_agent_key = (
        agent_key(executor_from_bucket, candidate_agent_name) if executor_from_bucket else ""
    )

    allowed = set(cfg.safe_agents)
    for key in (recorded_agent_key, candidate_agent_key):
        if key not in allowed:
            raise AgentNotAllowlisted(key)

    if request.mode == "rendered" and request.candidate.prompt_template:
        raise ValueError(
            "mode 'rendered' resends the recorded prompt verbatim — "
            "candidate.prompt_template must not be set (use mode 'rerender' instead)"
        )

    base_step = find_step_config(pipeline_registry, step_name)
    if base_step is None:
        raise ValueError(f"step '{step_name}' is not currently defined in any loaded pipeline")

    samples, distribution = await select_samples(session_factory, step_name, bucket, request.k)

    batch_run_id = str(uuid.uuid4())
    descriptor: dict[str, Any] = {
        "step_name": step_name,
        "source_bucket": {
            "agent": bucket.agent, "model": bucket.model, "provider": bucket.provider,
            "prompt_hash": bucket.prompt_hash, "agent_version": bucket.agent_version,
        },
        "candidate": request.candidate.model_dump(exclude_none=True),
        "mode": request.mode,
        "k": request.k,
        "recorded_labels": {s.id: {"label": s.label, "source": s.label_source} for s in samples},
        "distribution": distribution,
        "unreplayable": [],
        "replayed": {},
    }
    await _create_batch_run(session_factory, batch_run_id, step_name, descriptor)

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded(sample: SampleRow, index: int) -> None:
        async with semaphore:
            await _process_sample(
                session_factory, runner, batch_run_id, index, base_step, bucket,
                request.candidate, request.mode, sample, descriptor, gateway_rest_url,
            )

    await asyncio.gather(*(_bounded(s, i) for i, s in enumerate(samples)))
    await _finalise_batch_run(session_factory, batch_run_id, descriptor)
    return batch_run_id


async def _process_sample(
    session_factory: async_sessionmaker,
    runner: PipelineRunner,
    batch_run_id: str,
    index: int,
    base_step: StepConfig,
    bucket: BucketKey,
    candidate: CandidateSpec,
    mode: str,
    sample: SampleRow,
    descriptor: dict,
    gateway_rest_url: str | None,
) -> None:
    try:
        if mode == "rendered":
            recorded_prompt = rendered_prompt_recoverable(sample)
            if recorded_prompt is None:
                descriptor["unreplayable"].append({
                    "sample_step_id": sample.id,
                    "reason": (
                        "recorded rendered prompt is not recoverable for this sample "
                        "(recorded before rendered-prompt persistence existed for this executor)"
                    ),
                })
                return
            candidate_step = build_candidate_step(base_step, bucket, candidate, mode, recorded_prompt)
            ctx = minimal_ctx(batch_run_id, sample.step_name)
        else:
            try:
                ctx = await build_replay_context(
                    session_factory, runner._pipeline_registry or {}, sample.run_id, sample.step_name,
                    artifact_store=runner._artifact_store,
                )
            except ContextReconstructionError as exc:
                descriptor["unreplayable"].append({"sample_step_id": sample.id, "reason": str(exc)})
                return
            ctx["pipeline_run_id"] = batch_run_id
            candidate_step = build_candidate_step(base_step, bucket, candidate, mode, None)

        try:
            output: LLMOutput | None = await runner.execute_candidate(candidate_step, ctx)
            error: str | None = None
        except Exception as exc:
            logger.warning("replay: candidate execution failed for sample %s: %s", sample.id, exc)
            output, error = None, str(exc)

        d_check_results: list[dict] = []
        if output is not None and candidate_step.deterministic_checks:
            d_check_results = await runner.run_deterministic_checks(candidate_step, ctx)

        step_id = await _save_replay_step(
            session_factory, batch_run_id, index, candidate_step, sample,
            output=output, d_check_results=d_check_results, error=error,
            gateway_rest_url=gateway_rest_url,
        )
        descriptor["replayed"][sample.id] = step_id
    except Exception:
        logger.exception("replay: unexpected error processing sample %s", sample.id)
        descriptor["unreplayable"].append({
            "sample_step_id": sample.id, "reason": "internal error — see server logs",
        })


async def _create_batch_run(
    session_factory: async_sessionmaker, run_id: str, step_name: str, descriptor: dict,
) -> None:
    async with session_factory() as session:
        session.add(PipelineRun(
            id=run_id,
            pipeline_name=f"replay:{step_name}",
            source="replay",
            status="running",
            normalised_context="{}",
            raw_payload="{}",
            stage="testing",
            replay_of=json.dumps(descriptor),
        ))
        await session.commit()


async def _finalise_batch_run(session_factory: async_sessionmaker, run_id: str, descriptor: dict) -> None:
    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        run.replay_of = json.dumps(descriptor)
        run.status = "completed"
        run.completed_at = utc_now()
        await session.commit()


async def _save_replay_step(
    session_factory: async_sessionmaker,
    run_id: str,
    step_index: int,
    candidate_step: StepConfig,
    sample: SampleRow,
    *,
    output: LLMOutput | None,
    d_check_results: list[dict],
    error: str | None,
    gateway_rest_url: str | None,
) -> str:
    """Persists the candidate execution as an ordinary PipelineStep row on the
    batch's synthetic run — same shape _db_save_step (runner.py) writes, so
    every existing steps/marking-queue/insights surface that reads
    pipeline_steps sees a replay step exactly like any other, distinguished
    only by its run's stage='testing' + replay_of."""
    step_id = str(uuid.uuid4())
    det_passed = all(r["passed"] for r in d_check_results) if d_check_results else None
    p_hash = _prompt_hash(candidate_step.prompt_template)
    _agent = candidate_step.executor_config.get("agent")
    agent_key_ = f"{candidate_step.executor}:{_agent}" if _agent else None
    prompt_text = (
        (output.raw_response or {}).get("prompt", candidate_step.prompt_template)
        if output else candidate_step.prompt_template
    )

    async with session_factory() as session:
        session.add(PipelineStep(
            id=step_id,
            run_id=run_id,
            step_name=sample.step_name,
            step_index=step_index,
            executor=candidate_step.executor,
            agent=agent_key_,
            model=output.model if output else candidate_step.executor_config.get("model"),
            provider=output.provider if output else None,
            prompt_hash=p_hash,
            agent_version=output.agent_version if output else None,
            prompt=prompt_text,
            raw_output=json.dumps(output.raw_response) if output else (json.dumps({"error": error}) if error else None),
            parsed_output=output.model_dump_json(exclude={"raw_response"}) if output else None,
            status="failed" if error else "completed",
            primary_confidence=output.confidence if output else None,
            effective_confidence=output.confidence if output else None,
            deterministic_passed=det_passed,
            executed_at=utc_now(),
        ))
        if p_hash is not None:
            await record_prompt_version(
                session, hash_=p_hash, step_name=sample.step_name, template=candidate_step.prompt_template,
            )
        if gateway_rest_url and output and output.agent_version is not None and agent_key_ is not None:
            await record_agent_version(
                session, gateway_rest_url, agent_version=output.agent_version, agent=agent_key_,
            )
        await session.commit()
    return step_id


def compute_replay_rollup(
    descriptor: dict, candidate_rows: list[dict], marks: dict[str, str],
) -> dict:
    """Pure — no DB, no I/O. `candidate_rows` is
    [{"id", "primary_confidence", "deterministic_passed", "status", "summary"}]
    for the batch's persisted PipelineStep rows; `marks` is
    {candidate_step_id: outcome} from StepFeedback. Recomputed fresh on every
    report request (same pattern as calibration) — no persisted report
    artifact, so a mark submitted a second ago is already reflected.

    Candidate grading reuses resolve_label exactly — the same precedence chain
    (human mark > deterministic failure > nothing) production calibration
    uses, so "D-check failure auto-labels 0.0, a pass leaves the sample
    unmarked" falls out of that shared function rather than being
    reimplemented here.
    """
    recorded_labels: dict[str, dict] = descriptor["recorded_labels"]
    recorded_values = [v["label"] for v in recorded_labels.values()]
    recorded_accuracy = sum(recorded_values) / len(recorded_values) if recorded_values else None

    by_id = {r["id"]: r for r in candidate_rows}
    replayed: dict[str, str] = descriptor["replayed"]
    unreplayable_by_sample = {u["sample_step_id"]: u["reason"] for u in descriptor["unreplayable"]}

    rows: list[dict] = []
    graded_values: list[float] = []
    confidence_distribution: list[float] = []
    for sample_id, rec in recorded_labels.items():
        row: dict[str, Any] = {
            "sample_step_id": sample_id,
            "recorded_label": rec["label"],
            "recorded_label_source": rec["source"],
        }
        candidate_step_id = replayed.get(sample_id)
        if candidate_step_id is None:
            row["status"] = "unreplayable"
            row["unreplayable_reason"] = unreplayable_by_sample.get(sample_id, "unknown")
            rows.append(row)
            continue

        cand = by_id.get(candidate_step_id, {})
        row["candidate_step_id"] = candidate_step_id
        row["status"] = cand.get("status")
        row["candidate_confidence"] = cand.get("primary_confidence")
        row["candidate_summary"] = cand.get("summary")
        row["deterministic_passed"] = cand.get("deterministic_passed")
        mark_outcome = marks.get(candidate_step_id)
        row["mark_outcome"] = mark_outcome

        resolved = resolve_label(mark_outcome, cand.get("deterministic_passed"), None)
        if resolved is not None:
            row["candidate_label"], row["candidate_label_source"] = resolved
            graded_values.append(resolved[0])
        if cand.get("primary_confidence") is not None:
            confidence_distribution.append(cand["primary_confidence"])
        rows.append(row)

    candidate_accuracy_so_far = sum(graded_values) / len(graded_values) if graded_values else None

    return {
        "step_name": descriptor["step_name"],
        "mode": descriptor["mode"],
        "source_bucket": descriptor["source_bucket"],
        "candidate": descriptor["candidate"],
        "k": descriptor["k"],
        "recorded_distribution": descriptor["distribution"],
        "recorded_accuracy": recorded_accuracy,
        "candidate_accuracy_so_far": candidate_accuracy_so_far,
        "candidate_graded_n": len(graded_values),
        "candidate_total_n": len(recorded_labels),
        "unreplayable_count": len(descriptor["unreplayable"]),
        "confidence_distribution": confidence_distribution,
        "rows": rows,
    }


async def get_report(session_factory: async_sessionmaker, run_id: str) -> dict:
    """Thin I/O wrapper around compute_replay_rollup — loads the batch run's
    descriptor + its persisted candidate steps + any marks that have arrived,
    then delegates all the math."""
    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        if run is None or not run.replay_of:
            raise ValueError(f"replay run '{run_id}' not found")
        descriptor = json.loads(run.replay_of)

        step_rows = (await session.execute(
            select(
                PipelineStep.id, PipelineStep.primary_confidence, PipelineStep.deterministic_passed,
                PipelineStep.status, PipelineStep.parsed_output,
            ).where(PipelineStep.run_id == run_id)
        )).all()

        candidate_rows: list[dict] = []
        for id_, confidence, det_passed, status, parsed_output in step_rows:
            summary = None
            if parsed_output:
                try:
                    summary = json.loads(parsed_output).get("summary")
                except json.JSONDecodeError:
                    summary = None
            candidate_rows.append({
                "id": id_, "primary_confidence": confidence, "deterministic_passed": det_passed,
                "status": status, "summary": summary,
            })

        marks: dict[str, str] = {}
        step_ids = [r["id"] for r in candidate_rows]
        if step_ids:
            mark_rows = (await session.execute(
                select(StepFeedback.step_id, StepFeedback.outcome).where(StepFeedback.step_id.in_(step_ids))
            )).all()
            marks = dict(mark_rows)

    return compute_replay_rollup(descriptor, candidate_rows, marks)
