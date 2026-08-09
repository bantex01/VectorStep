"""Tests for replay / shadow evaluation (SPEC-replay-shadow-eval.md).

Follows the house pattern from test_calibration.py / test_durable_runs.py:
async, sqlite fixture from conftest.py, hand-seeded DB rows, a fake executor
instead of a live one. Covers the 9 cases in the spec's §5.
"""
import json
from datetime import datetime, timedelta

import pytest
from jinja2 import Environment, Undefined

from src import metrics, replay
from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, RunFeedback, StepFeedback
from src.models.context import NormalisedContext
from src.models.llm import LLMOutput
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig
from src.pipeline.replay_context import ContextReconstructionError
from src.pipeline.runner import PipelineRunner
from src.pipeline.versioning import prompt_hash as prompt_hash_fn
from src.replay import (
    AgentNotAllowlisted,
    BucketKey,
    BucketSelector,
    CandidateSpec,
    ReplayNotConfigured,
    ReplayRequest,
    compute_replay_rollup,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RecordingExecutor:
    """Renders the same way gateway.py/openclaw_ws.py do, so a test can assert
    on the exact text a candidate call would have sent, and records every call
    for inspection."""

    def __init__(self, output_factory=None, raise_exc: Exception | None = None):
        self.calls: list[dict] = []
        self._output_factory = output_factory or (
            lambda: LLMOutput(confidence=0.8, summary="ok candidate", next_step_context="")
        )
        self._raise_exc = raise_exc

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        rendered = Environment(undefined=Undefined).from_string(step.prompt_template).render(**context)
        self.calls.append({"step": step, "ctx": dict(context), "rendered_prompt": rendered})
        if self._raise_exc:
            raise self._raise_exc
        return self._output_factory()


def _make_runner(executor, pipeline_registry: dict | None = None, executor_name: str = "gateway") -> PipelineRunner:
    return PipelineRunner(
        executors={executor_name: (lambda: executor)},
        session_factory=get_session_factory(),
        pipeline_registry=pipeline_registry or {},
    )


def _make_pipeline(
    pipeline_name: str, step_name: str, *, executor: str = "gateway", agent: str = "recorded-agent",
    prompt_template: str = "hello",
) -> PipelineConfig:
    return PipelineConfig(
        name=pipeline_name,
        trigger=TriggerConfig(),
        steps=[
            StepConfig(
                name=step_name, executor=executor,
                executor_config={"agent": agent}, prompt_template=prompt_template,
            ),
        ],
    )


async def _seed_run(
    run_id: str, *, pipeline_name: str = "p", stage: str = "production",
    normalised_context: str = "{}",
) -> None:
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name=pipeline_name, source="test",
            normalised_context=normalised_context, raw_payload="{}", stage=stage,
        ))
        await session.commit()


async def _seed_step(
    run_id: str, step_name: str, *, index: int = 0, agent: str | None = "gateway:recorded-agent",
    model: str | None = "old-model", provider: str | None = None,
    prompt_hash: str | None = None, agent_version: str | None = None,
    prompt: str = "recorded prompt text", raw_output: dict | None = None,
    parsed_output: dict | None = None, det_passed: bool | None = None,
    executed_at: datetime | None = None, executor: str = "gateway",
) -> str:
    sf = get_session_factory()
    async with sf() as session:
        step = PipelineStep(
            run_id=run_id, step_name=step_name, step_index=index, executor=executor,
            agent=agent, model=model, provider=provider, prompt=prompt, status="completed",
            prompt_hash=prompt_hash, agent_version=agent_version,
            raw_output=json.dumps(raw_output) if raw_output is not None else None,
            parsed_output=json.dumps(parsed_output) if parsed_output is not None else None,
            effective_confidence=0.9, primary_confidence=0.9, deterministic_passed=det_passed,
            executed_at=executed_at or datetime(2026, 1, 1, 12, 0, 0),
        )
        session.add(step)
        await session.flush()
        step_id = step.id
        await session.commit()
    return step_id


async def _mark_step(step_id: str, run_id: str, pipeline_name: str, step_name: str, outcome: str) -> None:
    sf = get_session_factory()
    async with sf() as session:
        session.add(StepFeedback(
            step_id=step_id, run_id=run_id, pipeline_name=pipeline_name, step_name=step_name, outcome=outcome,
        ))
        await session.commit()


async def _get_run(run_id: str) -> PipelineRun:
    sf = get_session_factory()
    async with sf() as session:
        return await session.get(PipelineRun, run_id)


async def _get_steps(run_id: str) -> list[PipelineStep]:
    from sqlalchemy import select
    sf = get_session_factory()
    async with sf() as session:
        rows = (await session.execute(
            select(PipelineStep).where(PipelineStep.run_id == run_id).order_by(PipelineStep.step_index)
        )).scalars().all()
        return list(rows)


_BUCKET = BucketKey(agent="gateway:recorded-agent", model="old-model", provider=None, prompt_hash=None, agent_version=None)


def _bucket_selector() -> BucketSelector:
    return BucketSelector(agent=_BUCKET.agent, model=_BUCKET.model, provider=None, prompt_hash=None, agent_version=None)


@pytest.fixture(autouse=True)
def _reset_replay_config():
    yield
    replay.configure(None)


# ---------------------------------------------------------------------------
# 1. Allowlist
# ---------------------------------------------------------------------------

async def test_allowlist_empty_config_rejects(db):
    replay.configure(None)
    runner = _make_runner(_RecordingExecutor())
    request = ReplayRequest(bucket=_bucket_selector(), candidate=CandidateSpec(model="new-model"), mode="rendered")
    with pytest.raises(ReplayNotConfigured):
        await replay.run_replay_batch(get_session_factory(), runner, "step-a", request)


async def test_allowlist_rejects_unlisted_recorded_agent(db):
    # Candidate is listed, recorded is not — still rejected (both directions).
    replay.configure({"safe_agents": ["gateway:candidate-ok"]})
    runner = _make_runner(_RecordingExecutor())
    bucket = BucketSelector(agent="gateway:recorded-not-listed", model="old-model")
    request = ReplayRequest(bucket=bucket, candidate=CandidateSpec(agent="candidate-ok"), mode="rendered")
    with pytest.raises(AgentNotAllowlisted):
        await replay.run_replay_batch(get_session_factory(), runner, "step-a", request)


async def test_allowlist_rejects_unlisted_candidate_agent(db):
    # Recorded is listed, candidate is not — still rejected (both directions).
    replay.configure({"safe_agents": ["gateway:recorded-agent"]})
    runner = _make_runner(_RecordingExecutor())
    bucket = _bucket_selector()
    request = ReplayRequest(bucket=bucket, candidate=CandidateSpec(agent="candidate-not-listed"), mode="rendered")
    with pytest.raises(AgentNotAllowlisted):
        await replay.run_replay_batch(get_session_factory(), runner, "step-a", request)


# ---------------------------------------------------------------------------
# 2. Sample selection
# ---------------------------------------------------------------------------

async def test_select_samples_labelled_production_only_capped_with_distribution(db):
    step_name = "triage"
    base_time = datetime(2026, 1, 1, 12, 0, 0)

    # 3 labelled production rows matching the bucket (2 correct via feedback, 1
    # incorrect via deterministic failure), most recent last.
    for i, (outcome, det) in enumerate([("correct", None), ("correct", None), (None, False)]):
        run_id = f"prod-{i}"
        await _seed_run(run_id, stage="production")
        step_id = await _seed_step(
            run_id, step_name, agent=_BUCKET.agent, model=_BUCKET.model,
            det_passed=det, executed_at=base_time + timedelta(minutes=i),
        )
        if outcome:
            await _mark_step(step_id, run_id, "p", step_name, outcome)

    # An unlabelled production row matching the bucket — must be excluded.
    await _seed_run("prod-unlabelled", stage="production")
    await _seed_step(
        "prod-unlabelled", step_name, agent=_BUCKET.agent, model=_BUCKET.model,
        executed_at=base_time + timedelta(minutes=10),
    )

    # A labelled TESTING row matching the bucket — must be excluded.
    await _seed_run("testing-row", stage="testing")
    testing_step_id = await _seed_step(
        "testing-row", step_name, agent=_BUCKET.agent, model=_BUCKET.model,
        executed_at=base_time + timedelta(minutes=11),
    )
    await _mark_step(testing_step_id, "testing-row", "p", step_name, "correct")

    # A labelled production row with a DIFFERENT model — must be excluded (bucket match).
    await _seed_run("prod-other-model", stage="production")
    other_step_id = await _seed_step(
        "prod-other-model", step_name, agent=_BUCKET.agent, model="other-model",
        executed_at=base_time + timedelta(minutes=12),
    )
    await _mark_step(other_step_id, "prod-other-model", "p", step_name, "correct")

    samples, distribution = await replay.select_samples(get_session_factory(), step_name, _BUCKET, k=20)
    assert len(samples) == 3
    assert distribution == {"correct": 2, "partial": 0, "incorrect": 1}
    assert [s.run_id for s in samples] == ["prod-2", "prod-1", "prod-0"]  # most recent first

    # K cap
    capped, capped_dist = await replay.select_samples(get_session_factory(), step_name, _BUCKET, k=2)
    assert len(capped) == 2
    assert capped_dist == {"correct": 1, "partial": 0, "incorrect": 1}


# ---------------------------------------------------------------------------
# 3. rendered mode sends the recorded prompt verbatim
# ---------------------------------------------------------------------------

async def test_rendered_mode_sends_recorded_prompt_verbatim(db):
    step_name = "triage"
    verbatim_text = "Investigate {{not-a-real-var}} and {% not a tag %} please."

    await _seed_run("prod-0", stage="production")
    step_id = await _seed_step(
        "prod-0", step_name, agent=_BUCKET.agent, model=_BUCKET.model,
        raw_output={"prompt": verbatim_text}, det_passed=False,  # det failure => labelled
        executed_at=datetime(2026, 1, 1, 12, 0, 0),
    )

    replay.configure({"safe_agents": [_BUCKET.agent, "gateway:candidate-agent"]})
    executor = _RecordingExecutor()
    pipeline = _make_pipeline("p", step_name, agent="recorded-agent", prompt_template="ignored at request time")
    runner = _make_runner(executor, pipeline_registry={"p": pipeline})

    request = ReplayRequest(
        bucket=_bucket_selector(), candidate=CandidateSpec(agent="candidate-agent"), mode="rendered", k=20,
    )
    run_id = await replay.run_replay_batch(get_session_factory(), runner, step_name, request)

    assert len(executor.calls) == 1
    assert executor.calls[0]["rendered_prompt"] == verbatim_text

    steps = await _get_steps(run_id)
    assert len(steps) == 1
    assert steps[0].agent == "gateway:candidate-agent"


# ---------------------------------------------------------------------------
# 4. rerender mode + unreplayable counted, not dropped
# ---------------------------------------------------------------------------

async def test_rerender_mode_renders_against_reconstructed_context_and_counts_unreplayable(db):
    step_name = "verify"
    normalised = json.dumps({
        "pipeline": "p", "severity": "warning", "labels": {"service": "checkout"},
        "summary": "s", "raw": {}, "metadata": {},
    })
    await _seed_run("prod-0", pipeline_name="p", stage="production", normalised_context=normalised)
    # Prior step (index 0) whose output the candidate template references.
    await _seed_step(
        "prod-0", "investigate", index=0, agent=_BUCKET.agent, model=_BUCKET.model,
        parsed_output={"confidence": 0.9, "summary": "root cause found", "next_step_context": "",
                        "dashboard_uid": "abc123"},
        executed_at=datetime(2026, 1, 1, 11, 59, 0),
    )
    step_id = await _seed_step(
        "prod-0", step_name, index=1, agent=_BUCKET.agent, model=_BUCKET.model,
        det_passed=False, executed_at=datetime(2026, 1, 1, 12, 0, 0),
    )

    # Second sample: owned by a pipeline that is no longer loaded -> context
    # reconstruction fails (can't resolve context_template.include/vars).
    await _seed_run("prod-orphan-owner", pipeline_name="deleted-pipeline", stage="production")
    orphan_step_id = await _seed_step(
        "prod-orphan-owner", step_name, agent=_BUCKET.agent, model=_BUCKET.model,
        det_passed=False, executed_at=datetime(2026, 1, 1, 12, 1, 0),
    )

    replay.configure({"safe_agents": [_BUCKET.agent]})
    executor = _RecordingExecutor()
    pipeline = _make_pipeline("p", step_name, agent="recorded-agent")
    runner = _make_runner(executor, pipeline_registry={"p": pipeline})

    candidate_template = "svc={{ labels.service }} prior={{ steps.investigate.dashboard_uid }}"
    request = ReplayRequest(
        bucket=_bucket_selector(), candidate=CandidateSpec(prompt_template=candidate_template),
        mode="rerender", k=20,
    )
    run_id = await replay.run_replay_batch(get_session_factory(), runner, step_name, request)

    # Only the reconstructible sample actually executed.
    assert len(executor.calls) == 1
    assert executor.calls[0]["rendered_prompt"] == "svc=checkout prior=abc123"

    report = await replay.get_report(get_session_factory(), run_id)
    assert report["unreplayable_count"] == 1
    assert len(report["rows"]) == 2
    unreplayable_rows = [r for r in report["rows"] if r["status"] == "unreplayable"]
    assert len(unreplayable_rows) == 1
    assert unreplayable_rows[0]["sample_step_id"] == orphan_step_id


# ---------------------------------------------------------------------------
# 5. Synthetic run stage/replay_of + stats exclusion
# ---------------------------------------------------------------------------

async def test_synthetic_run_is_testing_stage_and_excluded_from_production_stats(db):
    step_name = "triage"
    await _seed_run("prod-0", stage="production")
    await _seed_step(
        "prod-0", step_name, agent=_BUCKET.agent, model=_BUCKET.model,
        raw_output={"prompt": "hi"}, det_passed=False,
    )

    replay.configure({"safe_agents": [_BUCKET.agent]})
    runner = _make_runner(_RecordingExecutor(), pipeline_registry={"p": _make_pipeline("p", step_name)})
    request = ReplayRequest(bucket=_bucket_selector(), candidate=CandidateSpec(model="new-model"), mode="rendered")
    run_id = await replay.run_replay_batch(get_session_factory(), runner, step_name, request)

    run = await _get_run(run_id)
    assert run.stage == "testing"
    assert run.replay_of is not None
    descriptor = json.loads(run.replay_of)
    assert descriptor["step_name"] == step_name

    data = await metrics.fetch_metrics_data(get_session_factory())
    # Excluded from every production-scoped series.
    assert not any(p == run.pipeline_name for p, *_ in data.run_counts)
    # Present in the replay-specific series.
    assert (step_name, 1) in data.replay_batches


# ---------------------------------------------------------------------------
# 6. D-check failure auto-labels 0.0; pass leaves unmarked (pure function)
# ---------------------------------------------------------------------------

def test_deterministic_failure_auto_labels_zero_pass_leaves_unmarked():
    descriptor = {
        "step_name": "s", "mode": "rendered", "source_bucket": {}, "candidate": {}, "k": 2,
        "recorded_labels": {
            "sample-fail": {"label": 1.0, "source": "human_step_feedback"},
            "sample-pass": {"label": 1.0, "source": "human_step_feedback"},
        },
        "distribution": {"correct": 2, "partial": 0, "incorrect": 0},
        "unreplayable": [],
        "replayed": {"sample-fail": "cand-fail", "sample-pass": "cand-pass"},
    }
    candidate_rows = [
        {"id": "cand-fail", "primary_confidence": 0.9, "deterministic_passed": False, "status": "completed", "summary": "x"},
        {"id": "cand-pass", "primary_confidence": 0.9, "deterministic_passed": True, "status": "completed", "summary": "y"},
    ]
    report = compute_replay_rollup(descriptor, candidate_rows, marks={})

    by_id = {r["sample_step_id"]: r for r in report["rows"]}
    assert by_id["sample-fail"]["candidate_label"] == 0.0
    assert by_id["sample-fail"]["candidate_label_source"] == "deterministic_failure"
    assert "candidate_label" not in by_id["sample-pass"]
    assert report["candidate_graded_n"] == 1
    assert report["candidate_accuracy_so_far"] == 0.0


# ---------------------------------------------------------------------------
# 7. Marks arriving move the report's candidate accuracy
# ---------------------------------------------------------------------------

def test_marks_move_candidate_accuracy_pure():
    descriptor = {
        "step_name": "s", "mode": "rendered", "source_bucket": {}, "candidate": {}, "k": 1,
        "recorded_labels": {"sample-1": {"label": 1.0, "source": "human_step_feedback"}},
        "distribution": {"correct": 1, "partial": 0, "incorrect": 0},
        "unreplayable": [], "replayed": {"sample-1": "cand-1"},
    }
    candidate_rows = [
        {"id": "cand-1", "primary_confidence": 0.6, "deterministic_passed": None, "status": "completed", "summary": "x"},
    ]

    before = compute_replay_rollup(descriptor, candidate_rows, marks={})
    assert before["candidate_accuracy_so_far"] is None

    after = compute_replay_rollup(descriptor, candidate_rows, marks={"cand-1": "partial"})
    assert after["candidate_accuracy_so_far"] == 0.5


async def test_marks_move_candidate_accuracy_via_get_report(db):
    step_name = "triage"
    await _seed_run("prod-0", stage="production")
    await _seed_step(
        "prod-0", step_name, agent=_BUCKET.agent, model=_BUCKET.model,
        raw_output={"prompt": "hi"}, det_passed=False,
    )

    replay.configure({"safe_agents": [_BUCKET.agent]})
    runner = _make_runner(_RecordingExecutor(), pipeline_registry={"p": _make_pipeline("p", step_name)})
    request = ReplayRequest(bucket=_bucket_selector(), candidate=CandidateSpec(model="new-model"), mode="rendered")
    run_id = await replay.run_replay_batch(get_session_factory(), runner, step_name, request)

    report_before = await replay.get_report(get_session_factory(), run_id)
    assert report_before["candidate_accuracy_so_far"] is None

    candidate_step_id = report_before["rows"][0]["candidate_step_id"]
    await _mark_step(candidate_step_id, run_id, f"replay:{step_name}", step_name, "correct")

    report_after = await replay.get_report(get_session_factory(), run_id)
    assert report_after["candidate_accuracy_so_far"] == 1.0


# ---------------------------------------------------------------------------
# 8. Candidate prompt_hash/agent_version stamped
# ---------------------------------------------------------------------------

async def test_candidate_prompt_hash_and_agent_version_stamped(db):
    step_name = "verify"
    normalised = json.dumps({"pipeline": "p", "labels": {}, "raw": {}, "metadata": {}})
    await _seed_run("prod-0", pipeline_name="p", stage="production", normalised_context=normalised)
    await _seed_step(
        "prod-0", step_name, agent=_BUCKET.agent, model=_BUCKET.model,
        prompt_hash="recorded-hash-value", det_passed=False,
    )

    replay.configure({"safe_agents": [_BUCKET.agent]})
    candidate_template = "a brand new candidate template"
    executor = _RecordingExecutor(
        output_factory=lambda: LLMOutput(confidence=0.7, summary="ok", next_step_context="", agent_version="v2"),
    )
    pipeline = _make_pipeline("p", step_name, prompt_template="original template")
    runner = _make_runner(executor, pipeline_registry={"p": pipeline})

    bucket = _bucket_selector()
    bucket.prompt_hash = "recorded-hash-value"
    request = ReplayRequest(
        bucket=bucket, candidate=CandidateSpec(prompt_template=candidate_template),
        mode="rerender", k=20,
    )
    run_id = await replay.run_replay_batch(get_session_factory(), runner, step_name, request)

    steps = await _get_steps(run_id)
    assert len(steps) == 1
    assert steps[0].prompt_hash == prompt_hash_fn(candidate_template)
    assert steps[0].prompt_hash != "recorded-hash-value"
    assert steps[0].agent_version == "v2"


# ---------------------------------------------------------------------------
# 9. Metrics increment
# ---------------------------------------------------------------------------

async def test_metrics_increment_on_replay_batch(db):
    step_name = "triage"
    await _seed_run("prod-0", stage="production")
    await _seed_step(
        "prod-0", step_name, agent=_BUCKET.agent, model=_BUCKET.model,
        raw_output={"prompt": "hi"}, det_passed=False,
    )

    before = await metrics.fetch_metrics_data(get_session_factory())
    assert (step_name, 1) not in before.replay_batches

    replay.configure({"safe_agents": [_BUCKET.agent]})
    runner = _make_runner(_RecordingExecutor(), pipeline_registry={"p": _make_pipeline("p", step_name)})
    request = ReplayRequest(bucket=_bucket_selector(), candidate=CandidateSpec(model="new-model"), mode="rendered")
    await replay.run_replay_batch(get_session_factory(), runner, step_name, request)

    after = await metrics.fetch_metrics_data(get_session_factory())
    assert (step_name, 1) in after.replay_batches
    assert (step_name, "completed", 1) in after.replay_steps
