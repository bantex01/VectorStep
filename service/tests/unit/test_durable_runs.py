"""Tests for durable runs — resume after restart (SPEC-durable-runs.md).

Two layers are tested separately, matching the actual split in the implementation:
  - database.py's sweep_and_partition_running_runs — the startup decision of WHETHER
    to resume a run (config fingerprint, max age) vs. mark it interrupted as always.
  - PipelineRunner.run(..., resume=True) — the execution of an already-decided
    resume: a 'running' row is pre-inserted directly (as if left behind by a crash)
    and run() is called with resume=True, bypassing the sweep.

Follows the house test style (async, sqlite fixture from conftest.py, no live
executors — fake executor pattern from test_loop.py/test_fan_out.py).
"""
import json
from datetime import datetime, timedelta

import pytest
from unittest.mock import MagicMock

from src import metrics
from src.db.database import (
    get_pending_approval,
    get_session_factory,
    save_pending_approval,
    sweep_and_partition_running_runs,
)
from src.db.models import PipelineRun, PipelineStep
from src.executors import human
from src.models.context import NormalisedContext
from src.models.llm import LLMOutput
from src.models.pipeline import (
    BudgetConfig,
    DurableConfig,
    FanOutConfig,
    FanOutGroupConfig,
    ParallelGroupConfig,
    ParallelGroupInner,
    ParallelStepConfig,
    PipelineConfig,
    StepConfig,
    TriggerConfig,
    pipeline_config_fingerprint,
)
from src.pipeline.runner import PipelineRunner
from src.utils import utc_now


def _make_normalised(**kwargs) -> NormalisedContext:
    defaults = dict(
        source="test", pipeline="p", severity="warning", summary="alert",
        labels={}, metadata={}, raw={}, received_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    defaults.update(kwargs)
    return NormalisedContext(**defaults)


def _make_output(confidence: float = 0.9, **extra) -> LLMOutput:
    kwargs = dict(summary="ok", next_step_context="", raw_response={})
    kwargs.update(extra)
    return LLMOutput(confidence=confidence, **kwargs)


def _make_runner() -> PipelineRunner:
    return PipelineRunner(executors={}, session_factory=get_session_factory())


async def _insert_running_row(
    pipeline: PipelineConfig, normalised: NormalisedContext, run_id: str,
    triggered_at: datetime | None = None, config_fingerprint: str | None = "__auto__",
) -> None:
    sf = get_session_factory()
    fp = pipeline_config_fingerprint(pipeline) if config_fingerprint == "__auto__" else config_fingerprint
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name=pipeline.name, source=normalised.source,
            triggered_at=triggered_at or utc_now(), status="running",
            normalised_context=normalised.model_dump_json(), raw_payload="{}",
            config_fingerprint=fp, stage=pipeline.stage,
        ))
        await session.commit()


async def _insert_completed_step(
    run_id: str, step_name: str, step_index: int, output: LLMOutput,
    executor: str = "gateway", input_tokens: int | None = None,
    output_tokens: int | None = None, cost: float | None = None,
) -> None:
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineStep(
            run_id=run_id, step_name=step_name, step_index=step_index, executor=executor,
            prompt="{}", status="completed",
            parsed_output=output.model_dump_json(exclude={"raw_response"}),
            primary_confidence=output.confidence, effective_confidence=output.confidence,
            executed_at=utc_now(), input_tokens=input_tokens, output_tokens=output_tokens,
            cost=cost,
        ))
        await session.commit()


async def _get_run(run_id: str) -> PipelineRun:
    sf = get_session_factory()
    async with sf() as session:
        return await session.get(PipelineRun, run_id)


def _events(run: PipelineRun) -> list[str]:
    return [e["event"] for e in json.loads(run.logs)] if run.logs else []


# ---------------------------------------------------------------------------
# 1. Non-durable pipeline, running row at startup -> interrupted (regression)
# ---------------------------------------------------------------------------

async def test_non_durable_pipeline_marked_interrupted_unchanged(db):
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[StepConfig(name="s1", executor="gateway")])
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-1")

    resumable = await sweep_and_partition_running_runs([pipeline])

    assert resumable == []
    run = await _get_run("run-1")
    assert run.status == "interrupted"
    assert run.completed_at is not None
    assert _events(run) == ["run_interrupted"]


# ---------------------------------------------------------------------------
# 2. Durable, 2-of-4 steps persisted -> resume executes exactly steps 3-4
# ---------------------------------------------------------------------------

async def test_durable_resume_executes_only_remaining_steps(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=True,
        steps=[
            StepConfig(name="step_one", executor="gateway"),
            StepConfig(name="step_two", executor="gateway"),
            StepConfig(name="step_three", executor="gateway"),
            StepConfig(name="step_four", executor="gateway"),
        ],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-2")
    await _insert_completed_step("run-2", "step_one", 0, _make_output(0.9, custom_field="from-step-one"))
    await _insert_completed_step("run-2", "step_two", 1000, _make_output(0.9))

    executed: list[str] = []
    seen_ctx: dict[str, dict] = {}

    async def fake_execute(step, ctx):
        executed.append(step.name)
        seen_ctx[step.name] = ctx
        return _make_output(0.9)

    runner = _make_runner()
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    result = await runner.run(pipeline=pipeline, normalised=normalised, run_id="run-2", resume=True)

    assert result.status == "completed"
    assert executed == ["step_three", "step_four"]
    # Extra-field fidelity: step_three's context carries step_one's persisted output,
    # including its extra field, exactly as a fresh run would have built it.
    assert seen_ctx["step_three"]["steps"]["step_one"]["custom_field"] == "from-step-one"

    run = await _get_run("run-2")
    assert "run_resumed" in _events(run)
    resumed_event = next(e for e in json.loads(run.logs) if e["event"] == "run_resumed")
    assert resumed_event["steps_skipped"] == 2


# ---------------------------------------------------------------------------
# 3. on_interrupted: rerun vs escalate
# ---------------------------------------------------------------------------

async def test_on_interrupted_rerun_reexecutes_in_flight_step(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=True,
        steps=[StepConfig(name="step_one", executor="gateway"), StepConfig(name="step_two", executor="gateway")],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-3a")
    await _insert_completed_step("run-3a", "step_one", 0, _make_output(0.9))

    call_count = 0

    async def fake_execute(step, ctx):
        nonlocal call_count
        call_count += 1
        return _make_output(0.9)

    runner = _make_runner()
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    result = await runner.run(pipeline=pipeline, normalised=normalised, run_id="run-3a", resume=True)

    assert call_count == 1
    assert result.status == "completed"


async def test_on_interrupted_escalate_does_not_reexecute(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=DurableConfig(on_interrupted="escalate"),
        steps=[StepConfig(name="step_one", executor="gateway"), StepConfig(name="step_two", executor="gateway")],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-3b")
    await _insert_completed_step("run-3b", "step_one", 0, _make_output(0.9))

    call_count = 0

    async def fake_execute(step, ctx):
        nonlocal call_count
        call_count += 1
        return _make_output(0.9)

    runner = _make_runner()
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    result = await runner.run(pipeline=pipeline, normalised=normalised, run_id="run-3b", resume=True)

    assert call_count == 0
    assert result.status == "escalated"

    sf = get_session_factory()
    async with sf() as session:
        rows = (await session.execute(
            PipelineStep.__table__.select().where(PipelineStep.step_name == "step_two")
        )).all()
    assert len(rows) == 1
    assert rows[0].status == "escalated"


# ---------------------------------------------------------------------------
# 4. Config fingerprint mismatch -> interrupted + resume_skipped_config_changed
# ---------------------------------------------------------------------------

async def test_config_fingerprint_mismatch_skips_resume(db):
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), durable=True, steps=[StepConfig(name="s1", executor="gateway")])
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-4", config_fingerprint="stale-fingerprint")

    resumable = await sweep_and_partition_running_runs([pipeline])

    assert resumable == []
    run = await _get_run("run-4")
    assert run.status == "interrupted"
    assert "resume_skipped_config_changed" in _events(run)
    assert "run_interrupted" in _events(run)


# ---------------------------------------------------------------------------
# 5. max_resume_age_seconds exceeded -> interrupted
# ---------------------------------------------------------------------------

async def test_max_resume_age_exceeded_skips_resume(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=DurableConfig(max_resume_age_seconds=60),
        steps=[StepConfig(name="s1", executor="gateway")],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-5", triggered_at=utc_now() - timedelta(seconds=120))

    resumable = await sweep_and_partition_running_runs([pipeline])

    assert resumable == []
    run = await _get_run("run-5")
    assert run.status == "interrupted"
    assert "resume_skipped_max_age_exceeded" in _events(run)


async def test_max_resume_age_uses_service_default_when_pipeline_unset(db):
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), durable=True, steps=[StepConfig(name="s1", executor="gateway")])
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-5b", triggered_at=utc_now() - timedelta(seconds=120))

    resumable = await sweep_and_partition_running_runs([pipeline], default_max_resume_age_seconds=60)

    assert resumable == []
    run = await _get_run("run-5b")
    assert "resume_skipped_max_age_exceeded" in _events(run)


# ---------------------------------------------------------------------------
# 6. Token budget continuity — persisted + resumed tokens both count
# ---------------------------------------------------------------------------

async def test_budget_accumulator_counts_persisted_tokens(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=True,
        budget=BudgetConfig(max_tokens=1000),
        steps=[StepConfig(name="step_one", executor="gateway"), StepConfig(name="step_two", executor="gateway")],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-6")
    await _insert_completed_step(
        "run-6", "step_one", 0, _make_output(0.9), input_tokens=600, output_tokens=300,
    )

    async def fake_execute(step, ctx):
        return _make_output(
            0.9, raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 150, "output_tokens": 50}}}}
        )

    runner = _make_runner()
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    result = await runner.run(pipeline=pipeline, normalised=normalised, run_id="run-6", resume=True)

    # 900 (persisted) + 200 (resumed) = 1100 > 1000 -> budget trips on the resumed step.
    assert result.status == "aborted"
    assert "Token budget exceeded" in (result.abort_reason or "")


# ---------------------------------------------------------------------------
# 7. Parallel group partially persisted -> only missing branches re-run
# ---------------------------------------------------------------------------

async def test_parallel_group_partial_resume_runs_only_missing_branches(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=True,
        steps=[ParallelGroupConfig(parallel=ParallelGroupInner(
            name="triage", confidence_threshold=0.0, on_low_confidence="proceed",
            steps=[
                ParallelStepConfig(name="a", executor="gateway", prompt_template=""),
                ParallelStepConfig(name="b", executor="gateway", prompt_template=""),
                ParallelStepConfig(name="c", executor="gateway", prompt_template=""),
            ],
        ))],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-7")
    await _insert_completed_step("run-7", "triage/a", 0, _make_output(0.6))

    executed: list[str] = []

    async def fake_execute(step, ctx):
        executed.append(step.name)
        return _make_output(1.0)

    runner = _make_runner()
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    result = await runner.run(pipeline=pipeline, normalised=normalised, run_id="run-7", resume=True)

    assert sorted(executed) == ["b", "c"]
    assert result.status == "completed"
    group_result = result.steps[0]
    assert set(group_result.branch_outputs) == {"a", "b", "c"}
    # Join (all_must_pass -> min) considers the persisted branch's confidence too.
    assert group_result.effective_confidence == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 8. HITL re-arm — pending approval survives restart
# ---------------------------------------------------------------------------

async def test_hitl_pending_approval_rearms_and_completes_on_resume(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=True,
        steps=[StepConfig(name="approve", executor="human", prompt_template="Approve?")],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-8")
    await save_pending_approval(
        token="tok-8", run_id="run-8", step_name="approve", pipeline_name="p",
        message="Approve?", team=None, stage="production",
    )
    # Simulate a fresh process: no in-memory state survives a restart.
    human._pending_approvals.clear()
    human._pending_meta.clear()

    runner = _make_runner()
    runner._executor_instances["human"] = human.HumanExecutor()
    import asyncio
    task = asyncio.create_task(runner.run(pipeline=pipeline, normalised=normalised, run_id="run-8", resume=True))

    for _ in range(500):
        if "tok-8" in human._pending_approvals:
            break
        await asyncio.sleep(0.01)
    assert "tok-8" in human._pending_approvals, "resume did not re-arm the persisted token"

    assert human.resolve_approval("tok-8", True)
    result = await task

    assert result.status == "completed"
    assert await get_pending_approval("tok-8") is None


# ---------------------------------------------------------------------------
# 9. Fan-out reconstruction — partially persisted branches resume correctly
# ---------------------------------------------------------------------------

async def test_fan_out_partial_resume_runs_only_missing_branches(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=True,
        steps=[
            StepConfig(name="list_services", executor="gateway"),
            FanOutGroupConfig(fan_out=FanOutConfig(
                name="fo", executor="gateway", over="{{steps.list_services.services}}",
                confidence_threshold=0.0, on_low_confidence="proceed",
            )),
        ],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-9")
    await _insert_completed_step(
        "run-9", "list_services", 0, _make_output(0.9, services=["svc-a", "svc-b", "svc-c"]),
    )
    await _insert_completed_step("run-9", "fo/0", 1000, _make_output(0.9))

    executed_indices: list[int] = []

    async def fake_execute(step, ctx):
        executed_indices.append(ctx["fan_out_index"])
        return _make_output(0.9)

    runner = _make_runner()
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    result = await runner.run(pipeline=pipeline, normalised=normalised, run_id="run-9", resume=True)

    assert sorted(executed_indices) == [1, 2]
    assert result.status == "completed"
    fan_out_result = result.steps[0]
    assert set(fan_out_result.branch_outputs) == {"fo/0", "fo/1", "fo/2"}


# ---------------------------------------------------------------------------
# 10. Metrics — vectorstep_runs_resumed_total increments
# ---------------------------------------------------------------------------

async def test_metrics_runs_resumed_total_increments(db):
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), durable=True, stage="production",
        steps=[StepConfig(name="s1", executor="gateway")],
    )
    normalised = _make_normalised(pipeline="p")
    await _insert_running_row(pipeline, normalised, "run-10")

    async def fake_execute(step, ctx):
        return _make_output(0.9)

    runner = _make_runner()
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    await runner.run(pipeline=pipeline, normalised=normalised, run_id="run-10", resume=True)

    run = await _get_run("run-10")
    assert run.resumed_at is not None

    data = await metrics.fetch_metrics_data(get_session_factory())
    assert dict(data.runs_resumed).get("p") == 1
