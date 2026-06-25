"""Tests for executor: pipeline (sub-pipeline calls)."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.executors.pipeline import PipelineExecutor
from src.models.context import NormalisedContext
from src.models.llm import LLMOutput
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig
from src.pipeline.runner import PipelineRunResult, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_normalised(**kwargs) -> NormalisedContext:
    defaults = dict(
        source="test",
        pipeline="parent-pipeline",
        severity="warning",
        summary="test alert",
        labels={"service": "api"},
        metadata={"region": "us-east-1"},
        raw={},
        received_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    defaults.update(kwargs)
    return NormalisedContext(**defaults)


def _make_sub_pipeline(name="sub-pipeline") -> PipelineConfig:
    return PipelineConfig(name=name, trigger=TriggerConfig(), steps=[])


def _make_output(confidence=0.9, summary="done") -> LLMOutput:
    return LLMOutput(confidence=confidence, summary=summary, next_step_context="ctx", raw_response={})


def _make_run_result(status="completed", final_output=None, abort_reason=None) -> PipelineRunResult:
    return PipelineRunResult(
        run_id="sub-run-1",
        pipeline_name="sub-pipeline",
        status=status,
        final_output=final_output,
        abort_reason=abort_reason,
    )


def _make_ctx(pipeline_name="sub-pipeline", normalised=None, runner=None, registry=None):
    normalised = normalised or _make_normalised()
    sub_pipeline = _make_sub_pipeline(pipeline_name)
    mock_runner = runner or MagicMock()
    mock_runner.run = AsyncMock(return_value=_make_run_result(final_output=_make_output()))
    reg = registry if registry is not None else {pipeline_name: sub_pipeline}
    return {
        "pipeline_run_id": "parent-run-1",
        "_pork_runner": mock_runner,
        "_pork_normalised": normalised,
        "_pork_registry": reg,
    }, mock_runner


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def test_pipeline_executor_missing_pipeline_name():
    ctx, _ = _make_ctx()
    step = StepConfig(name="s", executor="pipeline", executor_config={})
    with pytest.raises(ValueError, match="executor_config.pipeline is required"):
        await PipelineExecutor().execute(step, ctx)


async def test_pipeline_executor_unknown_pipeline():
    ctx, _ = _make_ctx(registry={})  # empty registry
    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "missing"})
    with pytest.raises(ValueError, match="not found in registry"):
        await PipelineExecutor().execute(step, ctx)


async def test_pipeline_executor_missing_registry():
    ctx = {
        "pipeline_run_id": "parent-run-1",
        "_pork_runner": MagicMock(),
        "_pork_normalised": _make_normalised(),
        # _pork_registry intentionally absent
    }
    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub"})
    with pytest.raises(RuntimeError, match="pipeline_registry"):
        await PipelineExecutor().execute(step, ctx)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_pipeline_executor_returns_final_output():
    final = _make_output(confidence=0.85, summary="sub done")
    ctx, mock_runner = _make_ctx()
    mock_runner.run = AsyncMock(return_value=_make_run_result(final_output=final))

    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub-pipeline"})
    result = await PipelineExecutor().execute(step, ctx)

    assert result.confidence == 0.85
    assert result.summary == "sub done"
    assert result.model_extra.get("sub_pipeline_status") == "completed"
    assert "sub_run_id" in result.model_extra


async def test_pipeline_executor_passes_parent_run_id():
    ctx, mock_runner = _make_ctx()
    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub-pipeline"})
    await PipelineExecutor().execute(step, ctx)

    call_kwargs = mock_runner.run.call_args.kwargs
    assert call_kwargs["parent_run_id"] == "parent-run-1"


async def test_pipeline_executor_sub_pipeline_name_in_call():
    ctx, mock_runner = _make_ctx()
    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub-pipeline"})
    await PipelineExecutor().execute(step, ctx)

    call_kwargs = mock_runner.run.call_args.kwargs
    assert call_kwargs["pipeline"].name == "sub-pipeline"


async def test_pipeline_executor_sub_normalised_source():
    ctx, mock_runner = _make_ctx()
    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub-pipeline"})
    await PipelineExecutor().execute(step, ctx)

    sub_normalised = mock_runner.run.call_args.kwargs["normalised"]
    assert sub_normalised.source == "sub-pipeline"
    assert sub_normalised.fingerprint is None  # dedup bypassed


async def test_pipeline_executor_failed_sub_pipeline():
    ctx, mock_runner = _make_ctx()
    mock_runner.run = AsyncMock(return_value=_make_run_result(
        status="failed", final_output=None, abort_reason="step error"
    ))

    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub-pipeline"})
    result = await PipelineExecutor().execute(step, ctx)

    assert result.confidence == 0.0
    assert "failed" in result.summary


async def test_pipeline_executor_completed_no_output():
    ctx, mock_runner = _make_ctx()
    mock_runner.run = AsyncMock(return_value=_make_run_result(status="completed", final_output=None))

    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub-pipeline"})
    result = await PipelineExecutor().execute(step, ctx)

    assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Context overrides
# ---------------------------------------------------------------------------

async def test_pipeline_executor_context_override_summary():
    normalised = _make_normalised(summary="original")
    ctx, mock_runner = _make_ctx(normalised=normalised)
    ctx["steps"] = {"triage": {"next_step_context": "focus on db layer"}}

    step = StepConfig(
        name="s",
        executor="pipeline",
        executor_config={
            "pipeline": "sub-pipeline",
            "context": {"summary": "{{ steps.triage.next_step_context }}"},
        },
    )
    await PipelineExecutor().execute(step, ctx)

    sub_normalised = mock_runner.run.call_args.kwargs["normalised"]
    assert sub_normalised.summary == "focus on db layer"


async def test_pipeline_executor_context_override_labels_merged():
    normalised = _make_normalised(labels={"service": "api", "env": "prod"})
    ctx, mock_runner = _make_ctx(normalised=normalised)

    step = StepConfig(
        name="s",
        executor="pipeline",
        executor_config={
            "pipeline": "sub-pipeline",
            "context": {"labels": {"team": "platform"}},
        },
    )
    await PipelineExecutor().execute(step, ctx)

    sub_normalised = mock_runner.run.call_args.kwargs["normalised"]
    # Parent labels preserved, new label added
    assert sub_normalised.labels["service"] == "api"
    assert sub_normalised.labels["env"] == "prod"
    assert sub_normalised.labels["team"] == "platform"


async def test_pipeline_executor_no_context_override_inherits_parent():
    normalised = _make_normalised(severity="critical", labels={"service": "db"})
    ctx, mock_runner = _make_ctx(normalised=normalised)

    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub-pipeline"})
    await PipelineExecutor().execute(step, ctx)

    sub_normalised = mock_runner.run.call_args.kwargs["normalised"]
    assert sub_normalised.severity == "critical"
    assert sub_normalised.labels["service"] == "db"


async def test_pipeline_executor_no_context_override_inherits_team():
    normalised = _make_normalised(team="payments")
    ctx, mock_runner = _make_ctx(normalised=normalised)

    step = StepConfig(name="s", executor="pipeline", executor_config={"pipeline": "sub-pipeline"})
    await PipelineExecutor().execute(step, ctx)

    sub_normalised = mock_runner.run.call_args.kwargs["normalised"]
    assert sub_normalised.team == "payments"


async def test_pipeline_executor_context_override_team():
    normalised = _make_normalised(team="payments")
    ctx, mock_runner = _make_ctx(normalised=normalised)

    step = StepConfig(
        name="s",
        executor="pipeline",
        executor_config={
            "pipeline": "sub-pipeline",
            "context": {"team": "platform"},
        },
    )
    await PipelineExecutor().execute(step, ctx)

    sub_normalised = mock_runner.run.call_args.kwargs["normalised"]
    assert sub_normalised.team == "platform"


# ---------------------------------------------------------------------------
# Runner context injection
# ---------------------------------------------------------------------------

async def test_runner_injects_pork_context_when_registry_set():
    """Runner should inject _pork_* keys into step context when registry is configured."""
    from src.pipeline.runner import PipelineRunner

    received_ctx: dict = {}

    async def fake_execute(step, ctx):
        received_ctx.update(ctx)
        return _make_output()

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    sub_pipeline = _make_sub_pipeline("my-sub")
    runner = PipelineRunner(
        executors={},
        session_factory=None,
        pipeline_registry={"my-sub": sub_pipeline},
    )
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="call", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(name="parent", trigger=TriggerConfig(), steps=[step])
    normalised = _make_normalised()

    await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=normalised,
        run_id="run-1", step_outputs={}, run_log=[],
    )

    assert "_pork_runner" in received_ctx
    assert "_pork_normalised" in received_ctx
    assert "_pork_registry" in received_ctx
    assert received_ctx["_pork_registry"] == {"my-sub": sub_pipeline}


async def test_runner_no_pork_context_without_registry():
    """Runner should not inject _pork_* keys when no pipeline_registry is set."""
    from src.pipeline.runner import PipelineRunner

    received_ctx: dict = {}

    async def fake_execute(step, ctx):
        received_ctx.update(ctx)
        return _make_output()

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = PipelineRunner(executors={}, session_factory=None)
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="s", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="run-1", step_outputs={}, run_log=[],
    )

    assert "_pork_runner" not in received_ctx
    assert "_pork_registry" not in received_ctx
