"""Tests for refinement loop config and runner loop behaviour."""
import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock, MagicMock

from src.models.pipeline import LoopConfig, StepConfig
from src.models.llm import LLMOutput


# ---------------------------------------------------------------------------
# LoopConfig model
# ---------------------------------------------------------------------------

def test_loop_config_requires_confidence():
    with pytest.raises(ValidationError):
        LoopConfig()


def test_loop_config_defaults_max_iterations():
    cfg = LoopConfig(confidence=0.85)
    assert cfg.max_iterations == 3


def test_loop_config_explicit_values():
    cfg = LoopConfig(confidence=0.9, max_iterations=5)
    assert cfg.confidence == 0.9
    assert cfg.max_iterations == 5


def test_step_config_no_loop_by_default():
    step = StepConfig(name="x", executor="gateway")
    assert step.loop_until is None


def test_step_config_with_loop_until():
    step = StepConfig(
        name="refine",
        executor="gateway",
        loop_until={"confidence": 0.8, "max_iterations": 2},
    )
    assert step.loop_until is not None
    assert step.loop_until.confidence == 0.8
    assert step.loop_until.max_iterations == 2


# ---------------------------------------------------------------------------
# Runner loop behaviour (mocked executor)
# ---------------------------------------------------------------------------

def _make_runner():
    from src.pipeline.runner import PipelineRunner
    return PipelineRunner(executors={}, session_factory=None)


def _make_output(confidence: float) -> LLMOutput:
    return LLMOutput(
        confidence=confidence,
        summary="test",
        next_step_context="",
        raw_response={},
    )


def _make_pipeline(step: StepConfig):
    from src.models.pipeline import PipelineConfig, TriggerConfig
    return PipelineConfig(
        name="test-pipeline",
        trigger=TriggerConfig(),
        steps=[step],
    )


def _make_normalised():
    from src.models.context import NormalisedContext
    from datetime import datetime
    return NormalisedContext(
        source="test",
        pipeline="test-pipeline",
        summary="test alert",
        severity="warning",
        received_at=datetime(2026, 1, 1, 12, 0, 0),
        labels={},
        raw={},
    )


async def test_loop_converges_early(monkeypatch):
    """Runner exits the loop once effective_confidence >= loop_until.confidence."""
    outputs = [_make_output(0.5), _make_output(0.9)]
    call_idx = 0

    async def fake_execute(step, ctx):
        nonlocal call_idx
        out = outputs[call_idx]
        call_idx += 1
        return out

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(
        name="refine",
        executor="gateway",
        prompt_template="",
        loop_until=LoopConfig(confidence=0.8, max_iterations=3),
    )

    result = await runner._run_step_impl(
        step=step,
        index=0,
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert result.status == "completed"
    assert result.effective_confidence == 0.9
    assert call_idx == 2  # stopped after iteration 2


async def test_loop_exhausts_max_iterations(monkeypatch):
    """Runner stops at max_iterations even if confidence never reaches target."""
    async def fake_execute(step, ctx):
        return _make_output(0.4)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(
        name="refine",
        executor="gateway",
        prompt_template="",
        confidence_threshold=0.0,  # don't escalate so we get a result
        on_low_confidence="proceed",
        loop_until=LoopConfig(confidence=0.8, max_iterations=3),
    )

    call_count = 0
    original_execute = executor_instance.execute

    async def counting_execute(s, ctx):
        nonlocal call_count
        call_count += 1
        return await original_execute(s, ctx)

    executor_instance.execute = counting_execute

    result = await runner._run_step_impl(
        step=step,
        index=0,
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert call_count == 3
    assert result.effective_confidence == 0.4


async def test_loop_context_injected_on_iteration_two():
    """The {{loop}} dict is injected into ctx from iteration 2 onward."""
    received_contexts = []

    async def fake_execute(step, ctx):
        received_contexts.append(dict(ctx))
        conf = 0.5 if len(received_contexts) < 2 else 0.9
        return _make_output(conf)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(
        name="refine",
        executor="gateway",
        prompt_template="",
        loop_until=LoopConfig(confidence=0.8, max_iterations=3),
    )

    await runner._run_step_impl(
        step=step,
        index=0,
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert len(received_contexts) == 2
    iter1_ctx = received_contexts[0]
    iter2_ctx = received_contexts[1]

    # Iteration 1: loop dict is present but prior_output is None
    assert iter1_ctx["loop"]["iteration"] == 1
    assert iter1_ctx["loop"]["prior_output"] is None

    # Iteration 2: loop dict has prior confidence and output
    assert iter2_ctx["loop"]["iteration"] == 2
    assert iter2_ctx["loop"]["prior_confidence"] == 0.5
    assert iter2_ctx["loop"]["prior_output"]["summary"] == "test"


async def test_no_loop_config_runs_once():
    """Without loop_until, the step executes exactly once."""
    call_count = 0

    async def fake_execute(step, ctx):
        nonlocal call_count
        call_count += 1
        return _make_output(0.9)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="simple", executor="gateway", prompt_template="")

    await runner._run_step_impl(
        step=step,
        index=0,
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert call_count == 1


async def test_loop_run_log_records_convergence():
    """Run log gets a loop_converged event when the loop exits successfully."""
    outputs = [_make_output(0.4), _make_output(0.9)]
    call_idx = 0

    async def fake_execute(step, ctx):
        nonlocal call_idx
        out = outputs[call_idx]
        call_idx += 1
        return out

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(
        name="refine",
        executor="gateway",
        prompt_template="",
        loop_until=LoopConfig(confidence=0.8, max_iterations=3),
    )
    run_log: list = []

    await runner._run_step_impl(
        step=step,
        index=0,
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=run_log,
    )

    events = [e["event"] for e in run_log]
    assert "loop_converged" in events


async def test_loop_run_log_records_exhaustion():
    """Run log gets a loop_exhausted event when max_iterations is reached."""
    async def fake_execute(step, ctx):
        return _make_output(0.3)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(
        name="refine",
        executor="gateway",
        prompt_template="",
        confidence_threshold=0.0,
        on_low_confidence="proceed",
        loop_until=LoopConfig(confidence=0.8, max_iterations=2),
    )
    run_log: list = []

    await runner._run_step_impl(
        step=step,
        index=0,
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=run_log,
    )

    events = [e["event"] for e in run_log]
    assert "loop_exhausted" in events
    assert "loop_converged" not in events
