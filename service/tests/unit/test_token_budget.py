"""Tests for token extraction, storage, and budget guardrail."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.models.pipeline import BudgetConfig, PipelineConfig, StepConfig, TriggerConfig
from src.models.context import NormalisedContext
from src.models.llm import LLMOutput
from src.pipeline.runner import PipelineRunner, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_normalised(**kwargs) -> NormalisedContext:
    defaults = dict(
        source="test", pipeline="test-pipeline", severity="warning",
        summary="test alert", labels={}, metadata={}, raw={},
        received_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    defaults.update(kwargs)
    return NormalisedContext(**defaults)


def _make_output(confidence=0.9, input_tokens=100, output_tokens=50) -> LLMOutput:
    meta = {"agentMeta": {"model": "claude-sonnet", "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}}
    return LLMOutput(
        confidence=confidence, summary="ok", next_step_context="",
        raw_response={"meta": meta},
    )


def _make_output_no_tokens(confidence=0.9) -> LLMOutput:
    return LLMOutput(confidence=confidence, summary="ok", next_step_context="", raw_response={})


def _make_runner(budget_max_tokens=None):
    pipeline = PipelineConfig(
        name="test",
        trigger=TriggerConfig(),
        steps=[],
        budget=BudgetConfig(max_tokens=budget_max_tokens) if budget_max_tokens else None,
    )
    runner = PipelineRunner(executors={}, session_factory=None)
    return runner, pipeline


# ---------------------------------------------------------------------------
# BudgetConfig model
# ---------------------------------------------------------------------------

def test_budget_config_default():
    b = BudgetConfig()
    assert b.max_tokens is None


def test_budget_config_explicit():
    b = BudgetConfig(max_tokens=50000)
    assert b.max_tokens == 50000


def test_pipeline_config_no_budget():
    p = PipelineConfig(name="x", trigger=TriggerConfig(), steps=[])
    assert p.budget is None


def test_pipeline_config_with_budget():
    p = PipelineConfig(
        name="x", trigger=TriggerConfig(), steps=[],
        budget={"max_tokens": 10000},
    )
    assert p.budget.max_tokens == 10000


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------

def test_extract_usage_present():
    raw = {"meta": {"agentMeta": {"usage": {"input_tokens": 200, "output_tokens": 80}}}}
    runner, _ = _make_runner()
    assert runner._extract_usage(raw) == (200, 80)


def test_extract_usage_missing_meta():
    runner, _ = _make_runner()
    assert runner._extract_usage({}) == (0, 0)


def test_extract_usage_empty_usage():
    raw = {"meta": {"agentMeta": {"usage": {}}}}
    runner, _ = _make_runner()
    assert runner._extract_usage(raw) == (0, 0)


def test_extract_usage_none_raw():
    runner, _ = _make_runner()
    assert runner._extract_usage(None) == (0, 0)


def test_extract_usage_partial_tokens():
    raw = {"meta": {"agentMeta": {"usage": {"input_tokens": 150}}}}
    runner, _ = _make_runner()
    assert runner._extract_usage(raw) == (150, 0)


# ---------------------------------------------------------------------------
# StepResult.total_tokens via runner
# ---------------------------------------------------------------------------

async def test_step_result_total_tokens_set():
    """Runner sets total_tokens on StepResult from executor output."""
    runner = PipelineRunner(executors={}, session_factory=None)

    async def fake_execute(step, ctx):
        return _make_output(input_tokens=300, output_tokens=120)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="s", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )
    assert result.total_tokens == 420  # 300 + 120


async def test_step_result_total_tokens_zero_when_no_usage():
    runner = PipelineRunner(executors={}, session_factory=None)

    async def fake_execute(step, ctx):
        return _make_output_no_tokens()

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="s", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )
    assert result.total_tokens == 0


# ---------------------------------------------------------------------------
# Budget guardrail
# ---------------------------------------------------------------------------

async def test_budget_not_exceeded_allows_pipeline_to_complete():
    call_count = 0

    async def fake_execute(step, ctx):
        nonlocal call_count
        call_count += 1
        return _make_output(input_tokens=100, output_tokens=50)

    runner = PipelineRunner(executors={}, session_factory=None)
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step1 = StepConfig(name="step1", executor="gateway", prompt_template="")
    step2 = StepConfig(name="step2", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), steps=[step1, step2],
        budget=BudgetConfig(max_tokens=1000),  # well above 300 total
    )

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r1")
    assert result.status == "completed"
    assert call_count == 2


async def test_budget_exceeded_aborts_pipeline():
    call_count = 0

    async def fake_execute(step, ctx):
        nonlocal call_count
        call_count += 1
        return _make_output(input_tokens=200, output_tokens=100)  # 300 per step

    runner = PipelineRunner(executors={}, session_factory=None)
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step1 = StepConfig(name="step1", executor="gateway", prompt_template="")
    step2 = StepConfig(name="step2", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), steps=[step1, step2],
        budget=BudgetConfig(max_tokens=299),  # 300 > 299 after step1
    )

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r1")
    assert result.status == "aborted"
    assert "budget exceeded" in result.abort_reason.lower()
    assert call_count == 1  # step2 never ran


async def test_budget_exceeded_logs_event():
    run_log: list = []

    async def fake_execute(step, ctx):
        return _make_output(input_tokens=500, output_tokens=200)

    runner = PipelineRunner(executors={}, session_factory=None)
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="s", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), steps=[step],
        budget=BudgetConfig(max_tokens=100),
    )

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r1")
    assert result.status == "aborted"
    # The run log for _db_complete_run collects events; check abort_reason instead
    assert "700" in result.abort_reason  # 500+200=700 tokens


async def test_budget_none_never_triggers():
    call_count = 0

    async def fake_execute(step, ctx):
        nonlocal call_count
        call_count += 1
        return _make_output(input_tokens=100000, output_tokens=100000)

    runner = PipelineRunner(executors={}, session_factory=None)
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step1 = StepConfig(name="s1", executor="gateway", prompt_template="")
    step2 = StepConfig(name="s2", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(
        name="p", trigger=TriggerConfig(), steps=[step1, step2],
        # no budget field
    )

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r1")
    assert result.status == "completed"
    assert call_count == 2
