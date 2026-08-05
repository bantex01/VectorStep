"""Tests for cost persistence at step-save time (SPEC-cost-accounting.md §5):
_db_save_step writes priced cost, an openclaw-style step (no usage) writes NULL,
verifier tokens are captured and priced, and a pricing reload only affects
steps priced after the reload."""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src import pricing
from src.db.database import get_session_factory
from src.db.models import PipelineStep
from src.models.context import NormalisedContext
from src.models.llm import LLMOutput
from src.models.pipeline import (
    GroundingConfig,
    PipelineConfig,
    StepConfig,
    TriggerConfig,
    VerifierConfig,
    VerifierTriggerConfig,
)
from src.pipeline.runner import PipelineRunner


@pytest.fixture(autouse=True)
def _reset_pricing_table():
    original = pricing.get_table()
    yield
    pricing._table = original


def _make_normalised(**kwargs) -> NormalisedContext:
    defaults = dict(
        source="test", pipeline="test-pipeline", severity="warning",
        summary="test alert", labels={}, metadata={}, raw={},
        received_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    defaults.update(kwargs)
    return NormalisedContext(**defaults)


async def _get_step(step_name: str) -> PipelineStep:
    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            PipelineStep.__table__.select().where(PipelineStep.step_name == step_name)
        )
        return rows.one()


async def test_priced_step_writes_cost(db):
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })

    async def fake_execute(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="ok", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 1000, "output_tokens": 500}}}},
            provider="anthropic", model="claude-sonnet",
        )

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="priced-step", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-priced")
    assert result.status == "completed"

    row = await _get_step("priced-step")
    assert row.cost == pytest.approx(1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000)
    assert row.input_tokens == 1000
    assert row.output_tokens == 500


async def test_openclaw_style_step_with_no_usage_writes_null_cost(db):
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })

    async def fake_execute(step, ctx):
        # No meta.agentMeta.usage at all — mirrors the openclaw executor's contract.
        return LLMOutput(confidence=0.9, summary="ok", next_step_context="", raw_response={})

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["openclaw"] = executor_instance

    step = StepConfig(name="openclaw-step", executor="openclaw", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-openclaw")
    assert result.status == "completed"

    row = await _get_step("openclaw-step")
    assert row.cost is None
    assert row.input_tokens is None
    assert row.output_tokens is None


async def test_unpriced_model_writes_null_cost_even_with_token_data(db):
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })

    async def fake_execute(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="ok", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 100, "output_tokens": 50}}}},
            provider="mystery-provider", model="unknown-model",
        )

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="unpriced-step", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-unpriced")
    assert result.status == "completed"

    row = await _get_step("unpriced-step")
    assert row.cost is None
    assert row.input_tokens == 100  # token data itself is still captured — only cost is unknown


async def test_verifier_tokens_captured_and_included_in_cost(db):
    pricing.configure({
        "currency": "USD",
        "models": [
            {"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0},
            {"match": {"provider": "anthropic", "model": "claude-haiku"}, "input_per_mtok": 1.0, "output_per_mtok": 5.0},
        ],
    })

    async def fake_primary(step, ctx):
        return LLMOutput(
            confidence=0.5, summary="ok", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 1000, "output_tokens": 500}}}},
            provider="anthropic", model="claude-sonnet",
        )

    async def fake_verifier(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="verified", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 200, "output_tokens": 100}}}},
            provider="anthropic", model="claude-haiku",
        )

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    primary_executor = MagicMock()
    primary_executor.execute = fake_primary
    verifier_executor = MagicMock()
    verifier_executor.execute = fake_verifier
    runner._executor_instances["gateway"] = primary_executor
    runner._executor_instances["verifier-exec"] = verifier_executor

    step = StepConfig(
        name="verified-step", executor="gateway", prompt_template="",
        verifier=VerifierConfig(executor="verifier-exec", trigger=VerifierTriggerConfig(always=True)),
        confidence_threshold=0.0,
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-verified")
    assert result.status == "completed"

    row = await _get_step("verified-step")
    assert row.verifier_input_tokens == 200
    assert row.verifier_output_tokens == 100
    primary_cost = 1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000
    verifier_cost = 200 * 1.0 / 1_000_000 + 100 * 5.0 / 1_000_000
    assert row.cost == pytest.approx(primary_cost + verifier_cost)


async def test_verifier_ran_but_unpriced_makes_whole_step_cost_null(db):
    # Primary is priced; verifier's model has no rate match. The combined cost must
    # be NULL (unknown), not a silently partial primary-only sum.
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })

    async def fake_primary(step, ctx):
        return LLMOutput(
            confidence=0.5, summary="ok", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 1000, "output_tokens": 500}}}},
            provider="anthropic", model="claude-sonnet",
        )

    async def fake_verifier(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="verified", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 200, "output_tokens": 100}}}},
            provider="mystery-provider", model="unknown-model",
        )

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    primary_executor = MagicMock()
    primary_executor.execute = fake_primary
    verifier_executor = MagicMock()
    verifier_executor.execute = fake_verifier
    runner._executor_instances["gateway"] = primary_executor
    runner._executor_instances["verifier-exec"] = verifier_executor

    step = StepConfig(
        name="partially-unpriced-step", executor="gateway", prompt_template="",
        verifier=VerifierConfig(executor="verifier-exec", trigger=VerifierTriggerConfig(always=True)),
        confidence_threshold=0.0,
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-partial")
    assert result.status == "completed"

    row = await _get_step("partially-unpriced-step")
    assert row.cost is None
    assert row.verifier_input_tokens == 200  # tokens are still captured even though unpriced


async def test_pricing_reload_changes_future_steps_only(db):
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })

    async def fake_execute(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="ok", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 1000, "output_tokens": 500}}}},
            provider="anthropic", model="claude-sonnet",
        )

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    executor_instance = MagicMock()
    executor_instance.execute = fake_execute
    runner._executor_instances["gateway"] = executor_instance

    step = StepConfig(name="reload-step", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-before-reload")
    cost_before = (await _get_step("reload-step")).cost

    # Reload with a much higher rate — must not retroactively change the row above.
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 300.0, "output_per_mtok": 1500.0}],
    })

    cost_after_reload_no_new_run = (await _get_step("reload-step")).cost
    assert cost_after_reload_no_new_run == cost_before

    # A step priced AFTER the reload uses the new rate.
    step2 = StepConfig(name="reload-step-2", executor="gateway", prompt_template="")
    pipeline2 = PipelineConfig(name="p2", trigger=TriggerConfig(), steps=[step2])
    await runner.run(pipeline=pipeline2, normalised=_make_normalised(pipeline="p2"), run_id="r-after-reload")
    cost_new = (await _get_step("reload-step-2")).cost

    assert cost_new == pytest.approx(1000 * 300.0 / 1_000_000 + 500 * 1500.0 / 1_000_000)
    assert cost_new > cost_before * 50  # sanity: unmistakably the new, much higher rate


# ---------------------------------------------------------------------------
# Grounding judge cost (SPEC-cost-accounting.md — grounding is priced too)
# ---------------------------------------------------------------------------

_TOOL_TRACE = [
    {"type": "tool_call", "name": "grafana_query", "input": {"q": "up"}},
    {"type": "tool_result", "name": "grafana_query", "content": "0.9%", "is_error": False},
]


async def test_grounding_tokens_captured_and_included_in_cost(db):
    pricing.configure({
        "currency": "USD",
        "models": [
            {"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0},
            {"match": {"provider": "azure", "model": "gpt-5"}, "input_per_mtok": 2.0, "output_per_mtok": 10.0},
        ],
    })

    async def fake_primary(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="ok", next_step_context="",
            raw_response={
                "meta": {"agentMeta": {"usage": {"input_tokens": 1000, "output_tokens": 500}}},
                "trace": _TOOL_TRACE,
            },
            provider="anthropic", model="claude-sonnet",
        )

    async def fake_grounding(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="grounded", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 300, "output_tokens": 120}}}},
            provider="azure", model="gpt-5",
        )

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    primary_executor = MagicMock()
    primary_executor.execute = fake_primary
    grounding_executor = MagicMock()
    grounding_executor.execute = fake_grounding
    runner._executor_instances["gateway"] = primary_executor
    runner._executor_instances["grounding-exec"] = grounding_executor

    step = StepConfig(
        name="grounded-step", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding-exec"),
        confidence_threshold=0.0,
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-grounded")
    assert result.status == "completed"

    row = await _get_step("grounded-step")
    assert row.grounding_model == "gpt-5"
    assert row.grounding_provider == "azure"
    assert row.grounding_input_tokens == 300
    assert row.grounding_output_tokens == 120
    primary_cost = 1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000
    grounding_cost = 300 * 2.0 / 1_000_000 + 120 * 10.0 / 1_000_000
    assert row.cost == pytest.approx(primary_cost + grounding_cost)


async def test_grounding_ran_but_unpriced_makes_whole_step_cost_null(db):
    # Primary is priced; the grounding judge's model has no rate match. The combined
    # cost must be NULL (unknown), not a silently partial primary-only sum — same
    # honest-NULL-propagation rule as verifier.
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })

    async def fake_primary(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="ok", next_step_context="",
            raw_response={
                "meta": {"agentMeta": {"usage": {"input_tokens": 1000, "output_tokens": 500}}},
                "trace": _TOOL_TRACE,
            },
            provider="anthropic", model="claude-sonnet",
        )

    async def fake_grounding(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="grounded", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 300, "output_tokens": 120}}}},
            provider="mystery-provider", model="unknown-model",
        )

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    primary_executor = MagicMock()
    primary_executor.execute = fake_primary
    grounding_executor = MagicMock()
    grounding_executor.execute = fake_grounding
    runner._executor_instances["gateway"] = primary_executor
    runner._executor_instances["grounding-exec"] = grounding_executor

    step = StepConfig(
        name="partially-unpriced-grounding-step", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding-exec"),
        confidence_threshold=0.0,
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-grounding-unpriced")
    assert result.status == "completed"

    row = await _get_step("partially-unpriced-grounding-step")
    assert row.cost is None
    assert row.grounding_input_tokens == 300  # tokens still captured even though unpriced


async def test_grounding_that_did_not_run_does_not_affect_cost(db):
    # No trace on the primary output -> grounding's "no_trace" early return -> it
    # never actually ran. Must NOT force the step's cost to NULL — that would be
    # exactly the wrong lesson from the honest-NULL-propagation rule.
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })

    async def fake_primary(step, ctx):
        return LLMOutput(
            confidence=0.9, summary="ok", next_step_context="",
            raw_response={"meta": {"agentMeta": {"usage": {"input_tokens": 1000, "output_tokens": 500}}}},
            provider="anthropic", model="claude-sonnet",
            # no "trace" key -> _run_grounding's transcript is empty -> computed=False
        )

    runner = PipelineRunner(executors={}, session_factory=get_session_factory())
    primary_executor = MagicMock()
    primary_executor.execute = fake_primary
    runner._executor_instances["gateway"] = primary_executor
    runner._executor_instances["grounding-exec"] = MagicMock()  # never called

    step = StepConfig(
        name="no-trace-grounding-step", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding-exec"),
        confidence_threshold=0.0,
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="r-no-trace")
    assert result.status == "completed"

    row = await _get_step("no-trace-grounding-step")
    assert row.grounding_model is None
    assert row.cost == pytest.approx(1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000)
