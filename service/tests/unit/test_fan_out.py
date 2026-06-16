"""Tests for fan-out (map over runtime list) config and runner behaviour."""
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock

from src.models.pipeline import (
    FanOutConfig,
    FanOutGroupConfig,
    PipelineConfig,
    StepConfig,
    TriggerConfig,
)
from src.models.llm import LLMOutput


# ---------------------------------------------------------------------------
# FanOutConfig model tests
# ---------------------------------------------------------------------------

def test_fan_out_config_requires_name():
    with pytest.raises(ValidationError):
        FanOutConfig(executor="gateway", over="{{ items }}")


def test_fan_out_config_requires_executor():
    with pytest.raises(ValidationError):
        FanOutConfig(name="fo", over="{{ items }}")


def test_fan_out_config_requires_over():
    with pytest.raises(ValidationError):
        FanOutConfig(name="fo", executor="gateway")


def test_fan_out_config_defaults():
    cfg = FanOutConfig(name="fo", executor="gateway", over="{{ items }}")
    assert cfg.as_var == "item"
    assert cfg.max_items == 20
    assert cfg.on_empty == "complete"
    assert cfg.join == "all_must_pass"
    assert cfg.confidence_threshold == 0.75
    assert cfg.on_low_confidence == "escalate"
    assert cfg.verifier is None
    assert cfg.when is None


def test_fan_out_config_as_alias_via_model_validate():
    cfg = FanOutConfig.model_validate(
        {"name": "fo", "executor": "gateway", "over": "{{ items }}", "as": "svc"}
    )
    assert cfg.as_var == "svc"


def test_fan_out_config_as_var_via_python():
    cfg = FanOutConfig(name="fo", executor="gateway", over="{{ items }}", as_var="service")
    assert cfg.as_var == "service"


def test_fan_out_group_config_wraps_fan_out():
    grp = FanOutGroupConfig(
        fan_out=FanOutConfig(name="fo", executor="gateway", over="{{ items }}")
    )
    assert grp.fan_out.name == "fo"


def test_pipeline_config_accepts_fan_out_step():
    pipeline = PipelineConfig(
        name="test",
        trigger=TriggerConfig(),
        steps=[
            StepConfig(name="list-services", executor="gateway"),
            FanOutGroupConfig(
                fan_out=FanOutConfig(
                    name="triage-services",
                    executor="gateway",
                    over="{{ steps.list_services.items }}",
                )
            ),
        ],
    )
    assert len(pipeline.steps) == 2
    assert isinstance(pipeline.steps[1], FanOutGroupConfig)


# ---------------------------------------------------------------------------
# Runner helpers (copied from test_loop.py — not yet in a shared fixture file)
# ---------------------------------------------------------------------------

def _make_runner():
    from src.pipeline.runner import PipelineRunner
    return PipelineRunner(executors={}, session_factory=None)


def _make_output(confidence: float, **extra) -> LLMOutput:
    return LLMOutput(
        confidence=confidence,
        summary="test",
        next_step_context="",
        raw_response={},
        **extra,
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


def _make_pipeline(steps=None):
    return PipelineConfig(
        name="test-pipeline",
        trigger=TriggerConfig(),
        steps=steps or [],
    )


def _make_fan_out(**kwargs) -> FanOutConfig:
    defaults = dict(
        name="triage",
        executor="gateway",
        over="['api', 'worker', 'db']",
        as_var="svc",
        confidence_threshold=0.0,
        on_low_confidence="proceed",
    )
    defaults.update(kwargs)
    return FanOutConfig(**defaults)


# ---------------------------------------------------------------------------
# Runner tests (mocked executor, no DB)
# ---------------------------------------------------------------------------

async def test_fan_out_branches_execute_for_each_item():
    received_items = []

    async def fake_execute(step, ctx):
        received_items.append(ctx.get("svc"))
        return _make_output(0.9)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert result.status == "completed"
    assert sorted(received_items) == ["api", "db", "worker"]
    assert len(result.branch_outputs) == 3


async def test_fan_out_branch_context_vars():
    received_ctxs = []

    async def fake_execute(step, ctx):
        received_ctxs.append(dict(ctx))
        return _make_output(0.9)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="['api', 'worker']"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert len(received_ctxs) == 2
    # Both contexts carry fan_out_total=2
    for ctx in received_ctxs:
        assert ctx["fan_out_total"] == 2
    # fan_out_index is 0 and 1 across the two branches (order may vary due to gather)
    indices = {ctx["fan_out_index"] for ctx in received_ctxs}
    assert indices == {0, 1}
    # as_var "svc" is injected
    items = {ctx["svc"] for ctx in received_ctxs}
    assert items == {"api", "worker"}


async def test_fan_out_branch_outputs_keyed_by_name_slash_index():
    async def fake_execute(step, ctx):
        return _make_output(0.9)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(name="triage-services", over="['a', 'b']"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert "triage-services/0" in result.branch_outputs
    assert "triage-services/1" in result.branch_outputs


async def test_fan_out_as_alias_works_end_to_end():
    received_ctxs = []

    async def fake_execute(step, ctx):
        received_ctxs.append(dict(ctx))
        return _make_output(0.9)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    fan_out = FanOutConfig.model_validate(
        {
            "name": "fo",
            "executor": "gateway",
            "over": "['x']",
            "as": "service",
            "confidence_threshold": 0.0,
            "on_low_confidence": "proceed",
        }
    )

    await runner._run_fan_out_impl(
        fan_out=fan_out,
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert len(received_ctxs) == 1
    assert "service" in received_ctxs[0]
    assert received_ctxs[0]["service"] == "x"


async def test_fan_out_max_items_cap():
    runner = _make_runner()

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="['a', 'b', 'c']", max_items=2),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert result.status == "failed"
    assert "max_items" in result.output.summary


async def test_fan_out_empty_on_empty_complete():
    runner = _make_runner()

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="[]", on_empty="complete"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert result.status == "completed"
    assert result.effective_confidence == 1.0


async def test_fan_out_empty_on_empty_abort():
    runner = _make_runner()

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="[]", on_empty="abort"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert result.status == "failed"


async def test_fan_out_empty_on_empty_skip():
    runner = _make_runner()

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="[]", on_empty="skip"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert result.status == "completed"
    assert result.effective_confidence == 1.0
    run_log: list = []
    await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="[]", on_empty="skip"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=run_log,
    )
    events = [e["event"] for e in run_log]
    assert "fan_out_skipped" in events


async def test_fan_out_bad_over_expression_fails_step():
    runner = _make_runner()

    # "not_a_list" is not defined in context — renders to "" — fails to parse
    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="{{ not_a_list }}"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    assert result.status == "failed"
    assert "failed to resolve" in result.output.summary


async def test_fan_out_over_resolves_via_step_outputs():
    received_items = []

    async def fake_execute(step, ctx):
        received_items.append(ctx.get("svc"))
        return _make_output(0.9)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    # Use "services" not "items" — "items" shadows the dict built-in method in Jinja2
    # attribute access (getattr wins over getitem for Python dict methods).
    prior_output = _make_output(0.9, services=["auth", "payments"])
    step_outputs = {"list-services": prior_output}

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="{{ steps.list_services.services }}"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs=step_outputs,
        run_log=[],
    )

    assert result.status == "completed"
    assert sorted(received_items) == ["auth", "payments"]


async def test_fan_out_all_must_pass_join():
    call_count = 0

    async def fake_execute(step, ctx):
        nonlocal call_count
        call_count += 1
        conf = 0.3 if call_count == 1 else 0.9
        return _make_output(conf)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    result = await runner._run_fan_out_impl(
        fan_out=_make_fan_out(
            over="['a', 'b']",
            join="all_must_pass",
            confidence_threshold=0.5,
            on_low_confidence="escalate",
        ),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=[],
    )

    # all_must_pass = min(0.3, 0.9) = 0.3, which is below threshold 0.5
    assert result.effective_confidence == 0.3
    assert result.status == "escalated"


async def test_fan_out_run_log_events():
    async def fake_execute(step, ctx):
        return _make_output(0.9)

    executor_instance = MagicMock()
    executor_instance.execute = fake_execute

    runner = _make_runner()
    runner._executor_instances["gateway"] = executor_instance

    run_log: list = []
    await runner._run_fan_out_impl(
        fan_out=_make_fan_out(over="['a', 'b']"),
        index=0,
        pipeline=_make_pipeline(),
        normalised=_make_normalised(),
        run_id="run-1",
        step_outputs={},
        run_log=run_log,
    )

    events = [e["event"] for e in run_log]
    assert "fan_out_started" in events
    assert "fan_out_branch_completed" in events
    assert "fan_out_completed" in events
