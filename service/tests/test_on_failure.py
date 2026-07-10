"""Tests for on_failure step policy and step-level failure webhook."""
import pytest
from datetime import datetime
from pydantic import ValidationError
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.pipeline import (
    StepConfig,
    StepFailureConfig,
    StepFailureWebhookConfig,
    PipelineConfig,
    TriggerConfig,
)
from src.models.llm import LLMOutput
from src.models.context import NormalisedContext


# ---------------------------------------------------------------------------
# StepFailureConfig model
# ---------------------------------------------------------------------------

def test_default_policy_is_abort():
    step = StepConfig(name="x", executor="gateway")
    assert step.on_failure.policy == "abort"
    assert step.on_failure.webhook is None


def test_string_abort_coerced():
    step = StepConfig(name="x", executor="gateway", on_failure="abort")
    assert step.on_failure.policy == "abort"


def test_string_continue_coerced():
    step = StepConfig(name="x", executor="gateway", on_failure="continue")
    assert step.on_failure.policy == "continue"


def test_invalid_policy_raises():
    with pytest.raises(ValidationError):
        StepConfig(name="x", executor="gateway", on_failure="skip")


def test_block_with_webhook():
    step = StepConfig(
        name="x",
        executor="gateway",
        on_failure={
            "policy": "continue",
            "webhook": {
                "url": "https://example.com/hook",
                "payload": {"text": "step {{step_failure.step}} failed"},
            },
        },
    )
    assert step.on_failure.policy == "continue"
    assert step.on_failure.webhook is not None
    assert step.on_failure.webhook.url == "https://example.com/hook"


def test_block_webhook_only_defaults_abort():
    step = StepConfig(
        name="x",
        executor="gateway",
        on_failure={"webhook": {"url": "https://example.com/hook"}},
    )
    assert step.on_failure.policy == "abort"
    assert step.on_failure.webhook is not None


# ---------------------------------------------------------------------------
# Helpers shared by runner behaviour tests
# ---------------------------------------------------------------------------

def _make_runner(executor_instance=None, executor_name="gateway"):
    from src.pipeline.runner import PipelineRunner
    runner = PipelineRunner(executors={}, session_factory=None)
    if executor_instance is not None:
        runner._executor_instances[executor_name] = executor_instance
    return runner


def _make_pipeline(step: StepConfig):
    return PipelineConfig(
        name="test-pipeline",
        trigger=TriggerConfig(),
        steps=[step],
    )


def _make_normalised():
    return NormalisedContext(
        source="test",
        pipeline="test-pipeline",
        summary="test alert",
        severity="warning",
        received_at=datetime(2026, 1, 1, 12, 0, 0),
        labels={},
        raw={},
    )


def _failed_executor():
    inst = MagicMock()
    inst.execute = AsyncMock(side_effect=RuntimeError("boom"))
    return inst


def _ok_executor(confidence=0.9):
    inst = MagicMock()
    inst.execute = AsyncMock(return_value=LLMOutput(
        confidence=confidence,
        summary="ok",
        next_step_context="",
        raw_response={},
    ))
    return inst


# ---------------------------------------------------------------------------
# on_failure: abort (default) — pipeline stops on step failure
# ---------------------------------------------------------------------------

async def test_failed_step_aborts_pipeline_by_default():
    step = StepConfig(name="risky", executor="gateway")
    runner = _make_runner(_failed_executor())
    result = await runner._run_pipeline_body(
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-1",
        run_log=[],
    )
    assert result.status == "failed"
    assert len(result.steps) == 1
    assert result.steps[0].status == "failed"


async def test_failed_step_abort_run_log_has_step_error():
    step = StepConfig(name="risky", executor="gateway")
    runner = _make_runner(_failed_executor())
    run_log: list = []
    await runner._run_pipeline_body(
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-1",
        run_log=run_log,
    )
    events = [e["event"] for e in run_log]
    assert "step_error" in events


# ---------------------------------------------------------------------------
# on_failure: continue — pipeline keeps running past a failed step
# ---------------------------------------------------------------------------

async def test_failed_step_continue_pipeline_proceeds():
    """With on_failure: continue, a failing step doesn't abort the pipeline."""
    fail_exec = _failed_executor()
    ok_exec = _ok_executor()

    step1 = StepConfig(name="risky", executor="risky_ex", on_failure="continue")
    step2 = StepConfig(name="safe", executor="safe_ex")

    pipeline = PipelineConfig(
        name="test-pipeline",
        trigger=TriggerConfig(),
        steps=[step1, step2],
    )
    runner = _make_runner()
    runner._executor_instances["risky_ex"] = fail_exec
    runner._executor_instances["safe_ex"] = ok_exec

    result = await runner._run_pipeline_body(
        pipeline=pipeline,
        normalised=_make_normalised(),
        run_id="run-2",
        run_log=[],
    )

    assert result.status == "completed"
    assert len(result.steps) == 2
    assert result.steps[0].status == "failed"
    assert result.steps[1].status == "completed"


async def test_failed_step_continue_run_log_records_continuing_event():
    step = StepConfig(name="risky", executor="gateway", on_failure="continue")
    runner = _make_runner(_failed_executor())
    run_log: list = []
    await runner._run_pipeline_body(
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-3",
        run_log=run_log,
    )
    events = [e["event"] for e in run_log]
    assert "step_failed_continuing" in events


async def test_continue_policy_only_applies_to_failed_not_aborted():
    """A step that aborts (low confidence) still stops the pipeline even with on_failure: continue."""
    output = LLMOutput(confidence=0.1, summary="low conf", next_step_context="", raw_response={})
    inst = MagicMock()
    inst.execute = AsyncMock(return_value=output)

    step = StepConfig(
        name="x",
        executor="gateway",
        confidence_threshold=0.8,
        on_low_confidence="abort",
        on_failure="continue",  # only affects executor errors, not LLM confidence
    )
    runner = _make_runner(inst)
    result = await runner._run_pipeline_body(
        pipeline=_make_pipeline(step),
        normalised=_make_normalised(),
        run_id="run-4",
        run_log=[],
    )
    assert result.status == "aborted"


# ---------------------------------------------------------------------------
# on_failure webhook callback
# ---------------------------------------------------------------------------

async def test_step_failure_webhook_fires_on_failure():
    """When a step fails, the on_failure.webhook is called."""
    step = StepConfig(
        name="flaky",
        executor="gateway",
        on_failure={
            "policy": "abort",
            "webhook": {
                "url": "https://hooks.example.com/notify",
                "payload": {"text": "step failed"},
            },
        },
    )
    runner = _make_runner(_failed_executor())

    with patch.object(runner, "_fire_step_failure_webhook", new=AsyncMock()) as mock_fire:
        await runner._run_pipeline_body(
            pipeline=_make_pipeline(step),
            normalised=_make_normalised(),
            run_id="run-5",
            run_log=[],
        )
        mock_fire.assert_awaited_once()


async def test_step_failure_webhook_fires_even_when_continuing():
    """Webhook fires regardless of on_failure.policy."""
    step = StepConfig(
        name="flaky",
        executor="gateway",
        on_failure={
            "policy": "continue",
            "webhook": {
                "url": "https://hooks.example.com/notify",
                "payload": {"text": "step failed"},
            },
        },
    )
    runner = _make_runner(_failed_executor())

    with patch.object(runner, "_fire_step_failure_webhook", new=AsyncMock()) as mock_fire:
        result = await runner._run_pipeline_body(
            pipeline=_make_pipeline(step),
            normalised=_make_normalised(),
            run_id="run-6",
            run_log=[],
        )
        mock_fire.assert_awaited_once()
        assert result.status == "completed"


async def test_no_webhook_no_fire_when_step_succeeds():
    """Webhook callback does NOT fire when the step succeeds."""
    step = StepConfig(
        name="healthy",
        executor="gateway",
        on_failure={
            "webhook": {
                "url": "https://hooks.example.com/notify",
                "payload": {"text": "step failed"},
            },
        },
    )
    runner = _make_runner(_ok_executor())

    with patch.object(runner, "_fire_step_failure_webhook", new=AsyncMock()) as mock_fire:
        result = await runner._run_pipeline_body(
            pipeline=_make_pipeline(step),
            normalised=_make_normalised(),
            run_id="run-7",
            run_log=[],
        )
        mock_fire.assert_not_awaited()
        assert result.status == "completed"


async def test_fire_step_failure_webhook_posts_rendered_payload():
    """_fire_step_failure_webhook renders Jinja2 values and POSTs to the URL."""
    step = StepConfig(
        name="flaky",
        executor="gateway",
        on_failure={
            "webhook": {
                "url": "https://hooks.example.com/notify",
                "payload": {
                    "text": "Step {{step_failure.step}} failed: {{step_failure.summary}}",
                    "severity": "critical",
                },
            },
        },
    )

    runner = _make_runner()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = ""

    sent_bodies = []

    async def fake_request(self, method, url, content=None, headers=None):
        sent_bodies.append(content)
        return mock_response

    import httpx
    with patch.object(httpx.AsyncClient, "request", new=fake_request):  # type: ignore[arg-type]
        run_log: list = []
        step_result = MagicMock()
        step_result.output = LLMOutput(
            confidence=0.0,
            summary="executor exploded",
            next_step_context="",
            raw_response={},
        )
        step_result.status = "failed"

        await runner._fire_step_failure_webhook(
            step=step,
            result=step_result,
            pipeline=PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step], stage="production"),
            normalised=_make_normalised(),
            run_id="run-8",
            step_outputs={},
            run_log=run_log,
        )

    import json
    assert len(sent_bodies) == 1
    payload = json.loads(sent_bodies[0])
    assert "flaky" in payload["text"]
    assert "executor exploded" in payload["text"]
    assert payload["severity"] == "critical"

    events = [e["event"] for e in run_log]
    assert "step_failure_webhook_sent" in events


async def test_fire_step_failure_webhook_logs_failure_and_never_raises():
    """Webhook delivery failure is logged as a warn event and never aborts the pipeline."""
    step = StepConfig(
        name="flaky",
        executor="gateway",
        on_failure={"webhook": {"url": "https://hooks.example.com/notify"}},
    )
    runner = _make_runner()

    import httpx
    with patch.object(httpx.AsyncClient, "request", side_effect=httpx.ConnectError("unreachable")):
        run_log: list = []
        step_result = MagicMock()
        step_result.output = None
        step_result.status = "failed"

        # Must not raise
        await runner._fire_step_failure_webhook(
            step=step,
            result=step_result,
            pipeline=PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step], stage="production"),
            normalised=_make_normalised(),
            run_id="run-9",
            step_outputs={},
            run_log=run_log,
        )

    events = [e["event"] for e in run_log]
    assert "step_failure_webhook_failed" in events


async def test_fire_step_failure_webhook_suppressed_in_testing():
    """stage=testing (the default) never sends the webhook, and logs the suppression instead."""
    step = StepConfig(
        name="flaky",
        executor="gateway",
        on_failure={"webhook": {"url": "https://hooks.example.com/notify"}},
    )
    runner = _make_runner()

    import httpx
    with patch.object(httpx.AsyncClient, "request", new=AsyncMock()) as mock_request:
        run_log: list = []
        step_result = MagicMock()
        step_result.output = None
        step_result.status = "failed"

        await runner._fire_step_failure_webhook(
            step=step,
            result=step_result,
            pipeline=PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step]),  # default stage=testing
            normalised=_make_normalised(),
            run_id="run-10",
            step_outputs={},
            run_log=run_log,
        )

    mock_request.assert_not_awaited()
    events = [e["event"] for e in run_log]
    assert "step_failure_webhook_suppressed_testing" in events
