"""Tests for GatewayExecutor.execute — specifically that the rendered prompt text is
stashed onto raw_response so it survives into PipelineStep.prompt (see runner.py's
_db_save_step), rather than being lost after the WS call returns."""
from unittest.mock import AsyncMock, patch

from src.executors.gateway import GatewayExecutor
from src.models.pipeline import StepConfig


def _ws_result(text='{"confidence": 0.9, "summary": "ok", "next_step_context": ""}'):
    return {
        "payloads": [{"text": text}],
        "meta": {"durationMs": 100, "agentMeta": {"model": "claude-sonnet-5", "provider": "anthropic"}},
    }


async def test_execute_stashes_rendered_prompt_on_raw_response():
    executor = GatewayExecutor(url="ws://test", token="tok")
    step = StepConfig(
        name="investigate", executor="gateway",
        executor_config={"agent": "sre-investigation"},
        prompt_template="Alert severity: {{severity}}. Investigate.",
    )

    with patch.object(executor, "_call_agent", new=AsyncMock(return_value=_ws_result())):
        output = await executor.execute(step, {"severity": "critical"})

    assert output.raw_response["prompt"] == "Alert severity: critical. Investigate."


async def test_execute_stashed_prompt_reflects_actual_rendered_context_not_template():
    """The stashed value must be the RENDERED text (with real run data substituted in),
    not the raw {{ }} template — otherwise it's no more useful than the config."""
    executor = GatewayExecutor(url="ws://test", token="tok")
    step = StepConfig(
        name="investigate", executor="gateway",
        executor_config={"agent": "sre-investigation"},
        prompt_template="Service: {{labels.service}}. Do not open a ticket for this alert.",
    )

    with patch.object(executor, "_call_agent", new=AsyncMock(return_value=_ws_result())):
        output = await executor.execute(step, {"labels": {"service": "payment-api"}})

    assert "payment-api" in output.raw_response["prompt"]
    assert "Do not open a ticket" in output.raw_response["prompt"]
    assert "{{" not in output.raw_response["prompt"]


async def test_execute_stashes_response_text_on_raw_response():
    """The exact payload text that parsed successfully must survive onto
    raw_response too — grounding's report reads this back as 'what the judge
    replied' (see runner.py's _run_grounding), same round-trip as 'prompt' above."""
    executor = GatewayExecutor(url="ws://test", token="tok")
    step = StepConfig(
        name="investigate:grounding", executor="gateway",
        executor_config={"agent": "grounding-judge"},
        prompt_template="Judge this.",
    )
    text = '{"confidence": 0.75, "summary": "3 of 4 claims supported", "next_step_context": ""}'

    with patch.object(executor, "_call_agent", new=AsyncMock(return_value=_ws_result(text=text))):
        output = await executor.execute(step, {})

    assert output.raw_response["response_text"] == text
