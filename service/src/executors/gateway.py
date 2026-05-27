import json
import logging
import uuid

import websockets
from jinja2 import Environment, Undefined

from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig
from .base import BaseExecutor

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from text that may contain surrounding prose.

    Tries the full text first (compliant models, zero overhead). If that
    fails, scans for the outermost { ... } span and parses that — handles
    models that add a sentence or two before or after the JSON payload.
    Markdown code fences are stripped before either attempt.
    """
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class GatewayExecutor(BaseExecutor):
    """Executor that invokes an agent via the P-Ork Gateway WebSocket API.

    executor_config keys (all optional except agent):
        agent          — gateway agent name (required)
        model         — model override string (e.g. "ollama/qwen3.5:4b")
        thinking_level — thinking level override (e.g. "low", "medium", "high")
        session_key   — Jinja2 template for session key; defaults to
                        "pipeline:{{pipeline_run_id}}:{{step_name}}"
        timeout_seconds — per-request timeout (default: 1200)
    """

    def __init__(self, url: str, token: str, timeout_seconds: int = 1200):
        self._url = url
        self._token = token
        self._timeout = timeout_seconds

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        agent = step.executor_config.get("agent")
        if not agent:
            raise ValueError(f"Step '{step.name}': executor_config.agent is required for GatewayExecutor")

        session_key = self._render(
            step.executor_config.get(
                "session_key",
                "pipeline:{{pipeline_run_id}}:{{step_name}}",
            ),
            context,
        )
        timeout = int(step.executor_config.get("timeout_seconds", self._timeout))
        prompt = self._render(step.prompt_template, context)

        model_override = step.executor_config.get("model")
        thinking_level = step.executor_config.get("thinking_level")

        logger.info(
            "Gateway execute: step=%s agent=%s session=%s",
            step.name, agent, session_key,
        )
        logger.debug("Prompt >>>\n%s", prompt)

        result = await self._call_agent(
            agent=agent,
            session_key=session_key,
            message=prompt,
            timeout=timeout,
            model_override=model_override,
            thinking_level=thinking_level,
        )

        response_text = (result.get("payloads") or [{}])[0].get("text", "")
        logger.debug("Response <<<\n%s", response_text)

        duration = result.get("meta", {}).get("durationMs")
        model = result.get("meta", {}).get("agentMeta", {}).get("model")
        logger.info("Gateway response: step=%s model=%s duration=%sms", step.name, model, duration)

        return self._parse_output(result, step.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render(self, template: str, context: dict) -> str:
        return _jinja_env.from_string(template).render(**context)

    async def _call_agent(
        self,
        agent: str,
        session_key: str,
        message: str,
        timeout: int,
        model_override: str | None,
        thinking_level: str | None,
    ) -> dict:
        import asyncio

        async def _do_connect():
            async with websockets.connect(self._url) as ws:
                # Step 1: receive challenge
                raw = await ws.recv()
                challenge = json.loads(raw)
                if challenge.get("event") != "challenge":
                    raise RuntimeError(f"Expected 'challenge' event, got: {raw[:200]}")

                # Step 2: send connect
                await ws.send(json.dumps({
                    "type": "req",
                    "id": str(uuid.uuid4()),
                    "method": "connect",
                    "params": {"auth": {"token": self._token}},
                }))

                # Step 3: receive connect response
                raw = await ws.recv()
                connect_resp = json.loads(raw)
                if not connect_resp.get("ok"):
                    err = connect_resp.get("error", {}).get("message", "unknown error")
                    raise RuntimeError(f"Gateway auth failed: {err}")

                # Step 4: send agent request
                req_id = str(uuid.uuid4())
                params = {
                    "agentId": agent,
                    "sessionKey": session_key,
                    "message": message,
                }
                if model_override:
                    params["model"] = model_override
                if thinking_level:
                    params["thinkingLevel"] = thinking_level

                await ws.send(json.dumps({
                    "type": "req",
                    "id": req_id,
                    "method": "agent",
                    "params": params,
                }))

                # Step 5: receive two responses with the same id
                # Frame 1: accepted
                raw = await ws.recv()
                frame1 = json.loads(raw)
                if frame1.get("id") != req_id:
                    raise RuntimeError(f"Unexpected response id in frame 1: {raw[:200]}")
                if not frame1.get("ok"):
                    err = frame1.get("error", {}).get("message", "unknown error")
                    raise RuntimeError(f"Gateway agent request rejected: {err}")

                # Frame 2: final result
                raw = await ws.recv()
                frame2 = json.loads(raw)
                if frame2.get("id") != req_id:
                    raise RuntimeError(f"Unexpected response id in frame 2: {raw[:200]}")
                if not frame2.get("ok"):
                    err = frame2.get("error", {}).get("message", "unknown error")
                    raise RuntimeError(f"Gateway agent run failed: {err}")

                return frame2.get("payload", {}).get("result", {})

        try:
            return await asyncio.wait_for(_do_connect(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Gateway agent call timed out after {timeout}s "
                f"(session={session_key})"
            )

    def _parse_output(self, result: dict, step_name: str) -> LLMOutput:
        payloads = result.get("payloads", [])
        if not payloads:
            raise RuntimeError(f"Gateway returned no payloads for step '{step_name}'")

        # Scan payloads in reverse — models sometimes narrate before returning JSON.
        # Take the last payload that yields a parseable JSON object.
        parsed = None
        for payload in reversed(payloads):
            text = payload.get("text", "").strip()
            parsed = _extract_json_object(text)
            if parsed is not None:
                break

        if parsed is None:
            last_text = payloads[-1].get("text", "")
            raise RuntimeError(
                f"Step '{step_name}': no payload contained valid JSON. "
                f"Last payload: {last_text[:500]}"
            )

        model = result.get("meta", {}).get("agentMeta", {}).get("model")

        try:
            return LLMOutput(
                **parsed,
                model=model,
                raw_response=result,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Step '{step_name}': agent JSON does not match LLMOutput schema: {exc}\n"
                f"Received: {parsed}"
            ) from exc
