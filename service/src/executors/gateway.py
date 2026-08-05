import json
import logging
import uuid

import websockets
from jinja2 import Environment, Undefined

from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig
from .. import run_events
from ..tracing import gen_ai_attrs_from_meta, inject_traceparent, tracer
from ..utils import utc_now
from .base import BaseExecutor, LLMParseError

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
    """Executor that invokes an agent via the VectorStep Gateway WebSocket API.

    executor_config keys (all optional except agent):
        agent          — gateway agent name (required)
        model         — model override string (e.g. "ollama/qwen3.5:4b")
        thinking_level — thinking level override (e.g. "low", "medium", "high")
        session_key   — Jinja2 template for session key; defaults to
                        "pipeline:{{pipeline_run_id}}:{{step_name}}"
        timeout_seconds — per-request timeout (default: 1200)
        trace_max_chars — overrides the Gateway's limits.trace_tool_result_max_chars
                        (default 3000) for this step's tool_result trace events only.
                        Only affects what's recorded/streamed for observability — the
                        agent's own conversation always sees the full tool output
                        regardless of this setting. Raise it on steps whose tools
                        return long content (a full document read, a large query
                        result) if grounding or a human is drawing false conclusions
                        from a trace that got cut off before the real evidence.
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
                "agent:{{agent}}:pipeline:{{pipeline_run_id}}:{{step_name}}",
            ),
            {**context, "agent": agent},
        )
        timeout = int(step.executor_config.get("timeout_seconds", self._timeout))
        prompt = self._render(step.prompt_template, context)

        model_override = step.executor_config.get("model")
        thinking_level = step.executor_config.get("thinking_level")
        trace_max_chars = step.executor_config.get("trace_max_chars")

        logger.info(
            "Gateway execute: step=%s agent=%s session=%s",
            step.name, agent, session_key,
        )
        logger.debug("Prompt >>>\n%s", prompt)

        with tracer.start_as_current_span(
            "gen_ai.gateway",
            attributes={
                "gen_ai.system": "gateway",
                "vectorstep.agent": agent,
                "gen_ai.request.model": model_override or "",
            },
        ) as span:
            result = await self._call_agent(
                agent=agent,
                session_key=session_key,
                message=prompt,
                timeout=timeout,
                model_override=model_override,
                thinking_level=thinking_level,
                trace_max_chars=trace_max_chars,
                run_id=context.get("pipeline_run_id", ""),
                step_name=step.name,
            )
            span.set_attributes(gen_ai_attrs_from_meta(result.get("meta", {})))

        response_text = (result.get("payloads") or [{}])[0].get("text", "")
        logger.debug("Response <<<\n%s", response_text)

        duration = result.get("meta", {}).get("durationMs")
        model = result.get("meta", {}).get("agentMeta", {}).get("model")
        logger.info("Gateway response: step=%s model=%s duration=%sms", step.name, model, duration)

        output = self._parse_output(result, step.name)
        # Stashed here (not in a separate return value) so it survives the same
        # raw_response round-trip everything else does — _db_save_step reads it back
        # out for PipelineStep.prompt, which otherwise has nothing but this step's own
        # rendered instructions to reconstruct what the agent was actually asked.
        output.raw_response["prompt"] = prompt
        return output

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
        trace_max_chars: int | None = None,
        run_id: str = "",
        step_name: str = "",
    ) -> dict:
        import asyncio

        async def _do_connect():
            async with websockets.connect(self._url) as ws:
                # Step 1: receive challenge
                raw = await ws.recv()
                challenge = json.loads(raw)
                event_name = challenge.get("event", "")
                if event_name not in ("challenge", "connect.challenge"):
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
                if trace_max_chars:
                    params["traceToolResultMax"] = trace_max_chars

                params = inject_traceparent(params)

                await ws.send(json.dumps({
                    "type": "req",
                    "id": req_id,
                    "method": "agent",
                    "params": params,
                }))

                # Step 5: loop until the final 'ok' frame.
                # Between 'accepted' and 'ok' the gateway may send any number of
                # 'trace_event' frames — one per LLM output block or tool call/result.
                while True:
                    raw = await ws.recv()
                    frame = json.loads(raw)
                    if frame.get("id") != req_id:
                        continue
                    if not frame.get("ok"):
                        err = frame.get("error", {}).get("message", "unknown error")
                        raise RuntimeError(f"Gateway agent run failed: {err}")

                    payload = frame.get("payload", {})
                    status = payload.get("status")

                    if status == "accepted":
                        logger.debug("Agent run accepted: runId=%s", payload.get("runId"))
                        continue

                    if status == "trace_event":
                        event = payload.get("event") or {}
                        if run_id and event:
                            self._publish_trace_event(run_id, step_name, event)
                        continue

                    if status == "ok":
                        return payload.get("result", {})

                    raise RuntimeError(
                        f"Unexpected agent result status '{status}' (session={session_key})"
                    )

        try:
            return await asyncio.wait_for(_do_connect(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Gateway agent call timed out after {timeout}s "
                f"(session={session_key})"
            )

    def _publish_trace_event(self, run_id: str, step_name: str, event: dict) -> None:
        """Publish a single gateway trace event to the live-tail event bus.

        Content fields are truncated for the live tail — the full content arrives
        in the batch trace on the final 'ok' frame and is stored in agent_trace.
        """
        _LIVE_MAX = 200
        live: dict = {
            "ts": utc_now().isoformat(timespec="milliseconds") + "Z",
            "type": "agent_trace",
            "step": step_name,
            "trace_type": event.get("type"),
        }
        t = event.get("type")
        if t == "llm_call":
            live["iteration"] = event.get("iteration")
        elif t in ("thinking", "text"):
            c = event.get("content", "")
            live["content"] = c[:_LIVE_MAX] + "…" if len(c) > _LIVE_MAX else c
        elif t == "tool_call":
            live["name"] = event.get("name")
            live["input"] = event.get("input")
        elif t == "tool_result":
            live["name"] = event.get("name")
            live["is_error"] = event.get("is_error", False)
            c = event.get("content", "")
            live["content"] = c[:_LIVE_MAX] + "…" if len(c) > _LIVE_MAX else c
        run_events.publish(run_id, live)

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
            raise LLMParseError(
                f"Step '{step_name}': no payload contained valid JSON. "
                f"Last payload: {last_text[:500]}",
                raw_text=last_text,
            )

        agent_meta = result.get("meta", {}).get("agentMeta", {})
        model = agent_meta.get("model")
        provider = agent_meta.get("provider")
        agent_version = agent_meta.get("agentVersion")

        try:
            output = LLMOutput(
                **parsed,
                model=model,
                provider=provider,
                agent_version=agent_version,
                raw_response=result,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Step '{step_name}': agent JSON does not match LLMOutput schema: {exc}\n"
                f"Received: {parsed}"
            ) from exc

        # The exact payload text that parsed successfully — stashed the same way
        # `execute()` stashes `prompt`, so a caller (e.g. grounding's report) can show
        # "what the agent actually replied", not just the structured fields we kept.
        output.raw_response["response_text"] = text
        return output
