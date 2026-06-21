import asyncio
import base64
import json
import logging
import time
import uuid
from pathlib import Path

import websockets
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)
from jinja2 import Environment, Undefined

from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig
from ..tracing import gen_ai_attrs_from_meta, tracer
from .base import BaseExecutor

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)

_VALID_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh"}

_DEFAULT_IDENTITY_DIR = Path.home() / ".openclaw" / "identity"
_GATEWAY_WS_URL = "ws://127.0.0.1:18789/rpc"

# Default session key — must start with "agent:{agent}:" for the gateway to accept it.
_DEFAULT_SESSION_KEY = "agent:{{agent}}:pipeline:{{pipeline_run_id}}:{{current_step}}"

_MISSING_IDENTITY_HINT = (
    "OpenClaw identity files are created automatically when OpenClaw is installed "
    "and run on this machine. If OpenClaw is running on a different machine, copy "
    "~/.openclaw/identity/ from that machine to this one (or to a custom path) and "
    "set executors.openclaw.identity_dir in config.yaml."
)


def _resolve_identity_dir(identity_dir: str) -> Path:
    return Path(identity_dir).expanduser() if identity_dir else _DEFAULT_IDENTITY_DIR


def validate_identity(identity_dir: str = "") -> None:
    """Check identity files exist and are readable. Call at startup for early failure."""
    id_dir = _resolve_identity_dir(identity_dir)
    for name in ("device.json", "device-auth.json"):
        path = id_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"OpenClaw identity file not found: {path}\n{_MISSING_IDENTITY_HINT}"
            )


def _load_device_identity(identity_dir: str) -> tuple[str, object, str]:
    """Return (device_id, private_key, pub_b64url)."""
    path = _resolve_identity_dir(identity_dir) / "device.json"
    data = json.loads(path.read_text())
    priv_key = load_pem_private_key(data["privateKeyPem"].encode(), password=None)
    pub_raw = priv_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    pub_b64url = base64.urlsafe_b64encode(pub_raw).rstrip(b"=").decode()
    return data["deviceId"], priv_key, pub_b64url


def _load_operator_token(identity_dir: str) -> tuple[str, list[str]]:
    """Return (token, scopes) from the cached device-auth store."""
    path = _resolve_identity_dir(identity_dir) / "device-auth.json"
    data = json.loads(path.read_text())
    entry = data["tokens"]["operator"]
    return entry["token"], entry["scopes"]


def _sign_connect_payload(
    device_id: str, priv_key, op_token: str, scopes: list[str], nonce: str
) -> tuple[str, int]:
    """Build v3 signed payload. Returns (signature_b64, signed_at_ms)."""
    signed_at = int(time.time() * 1000)
    payload_str = "|".join([
        "v3", device_id, "gateway-client", "backend", "operator",
        ",".join(scopes), str(signed_at), op_token, nonce,
        "darwin", "",
    ])
    sig = base64.b64encode(priv_key.sign(payload_str.encode())).decode()
    return sig, signed_at


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from text that may contain surrounding prose."""
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


class OpenClawWSExecutor(BaseExecutor):
    """Executor that invokes OpenClaw agents via the Gateway WebSocket API.

    Replaces the CLI subprocess executor. Supports model override and
    thinking level, and uses proper server-side session isolation so no
    session file deletion is needed.

    executor_config keys (all optional except agent):
        agent          — OpenClaw agent name (required)
        session_key    — Jinja2 template; defaults to
                         "agent:{{agent}}:pipeline:{{pipeline_run_id}}:{{current_step}}"
                         Must start with "agent:{agent-name}:" — the gateway
                         validates this.
        model          — Optional model override, e.g. "anthropic/claude-opus-4-7"
        thinking_level — Thinking budget: off|minimal|low|medium|high|xhigh
    """

    def __init__(self, gateway_url: str = _GATEWAY_WS_URL, identity_dir: str = ""):
        self._gateway_url = gateway_url
        self._device_id, self._priv_key, self._pub_b64url = _load_device_identity(identity_dir)
        self._op_token, self._scopes = _load_operator_token(identity_dir)

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        agent = step.executor_config.get("agent")
        if not agent:
            raise ValueError(
                f"Step '{step.name}': executor_config.agent is required"
            )

        session_key = _jinja_env.from_string(
            step.executor_config.get("session_key", _DEFAULT_SESSION_KEY)
        ).render(**context, agent=agent)

        model = step.executor_config.get("model")
        thinking_level = step.executor_config.get("thinking_level")
        if thinking_level is not None and thinking_level not in _VALID_THINKING_LEVELS:
            raise ValueError(
                f"Step '{step.name}': invalid thinking_level '{thinking_level}'. "
                f"Must be one of: {', '.join(sorted(_VALID_THINKING_LEVELS))}"
            )

        prompt = _jinja_env.from_string(step.prompt_template).render(**context)

        logger.info(
            "OpenClawWS execute: step=%s agent=%s session=%s model=%s thinking=%s",
            step.name, agent, session_key, model or "agent-default",
            thinking_level or "default",
        )
        logger.debug("Prompt >>>\n%s", prompt)

        with tracer.start_as_current_span(
            "gen_ai.openclaw_ws",
            attributes={
                "gen_ai.system": "openclaw_ws",
                "pork.agent": agent,
                "gen_ai.request.model": model or "",
            },
        ) as span:
            text, model_used, duration_ms = await self._run_agent(
                agent=agent,
                session_key=session_key,
                message=prompt,
                model=model,
                thinking_level=thinking_level,
            )
            span.set_attributes(gen_ai_attrs_from_meta(
                {"agentMeta": {"model": model_used}, "durationMs": duration_ms}
            ))

        logger.debug("Response <<<\n%s", text)
        logger.info(
            "OpenClawWS response: step=%s model=%s duration=%sms",
            step.name, model_used, duration_ms,
        )

        return self._parse_output(text, step.name, model_used, duration_ms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _connect(self, ws) -> None:
        """Complete the challenge-response handshake."""
        challenge = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        nonce = challenge["payload"]["nonce"]

        sig, signed_at = _sign_connect_payload(
            self._device_id, self._priv_key, self._op_token, self._scopes, nonce
        )

        resp = await self._request(ws, "connect", {
            "minProtocol": 3,
            "maxProtocol": 3,
            "client": {
                "id": "gateway-client",
                "version": "2026.4.11",
                "platform": "darwin",
                "mode": "backend",
            },
            "role": "operator",
            "scopes": self._scopes,
            "device": {
                "id": self._device_id,
                "publicKey": self._pub_b64url,
                "signature": sig,
                "signedAt": signed_at,
                "nonce": nonce,
            },
            "auth": {"token": self._op_token},
        })

        if not resp.get("ok"):
            raise RuntimeError(
                f"Gateway connect failed: {resp.get('error', {}).get('message', resp)}"
            )

    async def _request(self, ws, method: str, params: dict) -> dict:
        """Send a req frame and return the matching res frame (skipping events)."""
        req_id = str(uuid.uuid4())
        await ws.send(json.dumps({"type": "req", "id": req_id, "method": method, "params": params}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("type") == "res" and msg.get("id") == req_id:
                return msg

    async def _run_agent(
        self,
        agent: str,
        session_key: str,
        message: str,
        model: str | None,
        thinking_level: str | None,
    ) -> tuple[str, str | None, int | None]:
        """Open a gateway connection, run the agent.

        Returns (response_text, model_used, duration_ms).

        The gateway sends two res frames for the same agent request ID:
          1. status="accepted"  — run queued, contains runId
          2. status="ok"        — run complete, contains payloads + agentMeta
        We wait for the second frame; no agent.wait call needed.
        """
        async with websockets.connect(self._gateway_url) as ws:
            await self._connect(ws)

            agent_params: dict = {
                "idempotencyKey": str(uuid.uuid4()),
                "agentId": agent,
                "sessionKey": session_key,
                "message": message,
            }
            if model:
                agent_params["model"] = model
            if thinking_level:
                agent_params["thinking"] = thinking_level

            # NOTE: inject_traceparent() is intentionally NOT called here.
            # The OpenClaw gateway does not accept unknown params (like W3C
            # traceparent) and rejects the request with a validation error.
            # Trace context propagation is only used with the P-Ork Gateway
            # executor, which explicitly supports it.

            agent_req_id = str(uuid.uuid4())
            await ws.send(json.dumps({
                "type": "req",
                "id": agent_req_id,
                "method": "agent",
                "params": agent_params,
            }))

            # No message-count limit — the real safety nets are the per-recv
            # timeout=120 below and the step-level timeout_seconds in the pipeline
            # YAML (which cancels this entire coroutine via asyncio.wait_for in
            # the runner). Long investigation agents can emit hundreds of event
            # frames before the final result arrives.
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=120)
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"Timed out waiting for agent response (session={session_key})"
                    )

                msg = json.loads(raw)
                if msg.get("type") != "res" or msg.get("id") != agent_req_id:
                    continue  # skip event frames and unrelated responses

                if not msg.get("ok"):
                    err = msg.get("error", {})
                    raise RuntimeError(f"Agent call failed: {err.get('message', err)}")

                payload = msg.get("payload") or {}
                status = payload.get("status")

                if status == "accepted":
                    logger.debug("Agent run accepted: runId=%s", payload.get("runId"))
                    continue  # wait for the final result frame

                if status == "ok":
                    result = payload.get("result") or {}
                    meta = result.get("meta") or {}
                    agent_meta = meta.get("agentMeta") or {}

                    payloads = result.get("payloads") or []
                    response_text = self._extract_text(payloads, session_key)
                    model_used = agent_meta.get("model")
                    duration_ms = meta.get("durationMs")
                    return response_text, model_used, duration_ms

                raise RuntimeError(
                    f"Unexpected agent result status '{status}' (session={session_key})"
                )

    def _extract_text(self, payloads: list, session_key: str) -> str:
        """Scan payloads in reverse for the last one containing valid text."""
        for payload in reversed(payloads):
            text = (payload.get("text") or "").strip()
            if text:
                return text
        raise RuntimeError(
            f"Agent returned no text in payloads (session={session_key})"
        )

    def _parse_output(
        self, text: str, step_name: str, model_used: str | None, duration_ms: int | None
    ) -> LLMOutput:
        parsed = _extract_json_object(text)
        if parsed is None:
            raise RuntimeError(
                f"Step '{step_name}': agent response contained no valid JSON. "
                f"Response: {text[:500]}"
            )

        try:
            return LLMOutput(
                **parsed,
                model=model_used,
                raw_response={"text": text, "duration_ms": duration_ms},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Step '{step_name}': agent JSON does not match LLMOutput schema: {exc}\n"
                f"Received: {parsed}"
            ) from exc
