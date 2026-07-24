import asyncio
import json
import logging
import shutil
from pathlib import Path
from jinja2 import Environment, Undefined

from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig
from ..tracing import gen_ai_attrs_from_meta, tracer
from .base import BaseExecutor, LLMParseError

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)

# TODO: Replace session clearing with --new-session flag once the OpenClaw
# feature request is implemented. Tracked at:
# https://github.com/openclaw/openclaw/issues (--new-session CLI option)
#
# Background: openclaw agent --session-id <id> does not create isolated
# sessions per invocation — all calls route to agent:<name>:main regardless
# of the ID passed. This means accumulated session history causes models to
# replay previous answers rather than execute fresh tool calls. Until
# --new-session exists, we clear the agent's session files before each call.
# Limitation: concurrent pipeline runs targeting the same agent will clobber
# each other's sessions. Acceptable for single-machine dev; revisit before
# multi-worker deployment.
_OPENCLAW_HOME = Path.home() / ".openclaw"


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


_VALID_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh"}


class OpenClawExecutor(BaseExecutor):
    """Executor that invokes an OpenClaw agent via the openclaw CLI.

    executor_config keys (all optional except agent):
        agent          — OpenClaw agent name (required)
        session_key    — Jinja2 template for the session key; defaults to
                         "pipeline:{{pipeline_run_id}}:{{step_name}}"
        thinking_level — Thinking budget: off|minimal|low|medium|high|xhigh
    """

    def __init__(self, timeout_seconds: int = 1200):
        self._timeout = timeout_seconds
        self._openclaw_bin = shutil.which("openclaw")
        if not self._openclaw_bin:
            raise RuntimeError("openclaw binary not found on PATH")

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        agent = step.executor_config.get("agent")
        if not agent:
            raise ValueError(f"Step '{step.name}': executor_config.agent is required for OpenClawExecutor")

        session_key = self._render(
            step.executor_config.get(
                "session_key",
                "pipeline:{{pipeline_run_id}}:{{step_name}}",
            ),
            context,
        )
        timeout = int(step.executor_config.get("timeout_seconds", self._timeout))
        prompt = self._render(step.prompt_template, context)

        thinking_level = step.executor_config.get("thinking_level")
        if thinking_level is not None and thinking_level not in _VALID_THINKING_LEVELS:
            raise ValueError(
                f"Step '{step.name}': invalid thinking_level '{thinking_level}'. "
                f"Must be one of: {', '.join(sorted(_VALID_THINKING_LEVELS))}"
            )

        logger.info(
            "OpenClaw execute: step=%s agent=%s session=%s thinking=%s",
            step.name, agent, session_key, thinking_level or "default",
        )
        logger.debug("Prompt >>>\n%s", prompt)

        self._clear_agent_sessions(agent)

        with tracer.start_as_current_span(
            "gen_ai.openclaw",
            attributes={"gen_ai.system": "openclaw", "pork.agent": agent},
        ) as span:
            raw = await self._call_agent(agent, session_key, prompt, timeout, thinking_level)
            span.set_attributes(gen_ai_attrs_from_meta(raw.get("result", {}).get("meta", {})))

        response_text = (raw.get("result", {}).get("payloads") or [{}])[0].get("text", "")
        logger.debug("Response <<<\n%s", response_text)

        duration = raw.get("result", {}).get("meta", {}).get("durationMs")
        model = raw.get("result", {}).get("meta", {}).get("agentMeta", {}).get("model")
        logger.info("OpenClaw response: step=%s model=%s duration=%sms", step.name, model, duration)

        return self._parse_output(raw, step.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_agent_sessions(self, agent: str) -> None:
        """Delete all session files for the given agent.

        TODO: Remove once openclaw agent --new-session is implemented.
              See module-level comment for full context.
        """
        sessions_dir = _OPENCLAW_HOME / "agents" / agent / "sessions"
        if not sessions_dir.exists():
            return

        cleared = 0
        for path in sessions_dir.iterdir():
            name = path.name
            if (
                name == "sessions.json"
                or path.suffix == ".jsonl"
                or ".jsonl.reset." in name
                or ".jsonl.deleted." in name
            ):
                try:
                    path.unlink()
                    cleared += 1
                except OSError as exc:
                    logger.warning("Could not remove session file %s: %s", path, exc)

        if cleared:
            logger.info("Cleared %d session file(s) for agent '%s'", cleared, agent)

    def _render(self, template: str, context: dict) -> str:
        return _jinja_env.from_string(template).render(**context)

    async def _call_agent(
        self, agent: str, session_key: str, message: str, timeout: int,
        thinking_level: str | None = None,
    ) -> dict:
        cmd = [
            self._openclaw_bin,
            "agent",
            "--agent", agent,
            "--session-id", session_key,
            "--message", message,
            "--json",
        ]
        if thinking_level:
            cmd += ["--thinking", thinking_level]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(
                f"OpenClaw agent call timed out after {timeout}s "
                f"(session={session_key})"
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenClaw agent exited with code {proc.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )

        try:
            return json.loads(stdout.decode())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenClaw returned non-JSON output: {stdout.decode(errors='replace')[:500]}"
            ) from exc

    def _parse_output(self, raw: dict, step_name: str) -> LLMOutput:
        if raw.get("status") != "ok":
            raise RuntimeError(
                f"OpenClaw run failed for step '{step_name}': status={raw.get('status')}"
            )

        payloads = raw.get("result", {}).get("payloads", [])
        if not payloads:
            raise RuntimeError(f"OpenClaw returned no payloads for step '{step_name}'")

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

        # Extract model from response metadata — more reliable than agent self-report
        model = (
            raw.get("result", {})
            .get("meta", {})
            .get("agentMeta", {})
            .get("model")
        )

        try:
            return LLMOutput(
                **parsed,
                model=model,
                raw_response=raw,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Step '{step_name}': agent JSON does not match LLMOutput schema: {exc}\n"
                f"Received: {parsed}"
            ) from exc
