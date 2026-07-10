import json
import logging
import os

import httpx
from jinja2 import Environment, Undefined

from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig
from .base import BaseExecutor

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)


def _render_values(obj: object, context: dict) -> object:
    """Recursively render all string values in a structure as Jinja2 templates."""
    if isinstance(obj, str):
        return _jinja_env.from_string(obj).render(**context)
    if isinstance(obj, dict):
        return {k: _render_values(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_render_values(item, context) for item in obj]
    return obj


class NotifyExecutor(BaseExecutor):
    """Sends an outbound webhook notification with a structured payload.

    Unlike the generic WebhookExecutor (which uses prompt_template as the raw body),
    NotifyExecutor takes a payload dict in executor_config and renders each string value
    as a Jinja2 template before JSON-encoding the whole structure. This makes it
    ergonomic for Slack blocks, PagerDuty events, Teams cards, etc.

    executor_config keys:
        url             — required; target URL; supports ${ENV_VAR}
        method          — HTTP method (default: POST)
        payload         — dict whose string values are Jinja2 templates
        headers         — dict of request headers; values support ${ENV_VAR}
        content_type    — Content-Type shorthand (default: application/json)
        timeout_seconds — request timeout in seconds (default: 30)
    """

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        cfg = step.executor_config
        url = self._resolve_env(cfg.get("url", ""))
        if not url:
            raise ValueError(
                f"Step '{step.name}': executor_config.url is required for NotifyExecutor"
            )

        method = cfg.get("method", "POST").upper()
        timeout = float(cfg.get("timeout_seconds", 30))
        content_type = cfg.get("content_type", "application/json")

        raw_headers: dict = cfg.get("headers", {})
        headers = {k: self._resolve_env(str(v)) for k, v in raw_headers.items()}
        headers.setdefault("Content-Type", content_type)

        raw_payload = cfg.get("payload", {})
        rendered_payload = _render_values(raw_payload, context)
        body = json.dumps(rendered_payload).encode()

        if context.get("_testing"):
            logger.info(
                "[testing] NotifyExecutor suppressed: step=%s url=%s body=%s",
                step.name, url, body,
            )
            return LLMOutput(
                confidence=1.0,
                summary=f"[testing] notify suppressed (would POST to {url})",
                next_step_context="Notification suppressed in testing stage.",
                raw_response={"suppressed_testing": True, "url": url},
            )

        logger.info("NotifyExecutor: step=%s method=%s url=%s", step.name, method, url)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, content=body, headers=headers)
            response.raise_for_status()

        logger.info("NotifyExecutor: step=%s status=%d", step.name, response.status_code)

        return LLMOutput(
            confidence=1.0,
            summary=f"Notification sent: HTTP {response.status_code}",
            next_step_context=response.text[:200] if response.text else "",
            raw_response={"status_code": response.status_code, "body": response.text[:500]},
        )

    @staticmethod
    def _resolve_env(value: str) -> str:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value
