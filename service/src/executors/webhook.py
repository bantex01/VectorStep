import logging
import os

import httpx
from jinja2 import Environment, Undefined

from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig
from .base import BaseExecutor

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)


class WebhookExecutor(BaseExecutor):
    """POSTs the rendered prompt_template as the request body to a URL.

    executor_config keys:
        url             — required, target URL
        method          — HTTP method (default: POST)
        headers         — dict of request headers; values support ${ENV_VAR} substitution
        timeout_seconds — request timeout in seconds (default: 30)
        content_type    — Content-Type header shorthand (default: application/json)
    """

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        cfg = step.executor_config
        url = cfg.get("url")
        if not url:
            raise ValueError(
                f"Step '{step.name}': executor_config.url is required for WebhookExecutor"
            )

        method = cfg.get("method", "POST").upper()
        timeout = float(cfg.get("timeout_seconds", 30))
        content_type = cfg.get("content_type", "application/json")

        raw_headers: dict = cfg.get("headers", {})
        headers = {k: self._resolve_env(v) for k, v in raw_headers.items()}
        headers.setdefault("Content-Type", content_type)

        body = _jinja_env.from_string(step.prompt_template).render(**context)

        logger.info(
            "WebhookExecutor: step=%s method=%s url=%s",
            step.name, method, url,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, content=body.encode(), headers=headers)
            response.raise_for_status()

        preview = response.text[:200] if response.text else ""
        logger.info(
            "WebhookExecutor: step=%s status=%d",
            step.name, response.status_code,
        )

        return LLMOutput(
            confidence=1.0,
            summary=f"Webhook delivered: HTTP {response.status_code}",
            next_step_context=preview,
            raw_response={"status_code": response.status_code, "body": response.text[:500]},
        )

    @staticmethod
    def _resolve_env(value: str) -> str:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value
