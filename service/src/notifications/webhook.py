import logging
import os

import httpx
from jinja2 import Environment, Undefined

from ..models.pipeline import NotificationConfig

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)


class WebhookNotifier:
    """Fire-and-forget HTTP notifier for pipeline state transitions.

    config keys (under notification.config):
        url             — required, target URL
        method          — HTTP method (default: POST)
        content_type    — Content-Type header (default: application/json)
        headers         — dict of additional headers; values support ${ENV_VAR} substitution
        timeout_seconds — request timeout (default: 10)

    The notification template renders to the request body. Errors are logged
    but never propagate — a failed webhook notification never affects the pipeline.
    """

    async def send(self, notification: NotificationConfig, context: dict) -> None:
        cfg = notification.config
        url = cfg.get("url")
        if not url:
            logger.warning("WebhookNotifier: no url in notification config — skipping")
            return

        method = cfg.get("method", "POST").upper()
        content_type = cfg.get("content_type", "application/json")
        timeout = float(cfg.get("timeout_seconds", 10))

        raw_headers: dict = cfg.get("headers", {})
        headers = {k: self._resolve_env(v) for k, v in raw_headers.items()}
        headers.setdefault("Content-Type", content_type)

        body = _jinja_env.from_string(notification.template).render(**context).strip()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(method, url, content=body.encode(), headers=headers)
                response.raise_for_status()
            logger.info(
                "WebhookNotifier: delivered to %s — HTTP %d", url, response.status_code
            )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "WebhookNotifier: HTTP %d from %s — %s",
                exc.response.status_code, url, exc.response.text[:200],
            )
        except Exception as exc:
            logger.error("WebhookNotifier: failed to deliver to %s — %s", url, exc)

    @staticmethod
    def _resolve_env(value: str) -> str:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value
