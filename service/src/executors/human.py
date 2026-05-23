import asyncio
import logging
import os
import uuid

import httpx
from jinja2 import Environment, Undefined

from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig
from .base import BaseExecutor

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)

_SEND_MESSAGE_API = "https://api.telegram.org/bot{token}/sendMessage"

# token (str uuid) → asyncio.Future[bool]
# True  = approved, False = rejected.
# Resolved by the Telegram update poller when a button is clicked.
_pending_approvals: dict[str, "asyncio.Future[bool]"] = {}

_DEFAULT_TIMEOUT = 300  # seconds

# Set at startup via configure() from the loaded config so values from
# config.yaml are honoured; env vars are used as a fallback.
_configured_bot_token: str = ""
_configured_chat_id: str = ""


def configure(bot_token: str, chat_id: str) -> None:
    global _configured_bot_token, _configured_chat_id
    _configured_bot_token = bot_token
    _configured_chat_id = chat_id


def get_pending_approvals() -> dict[str, "asyncio.Future[bool]"]:
    return _pending_approvals


class HumanExecutor(BaseExecutor):
    """Pauses the pipeline and asks a human to approve or reject via Telegram.

    Sends an inline keyboard message to the configured Telegram chat. Waits
    up to step.timeout_seconds (default 300s) for a response. Returns:
        confidence=1.0, proceed=True   — approved
        confidence=0.0, proceed=True   — rejected (triggers on_low_confidence action)

    Raises RuntimeError on timeout so the runner marks the step as failed.

    executor_config keys: none required.
    Credentials read from TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.
    """

    def __init__(self) -> None:
        self._bot_token = _configured_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = _configured_chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        if not self._bot_token or not self._chat_id:
            raise RuntimeError(
                "HumanExecutor requires Telegram credentials — set notifications.telegram "
                "in config.yaml or TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID env vars"
            )

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        token = str(uuid.uuid4())
        timeout = step.timeout_seconds or _DEFAULT_TIMEOUT

        message_text = _jinja_env.from_string(step.prompt_template).render(**context)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        _pending_approvals[token] = future

        try:
            await self._send_approval_request(message_text, token)
            logger.info(
                "Human approval requested: step=%s token=%s timeout=%ss",
                step.name, token, timeout,
            )
            approved = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Human approval timed out: step=%s token=%s", step.name, token
            )
            raise RuntimeError(f"Human approval timed out after {timeout}s")
        finally:
            _pending_approvals.pop(token, None)

        if approved:
            logger.info("Human approved: step=%s token=%s", step.name, token)
            return LLMOutput(
                confidence=1.0,
                summary="Approved by human operator",
                next_step_context="Human reviewed and approved this step.",
                raw_response={"approved": True, "token": token},
            )

        logger.info("Human rejected: step=%s token=%s", step.name, token)
        return LLMOutput(
            confidence=0.0,
            summary="Rejected by human operator",
            next_step_context="Human reviewed and rejected this step.",
            raw_response={"approved": False, "token": token},
        )

    async def _send_approval_request(self, text: str, token: str) -> None:
        url = _SEND_MESSAGE_API.format(token=self._bot_token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "reply_markup": {
                "inline_keyboard": [[
                    {"text": "✅ Approve", "callback_data": f"approve:{token}"},
                    {"text": "❌ Reject", "callback_data": f"reject:{token}"},
                ]]
            },
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
