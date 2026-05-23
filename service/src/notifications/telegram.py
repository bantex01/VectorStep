import logging

import httpx
from jinja2 import Environment, Undefined

from ..models.pipeline import NotificationConfig

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Sends pipeline notifications via Telegram bot API.

    Renders the notification template with the provided context dict and
    POSTs to the Telegram sendMessage endpoint.
    """

    def __init__(self, bot_token: str, chat_id: str):
        if not bot_token:
            raise ValueError("Telegram bot_token is required")
        if not chat_id:
            raise ValueError("Telegram chat_id is required")
        self._bot_token = bot_token
        self._chat_id = str(chat_id)

    _MAX_LENGTH = 4096
    _TRUNCATION_SUFFIX = "\n\n[truncated]"

    async def send(self, notification: NotificationConfig, context: dict) -> None:
        text = _jinja_env.from_string(notification.template).render(**context).strip()
        if len(text) > self._MAX_LENGTH:
            cut = self._MAX_LENGTH - len(self._TRUNCATION_SUFFIX)
            text = text[:cut] + self._TRUNCATION_SUFFIX

        url = _TELEGRAM_API.format(token=self._bot_token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                logger.info("Telegram notification sent: action context=%s", context.get("current_step"))
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Telegram API error: status=%d body=%s",
                exc.response.status_code,
                exc.response.text[:200],
            )
        except Exception as exc:
            logger.error("Telegram notification failed: %s", exc)
