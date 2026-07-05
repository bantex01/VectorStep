import asyncio
import logging
import shlex

import httpx

from ..executors.human import resolve_approval

logger = logging.getLogger(__name__)

_GET_UPDATES_API = "https://api.telegram.org/bot{token}/getUpdates"
_ANSWER_CALLBACK_API = "https://api.telegram.org/bot{token}/answerCallbackQuery"
_DELETE_WEBHOOK_API = "https://api.telegram.org/bot{token}/deleteWebhook"
_SEND_MESSAGE_API = "https://api.telegram.org/bot{token}/sendMessage"


def _parse_run_command(text: str) -> tuple[str, dict] | None:
    """Parse /run <pipeline> [key=value ...] → (pipeline_name, data) or None.

    Handles the /run@BotName variant that Telegram appends in group chats.
    Any tokens after the pipeline name in key=value form are collected into data.
    """
    try:
        parts = shlex.split(text.strip())
    except ValueError:
        parts = text.strip().split()
    if not parts:
        return None
    command = parts[0].lower()
    if "@" in command:
        command = command[: command.index("@")]
    if command != "/run":
        return None
    if len(parts) < 2:
        return None

    pipeline_name = parts[1]
    data: dict = {}
    for token in parts[2:]:
        if "=" in token:
            k, _, v = token.partition("=")
            data[k.strip()] = v.strip()

    return pipeline_name, data


async def poll_telegram_updates(
    bot_token: str,
    webhook_base_url: str = "http://localhost:8000",
    allowed_chat_id: str | int | None = None,
    webhook_token: str | None = None,
) -> None:
    """Long-poll Telegram getUpdates, resolve HITL approval futures, and handle /run commands.

    Runs as a background asyncio task alongside the FastAPI app. Cancels cleanly
    on shutdown.

    /run <pipeline-name> [key=value ...] triggers a generic webhook POST to the
    local service. Commands from chats other than allowed_chat_id are ignored when
    allowed_chat_id is set. webhook_token, if set, is sent as a Bearer token on that
    POST — required once auth.teams/auth.token is configured (see README §3b),
    otherwise the call 401s like any other unauthenticated caller.
    """
    offset = 0
    updates_url = _GET_UPDATES_API.format(token=bot_token)
    answer_url = _ANSWER_CALLBACK_API.format(token=bot_token)
    delete_webhook_url = _DELETE_WEBHOOK_API.format(token=bot_token)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(delete_webhook_url)
            resp.raise_for_status()
        logger.info("Telegram webhook cleared — starting long-poll")
    except Exception as exc:
        logger.warning("Failed to delete Telegram webhook (will try polling anyway): %s", exc)

    logger.info("Telegram update poller started")

    while True:
        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                response = await client.get(
                    updates_url,
                    params={
                        "offset": offset,
                        "timeout": 30,
                        "allowed_updates": ["callback_query", "message"],
                    },
                )
                response.raise_for_status()
                data = response.json()

            for update in data.get("result", []):
                offset = update["update_id"] + 1

                # --- HITL approval callback ---
                callback = update.get("callback_query")
                if callback:
                    callback_id = callback.get("id")
                    callback_data = callback.get("data", "")

                    try:
                        async with httpx.AsyncClient(timeout=10.0) as ack_client:
                            await ack_client.post(
                                answer_url, json={"callback_query_id": callback_id}
                            )
                    except Exception as exc:
                        logger.debug("Failed to ack callback query: %s", exc)

                    if ":" not in callback_data:
                        continue

                    action, token = callback_data.split(":", 1)
                    if action not in ("approve", "reject"):
                        continue

                    if not resolve_approval(token, action == "approve"):
                        logger.debug(
                            "Received callback for unknown/expired token: %s", token
                        )
                        continue

                    logger.info(
                        "%s received for token=%s",
                        "Approval" if action == "approve" else "Rejection", token,
                    )
                    continue

                # --- /run command ---
                message = update.get("message")
                if not message:
                    continue

                text = message.get("text", "")
                chat_id = message.get("chat", {}).get("id")

                if not text or not chat_id:
                    continue

                if allowed_chat_id and str(chat_id) != str(allowed_chat_id):
                    logger.debug("Ignoring message from unauthorised chat %s", chat_id)
                    continue

                if not text.startswith("/run"):
                    continue

                parsed = _parse_run_command(text)
                if not parsed:
                    await _send_message(
                        bot_token, chat_id,
                        "Usage: /run &lt;pipeline-name&gt; [key=value ...]"
                    )
                    continue

                pipeline_name, run_data = parsed
                await _trigger_pipeline(
                    bot_token=bot_token,
                    webhook_base_url=webhook_base_url,
                    chat_id=chat_id,
                    pipeline_name=pipeline_name,
                    data=run_data,
                    webhook_token=webhook_token,
                )

        except asyncio.CancelledError:
            logger.info("Telegram update poller stopped")
            return
        except Exception as exc:
            logger.warning(
                "Telegram update poller error: %s — retrying in 5s", exc
            )
            await asyncio.sleep(5)


async def _trigger_pipeline(
    bot_token: str,
    webhook_base_url: str,
    chat_id: int,
    pipeline_name: str,
    data: dict,
    webhook_token: str | None = None,
) -> None:
    """POST a generic webhook to the local service and confirm to the user."""
    webhook_url = f"{webhook_base_url}/webhook?source=generic"
    payload = {
        "pipeline": pipeline_name,
        "source": "telegram",
        "summary": f"Triggered from Telegram (chat {chat_id})",
        "data": data,
    }
    headers = {"Authorization": f"Bearer {webhook_token}"} if webhook_token else {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook_url, json=payload, headers=headers)
            resp.raise_for_status()
            run_id = resp.json().get("run_id", "unknown")

        logger.info(
            "Pipeline '%s' triggered from Telegram — run_id=%s", pipeline_name, run_id
        )
        short_id = run_id[:8] if run_id != "unknown" else "unknown"
        await _send_message(
            bot_token, chat_id,
            f"⏳ Pipeline <code>{pipeline_name}</code> started "
            f"(run <code>{short_id}…</code>) — I'll notify you when it's done."
        )

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Failed to trigger pipeline '%s': HTTP %s — %s",
            pipeline_name, exc.response.status_code, exc.response.text,
        )
        await _send_message(
            bot_token, chat_id,
            f"❌ Failed to trigger <code>{pipeline_name}</code>: {exc.response.text[:200]}"
        )
    except Exception as exc:
        logger.error("Failed to trigger pipeline '%s': %s", pipeline_name, exc)
        await _send_message(
            bot_token, chat_id,
            f"❌ Could not reach the pipeline service: {exc}"
        )


async def _send_message(bot_token: str, chat_id: int, text: str) -> None:
    url = _SEND_MESSAGE_API.format(token=bot_token)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
            })
    except Exception as exc:
        logger.debug("Failed to send Telegram message: %s", exc)
