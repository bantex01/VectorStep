import asyncio
import json
import logging

import httpx
import websockets

from ..executors.human import resolve_approval

logger = logging.getLogger(__name__)

_CONNECTIONS_OPEN_API = "https://slack.com/api/apps.connections.open"


async def poll_slack_events(app_token: str) -> None:
    """Maintain a Slack Socket Mode connection and resolve HITL approval buttons.

    Runs as a background asyncio task alongside the FastAPI app, one per distinct Slack
    app (app_token) referenced across human_approval config. Cancels cleanly on shutdown.

    Socket Mode needs no public HTTPS endpoint — the app opens an outbound WebSocket to
    Slack and receives interaction payloads over it, mirroring the Telegram long-poll
    approach in telegram_poller.py. Button clicks arrive as `block_actions` interactive
    payloads whose action `value` is `approve:<token>` / `reject:<token>` (see
    SlackApprovalChannel.send in executors/human.py), resolved via resolve_approval().
    """
    logger.info("Slack Socket Mode listener starting")

    while True:
        try:
            url = await _open_connection(app_token)
            async with websockets.connect(url) as ws:
                logger.info("Slack Socket Mode connected")
                async for raw in ws:
                    envelope = json.loads(raw)
                    envelope_id = envelope.get("envelope_id")
                    if envelope_id:
                        await ws.send(json.dumps({"envelope_id": envelope_id}))

                    if envelope.get("type") == "disconnect":
                        logger.info("Slack requested reconnect — reopening connection")
                        break

                    _handle_envelope(envelope)

        except asyncio.CancelledError:
            logger.info("Slack Socket Mode listener stopped")
            return
        except Exception as exc:
            logger.warning(
                "Slack Socket Mode connection error: %s — reconnecting in 5s", exc
            )
            await asyncio.sleep(5)


async def _open_connection(app_token: str) -> str:
    headers = {"Authorization": f"Bearer {app_token}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(_CONNECTIONS_OPEN_API, headers=headers)
        response.raise_for_status()
        body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"apps.connections.open failed: {body.get('error')}")
    return body["url"]


def _handle_envelope(envelope: dict) -> None:
    if envelope.get("type") != "interactive":
        return

    payload = envelope.get("payload") or {}
    if payload.get("type") != "block_actions":
        return

    for action in payload.get("actions", []):
        value = action.get("value", "")
        if ":" not in value:
            continue
        decision, token = value.split(":", 1)
        if decision not in ("approve", "reject"):
            continue

        if not resolve_approval(token, decision == "approve"):
            logger.debug("Received Slack action for unknown/expired token: %s", token)
            continue

        logger.info(
            "%s received for token=%s",
            "Approval" if decision == "approve" else "Rejection", token,
        )
