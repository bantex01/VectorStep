import asyncio
import logging
import os
import uuid

import httpx
from jinja2 import Environment, Undefined

from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig
from ..utils import utc_now
from .base import BaseExecutor

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=Undefined)

_TELEGRAM_SEND_MESSAGE_API = "https://api.telegram.org/bot{token}/sendMessage"
_SLACK_POST_MESSAGE_API = "https://slack.com/api/chat.postMessage"

# token (str uuid) → asyncio.Future[bool]
# True  = approved, False = rejected.
# Resolved by whichever channel's callback listener sees the human's decision
# (Telegram poller, Slack Socket Mode listener, or the /ui/approvals POST route).
_pending_approvals: dict[str, "asyncio.Future[bool]"] = {}

# token → {"message", "step", "pipeline", "team", "created_at"} — rendered by the
# /ui/approvals/{token} page (the Teams flow links here instead of an in-chat button).
_pending_meta: dict[str, dict] = {}

_DEFAULT_TIMEOUT = 300  # seconds

# Set at startup via configure() from the loaded config so values from
# config.yaml are honoured; env vars are used as a legacy fallback.
_human_approval_cfg: dict = {}
_legacy_telegram_cfg: dict = {}
_ui_base_url: str = "http://localhost:8000"


def configure(
    human_approval: dict | None = None,
    legacy_telegram: dict | None = None,
    ui_base_url: str = "http://localhost:8000",
) -> None:
    """Configure per-team approval channel routing.

    human_approval: the human_approval config block — {"default": {...}, "teams": {...}}
    legacy_telegram: notifications.telegram config, used as the fallback channel when a
        team has no human_approval entry at all (keeps pre-existing deployments working
        unchanged).
    ui_base_url: base URL P-Ork's own UI is reachable at, used to build the
        /ui/approvals/{token} link sent to the Teams channel.
    """
    global _human_approval_cfg, _legacy_telegram_cfg, _ui_base_url
    _human_approval_cfg = human_approval or {}
    _legacy_telegram_cfg = legacy_telegram or {}
    _ui_base_url = ui_base_url.rstrip("/")


def get_pending_approvals() -> dict[str, "asyncio.Future[bool]"]:
    return _pending_approvals


def get_pending_meta(token: str) -> dict | None:
    return _pending_meta.get(token)


def resolve_approval(token: str, approved: bool) -> bool:
    """Resolve a pending approval by token. Returns False if the token is unknown/expired
    or already decided — used by every channel's callback listener/route."""
    future = _pending_approvals.get(token)
    if future is None or future.done():
        return False
    future.set_result(approved)
    return True


# ---------------------------------------------------------------------------
# Channel adapters — each renders the request into its platform and sends it.
# Resolution (turning a click/POST back into a decision) happens elsewhere:
# Telegram via telegram_poller.py, Slack via slack_poller.py, Teams via the
# /ui/approvals routes in ui.py — all three call resolve_approval() above.
# ---------------------------------------------------------------------------


class TelegramApprovalChannel:
    """Sends an inline-keyboard Approve/Reject message via the Telegram bot API."""

    def __init__(self, bot_token: str, chat_id: str):
        self._bot_token = bot_token
        self._chat_id = chat_id

    async def send(self, text: str, token: str) -> None:
        url = _TELEGRAM_SEND_MESSAGE_API.format(token=self._bot_token)
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


class SlackApprovalChannel:
    """Sends an interactive Approve/Reject message via the Slack Web API.

    Requires a bot token with chat:write scope. The button click is resolved
    by notifications/slack_poller.py's Socket Mode listener, not here.
    """

    def __init__(self, bot_token: str, channel_id: str):
        self._bot_token = bot_token
        self._channel_id = channel_id

    async def send(self, text: str, token: str) -> None:
        payload = {
            "channel": self._channel_id,
            "text": text,
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
                {
                    "type": "actions",
                    "block_id": f"pork_approval:{token}",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "✅ Approve"},
                            "style": "primary",
                            "action_id": "approve",
                            "value": f"approve:{token}",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "❌ Reject"},
                            "style": "danger",
                            "action_id": "reject",
                            "value": f"reject:{token}",
                        },
                    ],
                },
            ],
        }
        headers = {"Authorization": f"Bearer {self._bot_token}"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(_SLACK_POST_MESSAGE_API, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
            if not body.get("ok"):
                raise RuntimeError(f"Slack chat.postMessage failed: {body.get('error')}")


class TeamsApprovalChannel:
    """Posts a one-way notification to a Microsoft Teams channel (via a Power Automate
    webhook flow) linking to P-Ork's own /ui/approvals/{token} page.

    Teams Adaptive Card actions need a public Bot Framework callback endpoint, which this
    deployment doesn't expose — so the decision is made in P-Ork's UI instead of in-Teams.
    """

    def __init__(self, webhook_url: str, ui_base_url: str):
        self._webhook_url = webhook_url
        self._ui_base_url = ui_base_url.rstrip("/")

    async def send(self, text: str, token: str) -> None:
        approval_url = f"{self._ui_base_url}/ui/approvals/{token}"
        payload = {"text": f"{text}\n\nApprove or reject: {approval_url}"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(self._webhook_url, json=payload)
            response.raise_for_status()


def _resolve_channel_config(team: str | None) -> dict:
    """Look up which channel config applies to this run's team.

    Priority: human_approval.teams[team] → human_approval.default → legacy
    notifications.telegram (pre-existing deployments with no human_approval block at all).
    """
    teams_cfg = _human_approval_cfg.get("teams", {})
    if team and team in teams_cfg:
        return teams_cfg[team]

    default_cfg = _human_approval_cfg.get("default")
    if default_cfg:
        return default_cfg

    if _legacy_telegram_cfg.get("bot_token") and _legacy_telegram_cfg.get("chat_id"):
        return {"channel": "telegram", "telegram": _legacy_telegram_cfg}

    return {}


def _build_channel(cfg: dict):
    channel = cfg.get("channel")

    if channel == "telegram":
        tg = cfg.get("telegram") or _legacy_telegram_cfg
        bot_token = tg.get("bot_token") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = tg.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID", "")
        if not bot_token or not chat_id:
            raise RuntimeError(
                "Telegram approval channel missing bot_token/chat_id — set "
                "human_approval.*.telegram or notifications.telegram in config.yaml"
            )
        return TelegramApprovalChannel(bot_token, chat_id)

    if channel == "slack":
        slack = cfg.get("slack") or {}
        bot_token = slack.get("bot_token", "")
        channel_id = slack.get("channel_id", "")
        if not bot_token or not channel_id:
            raise RuntimeError(
                "Slack approval channel missing bot_token/channel_id in human_approval config"
            )
        return SlackApprovalChannel(bot_token, channel_id)

    if channel == "msteams":
        ms = cfg.get("msteams") or {}
        webhook_url = ms.get("webhook_url", "")
        if not webhook_url:
            raise RuntimeError(
                "Teams approval channel missing msteams.webhook_url in human_approval config"
            )
        return TeamsApprovalChannel(webhook_url, _ui_base_url)

    raise RuntimeError(f"Unknown human_approval channel: {channel!r}")


class HumanExecutor(BaseExecutor):
    """Pauses the pipeline and asks a human to approve or reject.

    The channel used (Telegram, Slack, or Microsoft Teams) is resolved per run from the
    step context's `team` field against the human_approval config — see README §
    "human — Human-in-the-Loop" for the full config shape. Waits up to
    step.timeout_seconds (default 300s) for a response. Returns:
        confidence=1.0, proceed=True   — approved
        confidence=0.0, proceed=True   — rejected (triggers on_low_confidence action)

    Raises RuntimeError on timeout so the runner marks the step as failed.

    executor_config keys: none required.
    """

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        token = str(uuid.uuid4())
        timeout = step.timeout_seconds or _DEFAULT_TIMEOUT
        team = context.get("team")

        message_text = _jinja_env.from_string(step.prompt_template).render(**context)

        channel_cfg = _resolve_channel_config(team)
        if not channel_cfg:
            raise RuntimeError(
                "HumanExecutor: no approval channel configured for team="
                f"{team!r} — set human_approval.default or human_approval.teams.{team} "
                "in config.yaml (or notifications.telegram for the legacy fallback)"
            )
        channel = _build_channel(channel_cfg)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        _pending_approvals[token] = future
        _pending_meta[token] = {
            "message": message_text,
            "step": step.name,
            "pipeline": context.get("pipeline_name"),
            "team": team,
            "created_at": utc_now().isoformat(),
        }

        try:
            await channel.send(message_text, token)
            logger.info(
                "Human approval requested: step=%s token=%s team=%s channel=%s timeout=%ss",
                step.name, token, team, channel_cfg.get("channel"), timeout,
            )
            approved = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Human approval timed out: step=%s token=%s", step.name, token
            )
            raise RuntimeError(f"Human approval timed out after {timeout}s")
        finally:
            _pending_approvals.pop(token, None)
            _pending_meta.pop(token, None)

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
