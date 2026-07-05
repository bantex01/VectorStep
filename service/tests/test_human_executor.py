"""Tests for the human executor's per-team channel routing (executor: human)."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.executors import human
from src.models.pipeline import StepConfig


@pytest.fixture(autouse=True)
def _reset_human_state():
    """Each test gets a clean slate — configure() and pending state are module globals."""
    human._pending_approvals.clear()
    human._pending_meta.clear()
    human.configure(human_approval={}, legacy_telegram={}, ui_base_url="http://localhost:8000")
    yield
    human._pending_approvals.clear()
    human._pending_meta.clear()


def _mock_response(status=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_body or {"ok": True})
    return resp


# ---------------------------------------------------------------------------
# Channel resolution
# ---------------------------------------------------------------------------

def test_resolve_channel_config_team_specific_wins():
    human.configure(
        human_approval={
            "default": {"channel": "telegram"},
            "teams": {"team-a": {"channel": "slack", "slack": {"bot_token": "x", "channel_id": "C1"}}},
        },
        legacy_telegram={},
        ui_base_url="http://x",
    )
    cfg = human._resolve_channel_config("team-a")
    assert cfg["channel"] == "slack"


def test_resolve_channel_config_falls_back_to_default():
    human.configure(
        human_approval={"default": {"channel": "telegram"}, "teams": {}},
        legacy_telegram={},
        ui_base_url="http://x",
    )
    cfg = human._resolve_channel_config("team-unknown")
    assert cfg["channel"] == "telegram"


def test_resolve_channel_config_falls_back_to_legacy_telegram():
    human.configure(
        human_approval={},
        legacy_telegram={"bot_token": "tok", "chat_id": "chat"},
        ui_base_url="http://x",
    )
    cfg = human._resolve_channel_config(None)
    assert cfg == {"channel": "telegram", "telegram": {"bot_token": "tok", "chat_id": "chat"}}


def test_resolve_channel_config_empty_when_nothing_configured():
    human.configure(human_approval={}, legacy_telegram={}, ui_base_url="http://x")
    assert human._resolve_channel_config("team-a") == {}


def test_build_channel_slack_missing_creds_raises():
    with pytest.raises(RuntimeError, match="Slack"):
        human._build_channel({"channel": "slack", "slack": {}})


def test_build_channel_msteams_missing_webhook_raises():
    with pytest.raises(RuntimeError, match="Teams"):
        human._build_channel({"channel": "msteams", "msteams": {}})


def test_build_channel_unknown_channel_raises():
    with pytest.raises(RuntimeError, match="Unknown"):
        human._build_channel({"channel": "carrier-pigeon"})


# ---------------------------------------------------------------------------
# SlackApprovalChannel / TeamsApprovalChannel payloads
# ---------------------------------------------------------------------------

async def test_slack_channel_posts_expected_payload():
    channel = human.SlackApprovalChannel(bot_token="xoxb-test", channel_id="C123")
    sent = {}

    async def fake_post(self, url, json=None, headers=None):
        sent["url"] = url
        sent["json"] = json
        sent["headers"] = headers
        return _mock_response(json_body={"ok": True})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        await channel.send("Approve deploy?", "tok-123")

    assert sent["url"] == human._SLACK_POST_MESSAGE_API
    assert sent["headers"]["Authorization"] == "Bearer xoxb-test"
    assert sent["json"]["channel"] == "C123"
    actions = sent["json"]["blocks"][1]["elements"]
    assert actions[0]["value"] == "approve:tok-123"
    assert actions[1]["value"] == "reject:tok-123"


async def test_slack_channel_raises_on_not_ok():
    channel = human.SlackApprovalChannel(bot_token="xoxb-test", channel_id="C123")

    async def fake_post(self, url, json=None, headers=None):
        return _mock_response(json_body={"ok": False, "error": "channel_not_found"})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        with pytest.raises(RuntimeError, match="channel_not_found"):
            await channel.send("hi", "tok")


async def test_teams_channel_posts_link_to_ui_approval_page():
    channel = human.TeamsApprovalChannel(webhook_url="https://flow.example.com/hook", ui_base_url="https://pork.example.com")
    sent = {}

    async def fake_post(self, url, json=None):
        sent["url"] = url
        sent["json"] = json
        return _mock_response()

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        await channel.send("Approve deploy?", "tok-456")

    assert sent["url"] == "https://flow.example.com/hook"
    assert "https://pork.example.com/ui/approvals/tok-456" in sent["json"]["text"]
    assert "Approve deploy?" in sent["json"]["text"]


# ---------------------------------------------------------------------------
# HumanExecutor.execute end-to-end (with a stubbed channel)
# ---------------------------------------------------------------------------

def _step(prompt="Approve?", timeout=1):
    return StepConfig(name="approve-step", executor="human", timeout_seconds=timeout, prompt_template=prompt)


async def test_execute_approved_via_resolve_approval():
    human.configure(
        human_approval={"default": {"channel": "slack", "slack": {"bot_token": "t", "channel_id": "c"}}},
        legacy_telegram={},
        ui_base_url="http://x",
    )

    async def fake_send(self, text, token):
        # Simulate the Slack listener resolving the button click shortly after send.
        asyncio.get_running_loop().call_later(0, lambda: human.resolve_approval(token, True))

    with patch.object(human.SlackApprovalChannel, "send", new=fake_send):
        result = await human.HumanExecutor().execute(_step(timeout=2), {"team": None, "pipeline_name": "p"})

    assert result.confidence == 1.0
    assert result.raw_response["approved"] is True


async def test_execute_rejected():
    human.configure(
        human_approval={"default": {"channel": "slack", "slack": {"bot_token": "t", "channel_id": "c"}}},
        legacy_telegram={},
        ui_base_url="http://x",
    )

    async def fake_send(self, text, token):
        asyncio.get_running_loop().call_later(0, lambda: human.resolve_approval(token, False))

    with patch.object(human.SlackApprovalChannel, "send", new=fake_send):
        result = await human.HumanExecutor().execute(_step(timeout=2), {"team": None, "pipeline_name": "p"})

    assert result.confidence == 0.0
    assert result.raw_response["approved"] is False


async def test_execute_timeout_raises_and_cleans_up_pending_state():
    human.configure(
        human_approval={"default": {"channel": "slack", "slack": {"bot_token": "t", "channel_id": "c"}}},
        legacy_telegram={},
        ui_base_url="http://x",
    )

    async def fake_send(self, text, token):
        pass  # never resolves — timeout should fire

    with patch.object(human.SlackApprovalChannel, "send", new=fake_send):
        with pytest.raises(RuntimeError, match="timed out"):
            await human.HumanExecutor().execute(_step(timeout=1), {"team": None, "pipeline_name": "p"})

    assert human._pending_approvals == {}
    assert human._pending_meta == {}


async def test_execute_no_channel_configured_raises():
    human.configure(human_approval={}, legacy_telegram={}, ui_base_url="http://x")
    with pytest.raises(RuntimeError, match="no approval channel configured"):
        await human.HumanExecutor().execute(_step(), {"team": "team-x", "pipeline_name": "p"})


def test_resolve_approval_unknown_token_returns_false():
    assert human.resolve_approval("does-not-exist", True) is False


def test_resolve_approval_already_done_returns_false():
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    future.set_result(True)
    human._pending_approvals["tok"] = future
    try:
        assert human.resolve_approval("tok", False) is False
    finally:
        human._pending_approvals.pop("tok", None)
        loop.close()
