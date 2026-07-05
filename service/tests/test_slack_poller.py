"""Tests for the Slack Socket Mode listener's pure logic (envelope parsing, connection open)."""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.executors import human
from src.notifications import slack_poller


@pytest.fixture(autouse=True)
def _reset_pending():
    human._pending_approvals.clear()
    yield
    human._pending_approvals.clear()


def _mock_response(json_body):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_body)
    return resp


async def test_open_connection_returns_url():
    async def fake_post(self, url, headers=None):
        assert headers["Authorization"] == "Bearer xapp-test"
        return _mock_response({"ok": True, "url": "wss://example.com/link"})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        url = await slack_poller._open_connection("xapp-test")

    assert url == "wss://example.com/link"


async def test_open_connection_raises_on_not_ok():
    async def fake_post(self, url, headers=None):
        return _mock_response({"ok": False, "error": "invalid_auth"})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        with pytest.raises(RuntimeError, match="invalid_auth"):
            await slack_poller._open_connection("xapp-bad")


def _pending_future():
    import asyncio
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    return loop, future


def test_handle_envelope_approve_resolves_future():
    loop, future = _pending_future()
    human._pending_approvals["tok-1"] = future
    envelope = {
        "type": "interactive",
        "payload": {
            "type": "block_actions",
            "actions": [{"action_id": "approve", "value": "approve:tok-1"}],
        },
    }
    try:
        slack_poller._handle_envelope(envelope)
        assert future.done()
        assert future.result() is True
    finally:
        loop.close()


def test_handle_envelope_reject_resolves_future():
    loop, future = _pending_future()
    human._pending_approvals["tok-2"] = future
    envelope = {
        "type": "interactive",
        "payload": {
            "type": "block_actions",
            "actions": [{"action_id": "reject", "value": "reject:tok-2"}],
        },
    }
    try:
        slack_poller._handle_envelope(envelope)
        assert future.done()
        assert future.result() is False
    finally:
        loop.close()


def test_handle_envelope_ignores_non_interactive():
    envelope = {"type": "events_api", "payload": {}}
    slack_poller._handle_envelope(envelope)  # should not raise


def test_handle_envelope_ignores_non_block_actions():
    envelope = {"type": "interactive", "payload": {"type": "view_submission"}}
    slack_poller._handle_envelope(envelope)  # should not raise


def test_handle_envelope_unknown_token_is_noop():
    envelope = {
        "type": "interactive",
        "payload": {
            "type": "block_actions",
            "actions": [{"action_id": "approve", "value": "approve:does-not-exist"}],
        },
    }
    slack_poller._handle_envelope(envelope)  # should not raise
