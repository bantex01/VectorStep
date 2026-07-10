"""Tests for NotifyExecutor (executor: notify)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.executors.notify import NotifyExecutor, _render_values
from src.models.pipeline import StepConfig
from src.models.llm import LLMOutput


def _step(url="https://hooks.example.com/alert", payload=None, headers=None, method="POST"):
    cfg = {"url": url, "method": method}
    if payload is not None:
        cfg["payload"] = payload
    if headers is not None:
        cfg["headers"] = headers
    return StepConfig(name="notify-step", executor="notify", executor_config=cfg)


def _mock_response(status=200, text="ok"):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _render_values helper
# ---------------------------------------------------------------------------

def test_render_string():
    assert _render_values("hello {{name}}", {"name": "world"}) == "hello world"


def test_render_dict_values():
    result = _render_values({"text": "hi {{who}}", "count": 3}, {"who": "team"})
    assert result == {"text": "hi team", "count": 3}


def test_render_list():
    result = _render_values(["{{a}}", "{{b}}"], {"a": "x", "b": "y"})
    assert result == ["x", "y"]


def test_render_nested():
    result = _render_values(
        {"blocks": [{"text": "alert: {{summary}}"}]},
        {"summary": "disk full"},
    )
    assert result == {"blocks": [{"text": "alert: disk full"}]}


def test_render_non_string_passthrough():
    assert _render_values(42, {}) == 42
    assert _render_values(True, {}) is True
    assert _render_values(None, {}) is None


# ---------------------------------------------------------------------------
# NotifyExecutor.execute
# ---------------------------------------------------------------------------

async def test_execute_posts_rendered_payload():
    step = _step(payload={"text": "Alert: {{summary}}"})
    executor = NotifyExecutor()
    ctx = {"summary": "disk full on prod"}

    sent = {}

    async def fake_request(self, method, url, content=None, headers=None):
        sent["method"] = method
        sent["url"] = url
        sent["body"] = json.loads(content)
        sent["headers"] = headers
        return _mock_response()

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        result = await executor.execute(step, ctx)

    assert sent["method"] == "POST"
    assert sent["url"] == "https://hooks.example.com/alert"
    assert sent["body"]["text"] == "Alert: disk full on prod"
    assert sent["headers"]["Content-Type"] == "application/json"
    assert result.confidence == 1.0
    assert "200" in result.summary


async def test_execute_missing_url_raises():
    step = StepConfig(name="x", executor="notify", executor_config={})
    with pytest.raises(ValueError, match="url is required"):
        await NotifyExecutor().execute(step, {})


async def test_execute_resolves_env_var_in_url(monkeypatch):
    monkeypatch.setenv("MY_HOOK_URL", "https://real.example.com/hook")
    step = _step(url="${MY_HOOK_URL}")
    executor = NotifyExecutor()

    sent = {}

    async def fake_request(self, method, url, content=None, headers=None):
        sent["url"] = url
        return _mock_response()

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        await executor.execute(step, {})

    assert sent["url"] == "https://real.example.com/hook"


async def test_execute_resolves_env_var_in_headers(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret-token")
    step = _step(headers={"Authorization": "${MY_TOKEN}"})
    executor = NotifyExecutor()

    sent = {}

    async def fake_request(self, method, url, content=None, headers=None):
        sent["headers"] = headers
        return _mock_response()

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        await executor.execute(step, {})

    assert sent["headers"]["Authorization"] == "secret-token"


async def test_execute_http_error_propagates():
    step = _step()
    executor = NotifyExecutor()

    resp = MagicMock()
    resp.status_code = 500
    resp.text = "server error"
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        "500", request=MagicMock(), response=resp
    ))

    async def fake_request(*args, **kwargs):
        return resp

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        with pytest.raises(httpx.HTTPStatusError):
            await executor.execute(step, {})


async def test_execute_empty_payload_sends_empty_object():
    step = _step()  # no payload
    executor = NotifyExecutor()

    sent = {}

    async def fake_request(self, method, url, content=None, headers=None):
        sent["body"] = json.loads(content)
        return _mock_response()

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        await executor.execute(step, {})

    assert sent["body"] == {}


async def test_execute_custom_method():
    step = _step(method="PUT")
    executor = NotifyExecutor()

    sent = {}

    async def fake_request(self, method, url, content=None, headers=None):
        sent["method"] = method
        return _mock_response()

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        await executor.execute(step, {})

    assert sent["method"] == "PUT"


# ---------------------------------------------------------------------------
# stage=testing gating (_testing in context)
# ---------------------------------------------------------------------------

async def test_execute_suppressed_when_testing():
    step = _step(payload={"text": "Alert: {{summary}}"})
    executor = NotifyExecutor()
    ctx = {"summary": "disk full on prod", "_testing": True}

    with patch.object(httpx.AsyncClient, "request", new=AsyncMock()) as mock_request:
        result = await executor.execute(step, ctx)

    mock_request.assert_not_awaited()
    assert result.confidence == 1.0
    assert result.raw_response["suppressed_testing"] is True
    assert result.raw_response["url"] == "https://hooks.example.com/alert"
    assert "suppressed" in result.summary


async def test_execute_not_suppressed_when_testing_false():
    step = _step(payload={"text": "hi"})
    executor = NotifyExecutor()

    async def fake_request(self, method, url, content=None, headers=None):
        return _mock_response()

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        result = await executor.execute(step, {"_testing": False})

    assert result.raw_response.get("suppressed_testing") is None
    assert "200" in result.summary


async def test_execute_not_suppressed_when_testing_absent():
    """Context without _testing at all (e.g. older/unrelated callers) behaves as production."""
    step = _step(payload={"text": "hi"})
    executor = NotifyExecutor()

    async def fake_request(self, method, url, content=None, headers=None):
        return _mock_response()

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        result = await executor.execute(step, {})

    assert result.raw_response.get("suppressed_testing") is None


# ---------------------------------------------------------------------------
# _resolve_env
# ---------------------------------------------------------------------------

def test_resolve_env_with_placeholder(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "hello")
    assert NotifyExecutor._resolve_env("${TEST_VAR}") == "hello"


def test_resolve_env_plain_passthrough():
    assert NotifyExecutor._resolve_env("plain-value") == "plain-value"


def test_resolve_env_unset_returns_empty(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    assert NotifyExecutor._resolve_env("${MISSING_VAR}") == ""
