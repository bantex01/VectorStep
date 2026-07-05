"""Tests for the Teams-oriented /ui/approvals/{token} web approval page."""
import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.executors import human
from src.ui import router as ui_router

app = FastAPI()
app.include_router(ui_router)
client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_human_state():
    human._pending_approvals.clear()
    human._pending_meta.clear()
    yield
    human._pending_approvals.clear()
    human._pending_meta.clear()


def _make_pending(token: str, **meta_overrides):
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    human._pending_approvals[token] = future
    human._pending_meta[token] = {
        "message": "Approve the deploy?",
        "step": "approve-deploy",
        "pipeline": "release-pipeline",
        "team": "team-b",
        "created_at": "2026-01-01T00:00:00",
        **meta_overrides,
    }
    return future, loop


def test_get_pending_approval_renders_message():
    future, loop = _make_pending("tok-1")
    try:
        resp = client.get("/ui/approvals/tok-1")
        assert resp.status_code == 200
        assert "Approve the deploy?" in resp.text
        assert "release-pipeline" in resp.text
    finally:
        loop.close()


def test_get_unknown_token_renders_not_found():
    resp = client.get("/ui/approvals/does-not-exist")
    assert resp.status_code == 200
    assert "no longer valid" in resp.text


def test_post_approve_resolves_future():
    future, loop = _make_pending("tok-2")
    try:
        resp = client.post("/ui/approvals/tok-2/approve")
        assert resp.status_code == 200
        assert "Approved" in resp.text
        assert future.done()
        assert future.result() is True
    finally:
        loop.close()


def test_post_reject_resolves_future():
    future, loop = _make_pending("tok-3")
    try:
        resp = client.post("/ui/approvals/tok-3/reject")
        assert resp.status_code == 200
        assert "Rejected" in resp.text
        assert future.done()
        assert future.result() is False
    finally:
        loop.close()


def test_post_approve_unknown_token_renders_not_found():
    resp = client.post("/ui/approvals/does-not-exist/approve")
    assert resp.status_code == 200
    assert "no longer valid" in resp.text


def test_post_approve_twice_second_time_not_found():
    future, loop = _make_pending("tok-5")
    try:
        first = client.post("/ui/approvals/tok-5/approve")
        assert "Approved" in first.text

        # human executor pops pending state once it wakes up from the future — simulate that.
        human._pending_approvals.pop("tok-5", None)
        human._pending_meta.pop("tok-5", None)

        second = client.post("/ui/approvals/tok-5/approve")
        assert "no longer valid" in second.text
    finally:
        loop.close()
