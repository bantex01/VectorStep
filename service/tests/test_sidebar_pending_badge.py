"""Tests for the pending-approvals count badge shown next to the Runs sidebar entry
(replaces a standalone top-level Approvals nav item — see base.html)."""
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


def _make_pending(token: str):
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    human._pending_approvals[token] = future
    human._pending_meta[token] = {
        "message": "Approve?",
        "step": "approve-step",
        "pipeline": "p",
        "run_id": "run-1",
        "team": None,
        "created_at": human.utc_now(),
    }
    return future, loop


def test_pending_count_zero_when_nothing_pending():
    assert human.pending_count() == 0


def test_pending_count_matches_number_of_pending_entries():
    _make_pending("tok-1")
    _make_pending("tok-2")
    assert human.pending_count() == 2


def test_approvals_list_page_has_no_badge_when_nothing_pending():
    resp = client.get("/ui/approvals")
    assert resp.status_code == 200
    assert "pending approval" not in resp.text  # badge title text absent


def test_approvals_list_page_shows_badge_count_in_nav():
    future, loop = _make_pending("tok-3")
    try:
        resp = client.get("/ui/approvals")
        assert resp.status_code == 200
        assert '"1 pending approval"' in resp.text or "1 pending approval" in resp.text
        assert 'href="/ui/approvals"' in resp.text
    finally:
        loop.close()
