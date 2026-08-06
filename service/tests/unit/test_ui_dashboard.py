"""Tests for the dashboard route: team count card, the pipeline activity table's
per-team breakdown / accuracy / token columns, and the live-reference-pricing panel."""
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

import src.ui as ui
from src import live_pricing, pricing
from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, RunFeedback
from src.ui import router as ui_router

app = FastAPI()
app.include_router(ui_router)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_team_count():
    yield
    ui.configure(team_count=0)


@pytest.fixture(autouse=True)
def _reset_pricing_and_catalog():
    original_table = pricing.get_table()
    original_catalog = live_pricing.get_catalog()
    yield
    pricing._table = original_table
    live_pricing._catalog = original_catalog


async def test_dashboard_empty_state_renders(db, client):
    ui.configure(team_count=0)

    resp = await client.get("/ui/")

    assert resp.status_code == 200
    assert "No runs in the last 7 days" in resp.text


async def test_dashboard_shows_configured_team_count(db, client):
    ui.configure(team_count=3)

    resp = await client.get("/ui/")

    assert resp.status_code == 200
    idx = resp.text.find("Teams</p>")
    assert ">3<" in resp.text[idx:idx + 100]


async def test_dashboard_pipeline_activity_shows_team_accuracy_tokens(db, client):
    sf = get_session_factory()

    async with sf() as session:
        session.add(PipelineRun(
            id="r1", pipeline_name="approval-test", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="barkham",
            triggered_at=datetime.utcnow(),
        ))
        session.add(PipelineStep(
            id="s1", run_id="r1", step_name="step", step_index=0, executor="gateway",
            agent="a", model="m", prompt="", status="completed",
            executed_at=datetime.utcnow(), input_tokens=100, output_tokens=50,
        ))
        session.add(RunFeedback(id="fb1", run_id="r1", pipeline_name="approval-test", outcome="correct"))
        await session.commit()

    resp = await client.get("/ui/")
    body = resp.text

    assert resp.status_code == 200
    assert "barkham: 1" in body
    assert "100%" in body
    assert "150" in body  # 100 input + 50 output tokens


async def test_dashboard_unattributed_team_bucketed(db, client):
    sf = get_session_factory()

    async with sf() as session:
        session.add(PipelineRun(
            id="r1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team=None,
            triggered_at=datetime.utcnow(),
        ))
        await session.commit()

    resp = await client.get("/ui/")

    assert resp.status_code == 200
    assert "unattributed: 1" in resp.text


# ---------------------------------------------------------------------------
# Dashboard — live reference pricing panel (under Backends/MCP servers/Models)
#
# Matched against each agent's *configured* live primary model (Gateway/OpenClaw
# agents.list), not run history — so these tests fake out the agent-fetch
# functions rather than seeding PipelineStep rows.
# ---------------------------------------------------------------------------

_OPENROUTER_CATALOG = [
    {"id": "anthropic/claude-3.5-sonnet", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
]


def _fake_agents(monkeypatch, gw_agents=None, oc_agents=None):
    async def _gw():
        return gw_agents or [], None

    async def _oc():
        return oc_agents or [], None

    monkeypatch.setattr(ui, "_fetch_vectorstep_gateway_agents", _gw)
    monkeypatch.setattr(ui, "_fetch_openclaw_agents", _oc)


async def test_dashboard_pricing_panel_shows_disabled_message_by_default(db, client):
    pricing.configure({"currency": "USD"})  # live_pricing.enabled defaults False

    resp = await client.get("/ui/")

    assert resp.status_code == 200
    assert "Live reference pricing" in resp.text
    assert "Live pricing is disabled" in resp.text


async def test_dashboard_pricing_panel_shows_table_when_enabled_and_matched(db, client, monkeypatch):
    pricing.configure({"currency": "USD", "live_pricing": {"enabled": True}})
    live_pricing._catalog = _OPENROUTER_CATALOG
    _fake_agents(monkeypatch, gw_agents=[{"name": "a1", "model": "anthropic/claude-sonnet-4-6"}])

    resp = await client.get("/ui/")

    assert resp.status_code == 200
    assert "Live reference pricing" in resp.text
    assert "Live pricing is disabled" not in resp.text
    assert "anthropic/claude-3.5-sonnet" in resp.text
    assert "Reference prices supplied by OpenRouter" in resp.text
    # Flat reference list — no provider column, no real/approx color coding (that
    # belongs on run detail's per-step badges, next to an actual cost figure).
    assert "(unknown)" not in resp.text
    assert "Native OpenRouter call" not in resp.text
    assert "fuzzy-matched to a different provider" not in resp.text


async def test_dashboard_pricing_panel_shows_no_match_message_when_catalog_not_fetched(db, client, monkeypatch):
    pricing.configure({"currency": "USD", "live_pricing": {"enabled": True}})
    live_pricing._catalog = None  # enabled, but no fetch has happened yet
    _fake_agents(monkeypatch, gw_agents=[{"name": "a1", "model": "anthropic/claude-sonnet-4-6"}])

    resp = await client.get("/ui/")

    assert resp.status_code == 200
    assert "No configured models matched the OpenRouter catalog yet" in resp.text
