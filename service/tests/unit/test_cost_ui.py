"""Tests for cost display in the UI (SPEC-cost-accounting.md §5): run-detail cost
card, the teams page's month-to-date budget bar, and the "unpriced steps"
annotation appearing when and only when NULL-cost steps were excluded from a
rollup."""
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep
from src import pricing
from src.ui import router as ui_router
from src.utils import utc_now

app = FastAPI()
app.include_router(ui_router)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_pricing_table():
    original = pricing.get_table()
    yield
    pricing._table = original


async def _seed_run(sf, run_id: str, team: str | None, steps: list[dict]) -> None:
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team=team, stage="production",
            triggered_at=utc_now(), completed_at=utc_now(),
        ))
        await session.flush()
        for i, s in enumerate(steps):
            session.add(PipelineStep(
                id=f"{run_id}-step-{i}", run_id=run_id, step_name=s.get("name", f"s{i}"),
                step_index=i, executor="gateway", model=s.get("model"), provider=s.get("provider"),
                prompt="", status="completed", executed_at=utc_now(),
                input_tokens=s.get("input_tokens"), output_tokens=s.get("output_tokens"),
                cost=s.get("cost"),
            ))
        await session.commit()


async def test_run_detail_shows_cost_card_and_no_unpriced_annotation_when_fully_priced(db, client):
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })
    sf = get_session_factory()
    await _seed_run(sf, "run-priced", "payments", [
        {"model": "claude-sonnet", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": 0.0105},
    ])

    resp = await client.get("/ui/runs/run-priced")
    assert resp.status_code == 200
    body = resp.text

    assert "$0.01" in body
    assert "unpriced steps" not in body.split("<script>")[0]


async def test_run_detail_shows_unpriced_annotation_when_a_step_has_no_rate(db, client):
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })
    sf = get_session_factory()
    await _seed_run(sf, "run-mixed", "payments", [
        {"name": "priced-step", "model": "claude-sonnet", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": 0.0105},
        {"name": "unpriced-step", "model": "unknown-model", "provider": "mystery", "input_tokens": 100, "output_tokens": 50, "cost": None},
    ])

    resp = await client.get("/ui/runs/run-mixed")
    assert resp.status_code == 200
    body = resp.text

    assert "unpriced steps: 1" in body


async def test_run_detail_shows_not_priced_when_no_steps_are_priced_at_all(db, client):
    pricing.configure(None)  # no pricing table at all — nothing can be priced
    sf = get_session_factory()
    await _seed_run(sf, "run-unpriced", "payments", [
        {"name": "s1", "model": "claude-sonnet", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": None},
    ])

    resp = await client.get("/ui/runs/run-unpriced")
    assert resp.status_code == 200
    body = resp.text

    assert "not priced" in body
    assert "unpriced steps: 1" in body


async def test_teams_page_shows_budget_bar_for_configured_team(db, client):
    pricing.configure({"currency": "USD", "team_budgets": {"payments": 5.0}})
    sf = get_session_factory()
    await _seed_run(sf, "run-budget", "payments", [
        {"model": "claude-sonnet", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": 1.0},
    ])

    resp = await client.get("/ui/insights/teams")
    assert resp.status_code == 200
    body = resp.text

    assert "Month-to-date budgets" in body
    assert "payments" in body
    # $1.00 spend / $5.00 budget = 20%
    assert "20%" in body


async def test_teams_page_no_budget_section_when_no_team_budgets_configured(db, client):
    pricing.configure({"currency": "USD"})  # no team_budgets at all
    sf = get_session_factory()
    await _seed_run(sf, "run-no-budget", "payments", [
        {"model": "claude-sonnet", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": 1.0},
    ])

    resp = await client.get("/ui/insights/teams")
    assert resp.status_code == 200

    assert "Month-to-date budgets" not in resp.text
