"""Tests for live/approximate pricing display in the UI (SPEC-live-pricing.md):
per-step colored real/approx cost badges on run detail, and the Overview
page's live-reference-pricing panel."""
import httpx
import pytest
from fastapi import FastAPI

from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep
from src import live_pricing, pricing
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
def _reset_pricing_and_catalog():
    original_table = pricing.get_table()
    original_catalog = live_pricing.get_catalog()
    yield
    pricing._table = original_table
    live_pricing._catalog = original_catalog


_OPENROUTER_CATALOG = [
    {"id": "anthropic/claude-3.5-sonnet", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
]


async def _seed_run(sf, run_id: str, steps: list[dict]) -> None:
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="payments", stage="production",
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


async def test_run_detail_shows_amber_approx_badge_for_cross_provider_guess(db, client):
    pricing.configure({"currency": "USD"})  # no manual rate for this model
    live_pricing._catalog = _OPENROUTER_CATALOG
    sf = get_session_factory()
    await _seed_run(sf, "run-cross", [
        {"name": "s1", "model": "claude-sonnet-4-6", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": None},
    ])

    resp = await client.get("/ui/runs/run-cross")
    assert resp.status_code == 200
    assert "text-amber-400" in resp.text
    assert "may not reflect what you actually pay" in resp.text


async def test_run_detail_shows_green_badge_for_native_openrouter_call(db, client):
    pricing.configure({"currency": "USD"})
    live_pricing._catalog = _OPENROUTER_CATALOG
    sf = get_session_factory()
    await _seed_run(sf, "run-native", [
        {"name": "s1", "model": "anthropic/claude-3.5-sonnet", "provider": "openrouter", "input_tokens": 1000, "output_tokens": 500, "cost": None},
    ])

    resp = await client.get("/ui/runs/run-native")
    assert resp.status_code == 200
    assert "text-green-400" in resp.text
    assert "real published rate" in resp.text


async def test_run_detail_no_approx_badge_when_step_already_has_real_cost(db, client):
    pricing.configure({
        "currency": "USD",
        "models": [{"match": {"provider": "anthropic", "model": "claude-sonnet"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0}],
    })
    live_pricing._catalog = _OPENROUTER_CATALOG
    sf = get_session_factory()
    await _seed_run(sf, "run-real", [
        {"name": "s1", "model": "claude-sonnet-4-6", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": 0.0105},
    ])

    resp = await client.get("/ui/runs/run-real")
    assert resp.status_code == 200
    # Real cost is priced — no approximation badge/disclaimer should render alongside it
    # ("text-amber-400" alone isn't a safe check — it's also used by unrelated feedback badges).
    assert "may not reflect what you actually pay" not in resp.text
    assert "real published rate" not in resp.text


async def test_run_detail_no_approx_badge_when_no_match_found(db, client):
    pricing.configure({"currency": "USD"})
    live_pricing._catalog = _OPENROUTER_CATALOG
    sf = get_session_factory()
    await _seed_run(sf, "run-nomatch", [
        {"name": "s1", "model": "totally-unrelated-model", "provider": "mystery", "input_tokens": 1000, "output_tokens": 500, "cost": None},
    ])

    resp = await client.get("/ui/runs/run-nomatch")
    assert resp.status_code == 200
    assert "may not reflect what you actually pay" not in resp.text
    assert "real published rate" not in resp.text


# ---------------------------------------------------------------------------
# Overview page — live reference pricing panel
# ---------------------------------------------------------------------------

async def test_overview_shows_live_pricing_panel_when_enabled_and_matched(db, client):
    pricing.configure({"currency": "USD", "live_pricing": {"enabled": True}})
    live_pricing._catalog = _OPENROUTER_CATALOG
    sf = get_session_factory()
    await _seed_run(sf, "run-1", [
        {"name": "s1", "model": "claude-sonnet-4-6", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": None},
    ])

    resp = await client.get("/ui/insights")
    assert resp.status_code == 200
    assert "Live reference pricing" in resp.text
    assert "anthropic/claude-3.5-sonnet" in resp.text
    assert "Reference only" in resp.text
    # Flat reference list, not tied to any real paid cost — no real/approx color
    # coding here (that only makes sense next to an actual cost, i.e. run detail's
    # per-step badges, tested separately above).
    assert "text-green-400" not in resp.text
    assert "text-amber-400" not in resp.text


async def test_overview_no_panel_when_live_pricing_disabled(db, client):
    pricing.configure({"currency": "USD"})  # live_pricing.enabled defaults False
    live_pricing._catalog = _OPENROUTER_CATALOG
    sf = get_session_factory()
    await _seed_run(sf, "run-1", [
        {"name": "s1", "model": "claude-sonnet-4-6", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": None},
    ])

    resp = await client.get("/ui/insights")
    assert resp.status_code == 200
    assert "Live reference pricing" not in resp.text


async def test_overview_no_panel_when_catalog_not_yet_fetched(db, client):
    pricing.configure({"currency": "USD", "live_pricing": {"enabled": True}})
    live_pricing._catalog = None  # enabled, but no fetch has happened yet
    sf = get_session_factory()
    await _seed_run(sf, "run-1", [
        {"name": "s1", "model": "claude-sonnet-4-6", "provider": "anthropic", "input_tokens": 1000, "output_tokens": 500, "cost": None},
    ])

    resp = await client.get("/ui/insights")
    assert resp.status_code == 200
    assert "Live reference pricing" not in resp.text
