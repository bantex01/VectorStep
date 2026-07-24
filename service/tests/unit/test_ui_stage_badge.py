"""Tests for stage=testing/production UI behaviour: the TESTING badge on browse
surfaces, the stage filter on /ui/runs, and production-only exclusion on rollup
surfaces (stat cards, pipeline list success rate, pipeline detail totals)."""
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig
from src.ui import router as ui_router

app = FastAPI()
app.include_router(ui_router)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed():
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="r-prod", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", triggered_at=datetime.utcnow(),
            stage="production",
        ))
        session.add(PipelineRun(
            id="r-test", pipeline_name="p", source="generic", status="failed",
            normalised_context="{}", raw_payload="{}", triggered_at=datetime.utcnow(),
            stage="testing",
        ))
        session.add(PipelineStep(
            id="s-prod", run_id="r-prod", step_name="s", step_index=0, executor="gateway",
            agent="a", model="m", prompt="", status="completed",
            executed_at=datetime.utcnow(), input_tokens=100, output_tokens=50,
        ))
        session.add(PipelineStep(
            id="s-test", run_id="r-test", step_name="s", step_index=0, executor="gateway",
            agent="a", model="m", prompt="", status="failed",
            executed_at=datetime.utcnow(), input_tokens=999, output_tokens=999,
        ))
        await session.commit()
    return sf


async def test_runs_page_shows_testing_badge_only_on_testing_run(db, client):
    await _seed()
    resp = await client.get("/ui/runs")
    assert resp.status_code == 200
    assert resp.text.count(">TESTING<") == 1


async def test_runs_page_stage_filter_shows_only_testing(db, client):
    await _seed()
    resp = await client.get("/ui/runs?stage=testing")
    assert resp.status_code == 200
    # Only the testing run should appear in the (browse) list — one row, one badge.
    assert resp.text.count(">TESTING<") == 1
    assert resp.text.count("<tr class=\"hover:bg-zinc-800 transition-colors\">") == 1


async def test_runs_page_stage_filter_production_excludes_testing_run(db, client):
    await _seed()
    resp = await client.get("/ui/runs?stage=production")
    assert resp.status_code == 200
    assert ">TESTING<" not in resp.text
    assert resp.text.count("<tr class=\"hover:bg-zinc-800 transition-colors\">") == 1


async def test_runs_page_stat_cards_exclude_testing_run(db, client):
    await _seed()
    resp = await client.get("/ui/runs")
    assert resp.status_code == 200
    # "Runs" stat card = total_matching, scoped to production only -> 1, not 2.
    idx = resp.text.find("Runs</p>")
    assert ">1<" in resp.text[idx:idx + 200]


async def test_dashboard_recent_runs_shows_testing_badge(db, client):
    await _seed()
    resp = await client.get("/ui/")
    assert resp.status_code == 200
    assert ">TESTING<" in resp.text


async def test_dashboard_counts_exclude_testing_run(db, client):
    await _seed()
    resp = await client.get("/ui/")
    assert resp.status_code == 200
    # counts_all/counts_24h are production-only -> 1 run total, not 2.
    assert "1 all time" in resp.text


async def test_pipeline_detail_shows_stage_badge_and_excludes_testing_from_totals(tmp_path, db, client):
    await _seed()
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[StepConfig(name="s", executor="gateway")])
    app.state.pipelines = [pipeline]
    app.state.pipeline_dir = str(tmp_path)

    resp = await client.get("/ui/pipelines/p")

    assert resp.status_code == 200
    # Pipeline-level badge (from in-memory PipelineConfig.stage, default "testing")
    assert ">TESTING<" in resp.text
    # total_runs stat card reflects only the production run (1), not both (2)
    idx = resp.text.find('<p class="text-2xl font-semibold text-zinc-50">')
    assert idx != -1
    assert "1" in resp.text[idx:idx + 60]


async def test_pipelines_list_shows_stage_badge(db, client):
    await _seed()

    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[StepConfig(name="s", executor="gateway")])
    app.state.pipelines = [pipeline]

    resp = await client.get("/ui/pipelines")

    assert resp.status_code == 200
    assert ">TESTING<" in resp.text
