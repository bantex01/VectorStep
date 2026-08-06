"""Tests for the feedback_badge indicator shown on every runs table (dashboard,
/runs, pipeline detail) — marks whether a run's accuracy has been recorded, and
what it was marked as, without needing to open the run."""
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

from src.db.database import get_session_factory
from src.db.models import PipelineRun, RunFeedback
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig
from src.ui.helpers import _feedback_by_run_id
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
            id="r-correct", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", triggered_at=datetime.utcnow(),
        ))
        session.add(PipelineRun(
            id="r-partial", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", triggered_at=datetime.utcnow(),
        ))
        session.add(PipelineRun(
            id="r-incorrect", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", triggered_at=datetime.utcnow(),
        ))
        session.add(PipelineRun(
            id="r-unmarked", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", triggered_at=datetime.utcnow(),
        ))
        session.add(RunFeedback(id="fb1", run_id="r-correct", pipeline_name="p", outcome="correct"))
        session.add(RunFeedback(id="fb2", run_id="r-partial", pipeline_name="p", outcome="partial"))
        session.add(RunFeedback(id="fb3", run_id="r-incorrect", pipeline_name="p", outcome="incorrect"))
        # r-unmarked deliberately has no RunFeedback row
        await session.commit()
    return sf


async def test_feedback_by_run_id_returns_only_marked_runs(db):
    sf = await _seed()
    async with sf() as session:
        result = await _feedback_by_run_id(
            session, ["r-correct", "r-partial", "r-incorrect", "r-unmarked"]
        )
    assert result == {"r-correct": "correct", "r-partial": "partial", "r-incorrect": "incorrect"}
    assert "r-unmarked" not in result


async def test_feedback_by_run_id_empty_list_returns_empty_dict(db):
    sf = await _seed()
    async with sf() as session:
        assert await _feedback_by_run_id(session, []) == {}


async def test_runs_page_shows_all_four_badge_states(db, client):
    await _seed()

    resp = await client.get("/ui/runs")

    assert resp.status_code == 200
    body = resp.text
    assert body.count('title="Marked correct"') == 1
    assert body.count('title="Marked partial"') == 1
    assert body.count('title="Marked incorrect"') == 1
    assert body.count('title="Not yet marked"') == 1


async def test_dashboard_recent_runs_shows_badges(db, client):
    await _seed()

    resp = await client.get("/ui/")

    assert resp.status_code == 200
    assert 'title="Marked correct"' in resp.text
    assert 'title="Not yet marked"' in resp.text


async def test_pipeline_detail_recent_runs_shows_badges(tmp_path, db, client):
    await _seed()
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[StepConfig(name="s", executor="human")])
    app.state.pipelines = [pipeline]
    app.state.pipeline_dir = str(tmp_path)

    resp = await client.get("/ui/pipelines/p")

    assert resp.status_code == 200
    assert 'title="Marked correct"' in resp.text
    assert 'title="Not yet marked"' in resp.text
