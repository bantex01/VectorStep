"""Tests for the dashboard route: team count card, and the pipeline activity table's
per-team breakdown / accuracy / token columns."""
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.ui as ui
from src.db.database import create_tables, init_db, get_session_factory
from src.db.models import PipelineRun, PipelineStep, RunFeedback
from src.ui import router as ui_router

app = FastAPI()
app.include_router(ui_router)
client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_team_count():
    yield
    ui.configure(team_count=0)


async def test_dashboard_empty_state_renders(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    ui.configure(team_count=0)

    resp = client.get("/ui/")

    assert resp.status_code == 200
    assert "No runs in the last 7 days" in resp.text


async def test_dashboard_shows_configured_team_count(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    ui.configure(team_count=3)

    resp = client.get("/ui/")

    assert resp.status_code == 200
    idx = resp.text.find("Teams</p>")
    assert ">3<" in resp.text[idx:idx + 100]


async def test_dashboard_pipeline_activity_shows_team_accuracy_tokens(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
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

    resp = client.get("/ui/")
    body = resp.text

    assert resp.status_code == 200
    assert "barkham: 1" in body
    assert "100%" in body
    assert "150" in body  # 100 input + 50 output tokens


async def test_dashboard_unattributed_team_bucketed(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    sf = get_session_factory()

    async with sf() as session:
        session.add(PipelineRun(
            id="r1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team=None,
            triggered_at=datetime.utcnow(),
        ))
        await session.commit()

    resp = client.get("/ui/")

    assert resp.status_code == 200
    assert "unattributed: 1" in resp.text
