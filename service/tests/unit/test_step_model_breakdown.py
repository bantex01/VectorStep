"""Tests for analytics.get_step_model_breakdown and GET /steps/{name}/models —
the per-(agent, model, provider) rollup added so a caller can compare which
model performs best for a given step (success rate, tokens, duration, judged
accuracy), rather than get_step_stats' single blended-across-models number."""
from datetime import datetime

import httpx
import pytest

import src.main as main
from src.analytics import get_step_model_breakdown
from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, StepFeedback
from src.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_two_models(now: datetime):
    sf = get_session_factory()
    async with sf() as session:
        # agent-a / model-fast: 2 runs, both completed, cheap+quick.
        session.add(PipelineRun(
            id="run-1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production", triggered_at=now,
        ))
        session.add(PipelineStep(
            id="step-1", run_id="run-1", step_name="triage", step_index=0, executor="gateway",
            agent="agent-a", model="model-fast", provider="anthropic", prompt="", status="completed",
            executed_at=now, duration_ms=500, input_tokens=100, output_tokens=50,
        ))
        session.add(PipelineRun(
            id="run-2", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production", triggered_at=now,
        ))
        session.add(PipelineStep(
            id="step-2", run_id="run-2", step_name="triage", step_index=0, executor="gateway",
            agent="agent-a", model="model-fast", provider="anthropic", prompt="", status="completed",
            executed_at=now, duration_ms=700, input_tokens=120, output_tokens=60,
        ))
        # agent-a / model-slow: 1 run, failed, expensive+slow.
        session.add(PipelineRun(
            id="run-3", pipeline_name="p", source="generic", status="failed",
            normalised_context="{}", raw_payload="{}", stage="production", triggered_at=now,
        ))
        session.add(PipelineStep(
            id="step-3", run_id="run-3", step_name="triage", step_index=0, executor="gateway",
            agent="agent-a", model="model-slow", provider="anthropic", prompt="", status="failed",
            executed_at=now, duration_ms=5000, input_tokens=1000, output_tokens=500,
        ))
        await session.flush()
        session.add(StepFeedback(id="fb-1", step_id="step-1", run_id="run-1", pipeline_name="p", step_name="triage", outcome="correct"))
        session.add(StepFeedback(id="fb-2", step_id="step-2", run_id="run-2", pipeline_name="p", step_name="triage", outcome="correct"))
        session.add(StepFeedback(id="fb-3", step_id="step-3", run_id="run-3", pipeline_name="p", step_name="triage", outcome="incorrect"))
        await session.commit()


async def test_breakdown_separates_models_instead_of_blending(db):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_two_models(now)

    rows = await get_step_model_breakdown(get_session_factory(), "triage", time_range="all", stage="production")

    by_model = {r["model"]: r for r in rows}
    assert set(by_model) == {"model-fast", "model-slow"}

    fast = by_model["model-fast"]
    assert fast["runs_total"] == 2
    assert fast["success_rate"] == pytest.approx(1.0)
    assert fast["avg_input_tokens"] == 110  # (100 + 120) / 2
    assert fast["avg_output_tokens"] == 55
    assert fast["avg_duration_seconds"] == pytest.approx(0.6)  # (500+700)/2/1000
    assert fast["accuracy"]["correct_pct"] == 100.0

    slow = by_model["model-slow"]
    assert slow["runs_total"] == 1
    assert slow["success_rate"] == pytest.approx(0.0)
    assert slow["avg_input_tokens"] == 1000
    assert slow["avg_duration_seconds"] == pytest.approx(5.0)
    assert slow["accuracy"]["correct_pct"] == 0.0

    # Sorted by runs_total descending.
    assert [r["model"] for r in rows] == ["model-fast", "model-slow"]


async def test_breakdown_scoped_by_stage_and_time_range(db):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_two_models(now)
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="run-test", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="testing", triggered_at=now,
        ))
        session.add(PipelineStep(
            id="step-test", run_id="run-test", step_name="triage", step_index=0, executor="gateway",
            agent="agent-a", model="model-testing-only", provider="anthropic", prompt="", status="completed",
            executed_at=now, duration_ms=100, input_tokens=10, output_tokens=5,
        ))
        await session.commit()

    prod_rows = await get_step_model_breakdown(sf, "triage", time_range="all", stage="production")
    all_rows = await get_step_model_breakdown(sf, "triage", time_range="all", stage="all")

    assert "model-testing-only" not in {r["model"] for r in prod_rows}
    assert "model-testing-only" in {r["model"] for r in all_rows}


async def test_endpoint_returns_breakdown(db, monkeypatch, client):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_two_models(now)
    monkeypatch.setattr(main, "_step_library", {"triage": {"name": "triage"}})

    resp = await client.get("/steps/triage/models?time_range=all&stage=production")

    assert resp.status_code == 200
    body = resp.json()
    assert body["step_name"] == "triage"
    assert {r["model"] for r in body["breakdown"]} == {"model-fast", "model-slow"}


async def test_endpoint_404_for_unknown_step(db, monkeypatch, client):
    monkeypatch.setattr(main, "_step_library", {})

    resp = await client.get("/steps/does-not-exist/models")

    assert resp.status_code == 404


async def test_endpoint_rejects_invalid_stage(db, monkeypatch, client):
    monkeypatch.setattr(main, "_step_library", {"triage": {"name": "triage"}})

    resp = await client.get("/steps/triage/models?stage=bogus")

    assert resp.status_code == 400
