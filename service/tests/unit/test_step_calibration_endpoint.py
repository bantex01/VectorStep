"""Tests for analytics.get_step_calibration and GET /steps/{name}/calibration —
exposes the calibration bins that previously only existed on the
/ui/insights/steps page, per (agent, model, provider), so a caller can answer
"is this step's confidence score trustworthy for this agent/model" without
reconstructing it by sampling individual runs."""
from datetime import datetime

import httpx
import pytest

import src.main as main
from src.analytics import get_step_calibration
from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, StepFeedback
from src.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_overconfident_bucket(n: int = 25):
    """n runs at effective_confidence=0.95 (bin 0.9-1.0), but only marked
    'correct' 60% of the time — a validated bucket that should diverge from
    its predicted midpoint by >= 15 points."""
    sf = get_session_factory()
    async with sf() as session:
        for i in range(n):
            run_id = f"run-{i}"
            step_id = f"step-{i}"
            session.add(PipelineRun(
                id=run_id, pipeline_name="p", source="generic", status="completed",
                normalised_context="{}", raw_payload="{}", stage="production",
                triggered_at=datetime(2026, 1, 1),
            ))
            session.add(PipelineStep(
                id=step_id, run_id=run_id, step_name="triage", step_index=0, executor="gateway",
                agent="agent-a", model="model-x", provider="anthropic", prompt="", status="completed",
                executed_at=datetime(2026, 1, 1), effective_confidence=0.95,
            ))
        await session.flush()
        for i in range(n):
            # 60% correct against a midpoint of 0.95 -> diverges by 0.35, well
            # clear of the recommendation threshold's floating-point boundary.
            outcome = "correct" if i % 5 < 3 else "incorrect"
            session.add(StepFeedback(
                id=f"fb-{i}", step_id=f"step-{i}", run_id=f"run-{i}",
                pipeline_name="p", step_name="triage", outcome=outcome,
            ))
        await session.commit()


async def test_get_step_calibration_returns_bucket_per_agent_model(db):
    await _seed_overconfident_bucket()

    buckets = await get_step_calibration(get_session_factory(), "triage")

    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket["agent"] == "agent-a"
    assert bucket["model"] == "model-x"
    assert bucket["provider"] == "anthropic"
    assert bucket["total_n"] == 25
    assert len(bucket["bins"]) == 10  # bin_width=0.1 -> 10 bins covering 0-1

    validated_bin = next(b for b in bucket["bins"] if b["lo"] == pytest.approx(0.9))
    assert validated_bin["validated"] is True
    assert validated_bin["n"] == 25
    assert validated_bin["mean_label"] == pytest.approx(0.6)  # 15/25 correct


async def test_get_step_calibration_flags_divergent_bucket(db):
    await _seed_overconfident_bucket()

    buckets = await get_step_calibration(get_session_factory(), "triage")

    # predicted midpoint 95%, observed 60% -> diverges by 35 points, well over threshold
    assert buckets[0]["recommendation"] is not None
    assert "60%" in buckets[0]["recommendation"]
    assert "95%" in buckets[0]["recommendation"]


async def test_get_step_calibration_unvalidated_bucket_has_no_recommendation(db):
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="run-1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
            triggered_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineStep(
            id="step-1", run_id="run-1", step_name="triage", step_index=0, executor="gateway",
            agent="agent-a", model="model-x", provider="anthropic", prompt="", status="completed",
            executed_at=datetime(2026, 1, 1), effective_confidence=0.95,
        ))
        await session.flush()
        session.add(StepFeedback(id="fb-1", step_id="step-1", run_id="run-1", pipeline_name="p", step_name="triage", outcome="incorrect"))
        await session.commit()

    buckets = await get_step_calibration(get_session_factory(), "triage")

    assert buckets[0]["bins"][-1]["validated"] is False  # only 1 sample, n_min=20
    assert buckets[0]["recommendation"] is None


async def test_get_step_calibration_empty_for_unknown_step(db):
    buckets = await get_step_calibration(get_session_factory(), "does-not-exist")

    assert buckets == []


async def test_endpoint_returns_calibration_buckets(db, monkeypatch, client):
    await _seed_overconfident_bucket()
    monkeypatch.setattr(main, "_step_library", {"triage": {"name": "triage"}})

    resp = await client.get("/steps/triage/calibration")

    assert resp.status_code == 200
    body = resp.json()
    assert body["step_name"] == "triage"
    assert body["buckets"][0]["agent"] == "agent-a"
    assert body["buckets"][0]["recommendation"] is not None


async def test_endpoint_404_for_unknown_step(db, monkeypatch, client):
    monkeypatch.setattr(main, "_step_library", {})

    resp = await client.get("/steps/does-not-exist/calibration")

    assert resp.status_code == 404
