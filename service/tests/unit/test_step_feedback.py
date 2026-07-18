"""Tests for per-step accuracy feedback: POST/GET /runs/{run_id}/steps/{step_name}/feedback,
the Steps Insights rollup, and the pork_step_feedback_total metric."""
from datetime import datetime

from fastapi.testclient import TestClient

from src.db.database import create_tables, get_session_factory, init_db
from src.db.models import PipelineRun, PipelineStep, StepFeedback
from src.main import app
from src.metrics import MetricsData, PorkCollector, fetch_metrics_data

client = TestClient(app)


def _find_family(families, sample_name):
    return next(f for f in families if any(s.name == sample_name for s in f.samples))


async def _seed_run_with_step(tmp_path, step_name="triage", run_id="run-1", stage="production"):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage=stage,
            triggered_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineStep(
            id=f"step-{run_id}-{step_name.replace('/', '-')}", run_id=run_id, step_name=step_name,
            step_index=0, executor="gateway", agent="agent-x", model="claude-sonnet-5",
            provider="anthropic", prompt="", status="completed",
            executed_at=datetime(2026, 1, 1),
        ))
        await session.commit()
    return sf


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def test_post_valid_outcome_creates_row(tmp_path):
    await _seed_run_with_step(tmp_path)

    resp = client.post("/runs/run-1/steps/triage/feedback", json={"outcome": "correct", "notes": "looks right"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run-1"
    assert body["step_name"] == "triage"
    assert body["outcome"] == "correct"
    assert body["notes"] == "looks right"

    sf = get_session_factory()
    async with sf() as session:
        from sqlalchemy import select
        result = await session.execute(select(StepFeedback).where(StepFeedback.run_id == "run-1"))
        rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].outcome == "correct"


async def test_post_again_upserts(tmp_path):
    await _seed_run_with_step(tmp_path)

    client.post("/runs/run-1/steps/triage/feedback", json={"outcome": "correct"})
    resp = client.post("/runs/run-1/steps/triage/feedback", json={"outcome": "incorrect", "notes": "changed my mind"})

    assert resp.status_code == 200
    assert resp.json()["outcome"] == "incorrect"

    sf = get_session_factory()
    async with sf() as session:
        from sqlalchemy import select
        result = await session.execute(select(StepFeedback).where(StepFeedback.run_id == "run-1"))
        rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].outcome == "incorrect"
    assert rows[0].notes == "changed my mind"


async def test_post_invalid_outcome_returns_400(tmp_path):
    await _seed_run_with_step(tmp_path)

    resp = client.post("/runs/run-1/steps/triage/feedback", json={"outcome": "maybe"})

    assert resp.status_code == 400


async def test_post_unknown_step_returns_404(tmp_path):
    await _seed_run_with_step(tmp_path)

    resp = client.post("/runs/run-1/steps/nonexistent/feedback", json={"outcome": "correct"})

    assert resp.status_code == 404


async def test_get_unknown_step_returns_null_feedback(tmp_path):
    await _seed_run_with_step(tmp_path)

    resp = client.get("/runs/run-1/steps/nonexistent/feedback")

    assert resp.status_code == 200
    assert resp.json() == {"feedback": None}


async def test_fanout_branch_name_with_slash(tmp_path):
    await _seed_run_with_step(tmp_path, step_name="triage/0")

    resp = client.post("/runs/run-1/steps/triage/0/feedback", json={"outcome": "partial"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["step_name"] == "triage/0"
    assert body["outcome"] == "partial"

    get_resp = client.get("/runs/run-1/steps/triage/0/feedback")
    assert get_resp.status_code == 200
    assert get_resp.json()["feedback"]["outcome"] == "partial"


async def test_get_returns_stored_feedback_after_post(tmp_path):
    await _seed_run_with_step(tmp_path)
    client.post("/runs/run-1/steps/triage/feedback", json={"outcome": "correct", "notes": "n"})

    resp = client.get("/runs/run-1/steps/triage/feedback")

    assert resp.status_code == 200
    fb = resp.json()["feedback"]
    assert fb["run_id"] == "run-1"
    assert fb["step_name"] == "triage"
    assert fb["outcome"] == "correct"
    assert fb["notes"] == "n"


# ---------------------------------------------------------------------------
# Rollup — Steps Insights
# ---------------------------------------------------------------------------

async def test_steps_insights_rollup_includes_accuracy(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="run-prod", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
            triggered_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineStep(
            id="step-a", run_id="run-prod", step_name="triage", step_index=0,
            executor="gateway", agent="agent-x", model="claude-sonnet-5", provider="anthropic",
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineStep(
            id="step-b", run_id="run-prod", step_name="investigate", step_index=1,
            executor="gateway", agent="agent-x", model="claude-sonnet-5", provider="anthropic",
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
        ))
        session.add(StepFeedback(id="fb-a", step_id="step-a", run_id="run-prod", pipeline_name="p", step_name="triage", outcome="correct"))
        session.add(StepFeedback(id="fb-b", step_id="step-b", run_id="run-prod", pipeline_name="p", step_name="investigate", outcome="incorrect"))

        # Testing-stage run — must be excluded from the rollup.
        session.add(PipelineRun(
            id="run-test", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="testing",
            triggered_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineStep(
            id="step-c", run_id="run-test", step_name="triage", step_index=0,
            executor="gateway", agent="agent-x", model="claude-sonnet-5", provider="anthropic",
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
        ))
        session.add(StepFeedback(id="fb-c", step_id="step-c", run_id="run-test", pipeline_name="p", step_name="triage", outcome="incorrect"))
        await session.commit()

    resp = client.get("/ui/insights/steps?time_range=all")

    assert resp.status_code == 200
    body = resp.text
    # triage: only the production "correct" mark counts -> 100% (1)
    assert '"accuracy_pct": 100' in body or '"accuracy_pct":100' in body
    assert '"marked_total": 1' in body or '"marked_total":1' in body


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def test_collect_emits_pork_step_feedback_total():
    data = MetricsData(
        run_counts=[], runs_in_progress=0, step_counts=[], step_durations=[],
        verifier_counts=[], token_usage=[], human_decisions=[], feedback_counts=[],
        step_feedback_counts=[
            ("p", "triage", "agent-x", "claude-sonnet-5", "anthropic", "correct", 3),
            ("p", "triage", None, None, None, "incorrect", 1),
        ],
        grounding_scores=[],
    )
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_step_feedback_total")

    by_outcome = {s.labels["outcome"]: (s.value, s.labels) for s in family.samples}
    assert by_outcome["correct"][0] == 3
    assert by_outcome["correct"][1]["agent"] == "agent-x"
    assert by_outcome["incorrect"][0] == 1
    assert by_outcome["incorrect"][1]["agent"] == ""
    assert by_outcome["incorrect"][1]["model"] == ""
    assert by_outcome["incorrect"][1]["provider"] == ""


async def test_fetch_metrics_data_excludes_testing_stage_step_feedback(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="run-prod", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
        ))
        session.add(PipelineRun(
            id="run-test", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="testing",
        ))
        session.add(PipelineStep(
            id="step-prod", run_id="run-prod", step_name="triage", step_index=0,
            executor="gateway", agent="agent-x", model="m", provider="anthropic",
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineStep(
            id="step-test", run_id="run-test", step_name="triage", step_index=0,
            executor="gateway", agent="agent-x", model="m", provider="anthropic",
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
        ))
        session.add(StepFeedback(id="fb-prod", step_id="step-prod", run_id="run-prod", pipeline_name="p", step_name="triage", outcome="correct"))
        session.add(StepFeedback(id="fb-test", step_id="step-test", run_id="run-test", pipeline_name="p", step_name="triage", outcome="incorrect"))
        await session.commit()

    metrics_data = await fetch_metrics_data(sf)

    assert len(metrics_data.step_feedback_counts) == 1
    assert metrics_data.step_feedback_counts[0][:2] == ("p", "triage")
    assert metrics_data.step_feedback_counts[0][-2] == "correct"
