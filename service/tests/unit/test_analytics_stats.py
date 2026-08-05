"""Tests for service/src/analytics.py and the JSON /stats endpoints it powers
(SPEC-vectorstep-service-mcp.md §4/§5b/§6/§9).

Covers: operational rollups (run/status counts, tokens, duration avg/p95),
judged accuracy (RunFeedback/StepFeedback), production-vs-testing/all stage
scoping, time_range filtering, and a parity check proving the JSON endpoint
and the /ui/insights/pipelines page derive from the exact same numbers for a
seeded dataset (the drift guard called for in §9)."""
from datetime import datetime, timedelta

import httpx
import pytest

import src.main as main
from src.analytics import get_pipeline_stats, get_step_stats, list_pipeline_stats
from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, RunFeedback, StepFeedback
from src.main import app
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig
from src.utils import utc_now


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _pipeline(name="p") -> PipelineConfig:
    return PipelineConfig(name=name, trigger=TriggerConfig(match={}), steps=[StepConfig(name="s", executor="openclaw")])


async def _seed(now: datetime):
    sf = get_session_factory()
    async with sf() as session:
        # Two production runs for pipeline "p": one completed (fast), one failed (slow).
        session.add(PipelineRun(
            id="run-1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="payments", stage="production",
            triggered_at=now - timedelta(seconds=10), completed_at=now,
        ))
        session.add(PipelineRun(
            id="run-2", pipeline_name="p", source="generic", status="failed",
            normalised_context="{}", raw_payload="{}", team="payments", stage="production",
            triggered_at=now - timedelta(seconds=100), completed_at=now,
        ))
        # A still-running production run — must not count as terminal.
        session.add(PipelineRun(
            id="run-3", pipeline_name="p", source="generic", status="running",
            normalised_context="{}", raw_payload="{}", team="sre", stage="production",
            triggered_at=now,
        ))
        # A testing-stage run for the same pipeline — must be excluded from
        # stage="production" (the default) and only counted under stage="all"/"testing".
        session.add(PipelineRun(
            id="run-4", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="testing",
            triggered_at=now,
        ))
        session.add(PipelineStep(
            id="step-1", run_id="run-1", step_name="triage", step_index=0, executor="gateway",
            agent="agent-x", model="m", provider="anthropic", prompt="", status="completed",
            executed_at=now, duration_ms=1000, input_tokens=100, output_tokens=50,
        ))
        session.add(PipelineStep(
            id="step-2", run_id="run-2", step_name="triage", step_index=0, executor="gateway",
            agent="agent-x", model="m", provider="anthropic", prompt="", status="failed",
            executed_at=now, duration_ms=3000, input_tokens=200, output_tokens=75,
        ))
        await session.flush()
        session.add(RunFeedback(id="fb-1", run_id="run-1", pipeline_name="p", outcome="correct"))
        session.add(RunFeedback(id="fb-2", run_id="run-2", pipeline_name="p", outcome="incorrect"))
        session.add(StepFeedback(id="sfb-1", step_id="step-1", run_id="run-1", pipeline_name="p", step_name="triage", outcome="correct"))
        await session.commit()


# ---------------------------------------------------------------------------
# analytics.get_pipeline_stats
# ---------------------------------------------------------------------------

async def test_get_pipeline_stats_operational_and_judged_numbers(db):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed(now)

    stats = await get_pipeline_stats(get_session_factory(), "p", time_range="all", stage="production")

    assert stats["runs_total"] == 3  # run-1, run-2, run-3 — production only
    assert stats["status_counts"]["completed"] == 1
    assert stats["status_counts"]["failed"] == 1
    assert stats["status_counts"]["running"] == 1
    # success_rate = completed / terminal (excludes the still-running run)
    assert stats["success_rate"] == pytest.approx(0.5)
    assert stats["tokens"] == {"input": 300, "output": 125, "total": 425}
    assert stats["duration_seconds"]["avg"] == pytest.approx(55.0)  # (10 + 100) / 2
    assert stats["accuracy"] == {"correct": 1, "partial": 0, "incorrect": 1, "total": 2, "correct_pct": 50.0}
    assert stats["teams"] == ["payments", "sre"]


async def test_get_pipeline_stats_stage_all_includes_testing_run(db):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed(now)

    stats = await get_pipeline_stats(get_session_factory(), "p", time_range="all", stage="all")

    assert stats["runs_total"] == 4


async def test_get_pipeline_stats_unknown_pipeline_returns_zeroed_payload(db):
    stats = await get_pipeline_stats(get_session_factory(), "does-not-exist", time_range="all")

    assert stats["runs_total"] == 0
    assert stats["success_rate"] is None
    assert stats["accuracy"]["correct_pct"] is None


async def test_get_pipeline_stats_time_range_excludes_old_runs(db):
    recent = utc_now()
    old = recent - timedelta(days=30)
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="run-recent", pipeline_name="q", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
            triggered_at=recent, completed_at=recent,
        ))
        session.add(PipelineRun(
            id="run-old", pipeline_name="q", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
            triggered_at=old, completed_at=old,
        ))
        await session.commit()

    stats_7d = await get_pipeline_stats(sf, "q", time_range="7d", stage="production")
    stats_all = await get_pipeline_stats(sf, "q", time_range="all", stage="production")

    assert stats_7d["runs_total"] == 1
    assert stats_all["runs_total"] == 2


# ---------------------------------------------------------------------------
# analytics.list_pipeline_stats
# ---------------------------------------------------------------------------

async def test_list_pipeline_stats_only_includes_pipelines_with_runs(db):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed(now)

    rows = await list_pipeline_stats(get_session_factory(), time_range="all", stage="production")

    assert [r["pipeline_name"] for r in rows] == ["p"]


# ---------------------------------------------------------------------------
# analytics.get_step_stats
# ---------------------------------------------------------------------------

async def test_get_step_stats_operational_and_judged_numbers(db):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed(now)

    stats = await get_step_stats(get_session_factory(), "triage", time_range="all", stage="production")

    assert stats["runs_total"] == 2
    assert stats["status_counts"]["completed"] == 1
    assert stats["status_counts"]["failed"] == 1
    assert stats["success_rate"] == pytest.approx(0.5)
    assert stats["tokens"] == {"input": 300, "output": 125, "total": 425}
    assert stats["duration_seconds"]["avg"] == pytest.approx(2.0)  # (1000 + 3000) ms / 2 / 1000
    assert stats["accuracy"] == {"correct": 1, "partial": 0, "incorrect": 0, "total": 1, "correct_pct": 100.0}


async def test_get_step_stats_unknown_step_returns_zeroed_payload(db):
    stats = await get_step_stats(get_session_factory(), "does-not-exist", time_range="all")

    assert stats["runs_total"] == 0
    assert stats["accuracy"]["total"] == 0


# ---------------------------------------------------------------------------
# JSON endpoints
# ---------------------------------------------------------------------------

async def test_pipeline_stats_endpoint_404_for_unknown_pipeline(db, monkeypatch, client):
    monkeypatch.setattr(main, "_pipelines", [])

    resp = await client.get("/pipelines/nonexistent/stats")

    assert resp.status_code == 404


async def test_pipeline_stats_endpoint_matches_analytics_module(db, monkeypatch, client):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed(now)
    monkeypatch.setattr(main, "_pipelines", [_pipeline("p")])

    resp = await client.get("/pipelines/p/stats?time_range=all&stage=production")

    assert resp.status_code == 200
    expected = await get_pipeline_stats(get_session_factory(), "p", time_range="all", stage="production")
    assert resp.json() == expected


async def test_stats_pipelines_endpoint_lists_all(db, monkeypatch, client):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed(now)
    monkeypatch.setattr(main, "_pipelines", [_pipeline("p")])

    resp = await client.get("/stats/pipelines?time_range=all&stage=production")

    assert resp.status_code == 200
    assert [r["pipeline_name"] for r in resp.json()["pipelines"]] == ["p"]


async def test_step_stats_endpoint_404_for_unknown_step(db, monkeypatch, client):
    monkeypatch.setattr(main, "_step_library", {})

    resp = await client.get("/steps/nonexistent/stats")

    assert resp.status_code == 404


async def test_stats_endpoint_rejects_invalid_stage(db, monkeypatch, client):
    monkeypatch.setattr(main, "_pipelines", [_pipeline("p")])

    resp = await client.get("/pipelines/p/stats?stage=bogus")

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Parity — /ui/insights/pipelines must show the same numbers as the JSON
# endpoint for the same seeded data (§9's drift guard).
# ---------------------------------------------------------------------------

async def test_insights_pipelines_page_matches_stats_endpoint(db, monkeypatch, client):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed(now)
    monkeypatch.setattr(main, "_pipelines", [_pipeline("p")])

    stats_resp = await client.get("/pipelines/p/stats?time_range=all&stage=production")
    page_resp = await client.get("/ui/insights/pipelines?time_range=all")

    assert stats_resp.status_code == 200
    assert page_resp.status_code == 200
    stats = stats_resp.json()
    page = page_resp.text

    # The page's drilldown JSON embeds run_count/failed_count/input+output tokens
    # per pipeline — assert they match the JSON endpoint's numbers exactly.
    assert f'"run_count": {stats["runs_total"]}' in page or f'"run_count":{stats["runs_total"]}' in page
    assert f'"failed_count": {stats["status_counts"]["failed"]}' in page or f'"failed_count":{stats["status_counts"]["failed"]}' in page
    assert f'"input_tokens": {stats["tokens"]["input"]}' in page or f'"input_tokens":{stats["tokens"]["input"]}' in page
    assert f'"output_tokens": {stats["tokens"]["output"]}' in page or f'"output_tokens":{stats["tokens"]["output"]}' in page
