"""Tests for stage=testing/production trigger gating on _trigger_run (main.py)."""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

import src.main as main
from src.db.models import PipelineRun
from src.models.context import NormalisedContext
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig


def _pipeline(name="p", stage="testing"):
    return PipelineConfig(
        name=name,
        trigger=TriggerConfig(),
        steps=[StepConfig(name="s", executor="gateway")],
        stage=stage,
    )


def _normalised(pipeline="p"):
    return NormalisedContext(
        source="generic",
        pipeline=pipeline,
        summary="test",
        severity=None,
        received_at=datetime(2026, 1, 1, 12, 0, 0),
        labels={},
        raw={},
        fingerprint=None,  # skip dedup path entirely
    )


@pytest.fixture(autouse=True)
def _stub_run_pipeline(monkeypatch):
    """Prevent _trigger_run's background task from touching a real runner/DB."""
    monkeypatch.setattr(main, "_run_pipeline", AsyncMock())
    monkeypatch.setattr(main, "_active_runs", 0)
    monkeypatch.setattr(main, "_max_concurrent_runs", 10)
    yield


async def test_testing_pipeline_skipped_without_allow_testing(monkeypatch):
    monkeypatch.setattr(main, "_pipelines", [_pipeline(stage="testing")])

    resp = await main._trigger_run(_normalised())

    assert resp.status_code == 202
    import json
    body = json.loads(resp.body)
    assert body["status"] == "skipped_testing"
    assert body["pipeline"] == "p"


async def test_testing_pipeline_accepted_with_allow_testing(monkeypatch):
    monkeypatch.setattr(main, "_pipelines", [_pipeline(stage="testing")])

    resp = await main._trigger_run(_normalised(), allow_testing=True)
    await asyncio.sleep(0)  # let the stubbed background task settle

    assert resp.status_code == 202
    import json
    body = json.loads(resp.body)
    assert body["status"] == "accepted"


async def test_production_pipeline_unaffected_by_allow_testing_flag(monkeypatch):
    monkeypatch.setattr(main, "_pipelines", [_pipeline(stage="production")])

    resp = await main._trigger_run(_normalised())
    await asyncio.sleep(0)

    import json
    body = json.loads(resp.body)
    assert body["status"] == "accepted"


async def test_run_pipeline_now_always_allows_testing_pipeline(monkeypatch):
    """POST /pipelines/{name}/run (run_pipeline_now) passes allow_testing=True."""
    from fastapi import Request

    monkeypatch.setattr(main, "_pipelines", [_pipeline(stage="testing")])

    class _FakeRequest:
        async def json(self):
            return {}

    resp = await main.run_pipeline_now("p", _FakeRequest())
    await asyncio.sleep(0)

    import json
    body = json.loads(resp.body)
    assert body["status"] == "accepted"


# ---------------------------------------------------------------------------
# stage on the GET /runs and GET /runs/{run_id} JSON response bodies
# ---------------------------------------------------------------------------

def _run(stage="testing", **overrides):
    return PipelineRun(
        id="run-1", pipeline_name="p", source="generic", status="completed",
        normalised_context="{}", raw_payload="{}",
        triggered_at=datetime(2026, 1, 1, 12, 0, 0), stage=stage,
        **overrides,
    )


def test_format_run_summary_includes_stage_testing():
    assert main._format_run_summary(_run(stage="testing"))["stage"] == "testing"


def test_format_run_summary_includes_stage_production():
    assert main._format_run_summary(_run(stage="production"))["stage"] == "production"


def test_format_run_detail_includes_stage():
    run = _run(stage="testing")
    run.steps = []
    detail = main._format_run_detail(run)
    assert detail["stage"] == "testing"
