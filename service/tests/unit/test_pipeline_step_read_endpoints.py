"""Tests for the new read endpoints added for the VectorStep Service MCP
(SPEC-vectorstep-service-mcp.md §5a): GET /pipelines/{name}, GET /steps,
GET /steps/{name}, GET /agents; plus the new stage filter on GET /runs
(needed for the list_runs tool's stage? parameter)."""
from datetime import datetime

import httpx
import pytest

import src.main as main
from src.db.database import get_session_factory
from src.db.models import PipelineRun
from src.main import app
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _pipeline(name="p", **kw) -> PipelineConfig:
    return PipelineConfig(
        name=name,
        trigger=TriggerConfig(match={}),
        steps=[StepConfig(name="s", executor="openclaw")],
        **kw,
    )


# ---------------------------------------------------------------------------
# GET /pipelines
# ---------------------------------------------------------------------------

async def test_list_pipelines_includes_stage_and_tags(monkeypatch, client):
    monkeypatch.setattr(main, "_pipelines", [_pipeline(stage="production", tags=["triage"])])

    resp = await client.get("/pipelines")

    assert resp.status_code == 200
    body = resp.json()["pipelines"][0]
    assert body == {"name": "p", "description": "", "version": 1, "stage": "production", "tags": ["triage"]}


# ---------------------------------------------------------------------------
# GET /pipelines/{name}
# ---------------------------------------------------------------------------

async def test_get_pipeline_returns_config_and_raw_yaml(tmp_path, monkeypatch, client):
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text("name: p\ntrigger:\n  match: {}\nsteps:\n  - name: s\n    executor: openclaw\n")
    monkeypatch.setattr(main, "_pipelines", [_pipeline()])
    monkeypatch.setattr(main, "_pipeline_dir", str(tmp_path))

    resp = await client.get("/pipelines/p")

    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["name"] == "p"
    assert "name: p" in body["yaml"]


async def test_get_pipeline_unknown_returns_404(monkeypatch, client):
    monkeypatch.setattr(main, "_pipelines", [])

    resp = await client.get("/pipelines/nonexistent")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /steps, GET /steps/{name}
# ---------------------------------------------------------------------------

_RAW_STEP = {
    "name": "triage",
    "description": "Triage an alert",
    "tags": ["alerting"],
    "executor": "gateway",
    "executor_config": {"agent": "triage-agent"},
}


async def test_list_steps_summaries(monkeypatch, client):
    monkeypatch.setattr(main, "_step_library", {"triage": _RAW_STEP})

    resp = await client.get("/steps")

    assert resp.status_code == 200
    assert resp.json()["steps"] == [{
        "name": "triage", "description": "Triage an alert", "tags": ["alerting"],
        "executor": "gateway", "agent": "triage-agent",
    }]


async def test_get_step_returns_config_and_raw_yaml(tmp_path, monkeypatch, client):
    yaml_path = tmp_path / "triage.yaml"
    yaml_path.write_text("name: triage\nexecutor: gateway\nexecutor_config:\n  agent: triage-agent\n")
    monkeypatch.setattr(main, "_step_library", {"triage": _RAW_STEP})
    monkeypatch.setattr(main, "_step_library_dir", str(tmp_path))

    resp = await client.get("/steps/triage")

    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["name"] == "triage"
    assert "name: triage" in body["yaml"]


async def test_get_step_unknown_returns_404(monkeypatch, client):
    monkeypatch.setattr(main, "_step_library", {})

    resp = await client.get("/steps/nonexistent")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /agents
# ---------------------------------------------------------------------------

async def test_list_agents_merges_backends_and_fails_soft(monkeypatch, client):
    async def fake_openclaw():
        return [{"id": "oc-1", "model": "claude-sonnet-5"}], None

    async def fake_gateway():
        return [], "Could not reach VectorStep Gateway at http://x — is it running?"

    monkeypatch.setattr(main, "_fetch_openclaw_agents", fake_openclaw)
    monkeypatch.setattr(main, "_fetch_vectorstep_gateway_agents", fake_gateway)

    resp = await client.get("/agents")

    assert resp.status_code == 200
    body = resp.json()
    assert body["agents"] == [{"name": "oc-1", "executor": "openclaw", "model": "claude-sonnet-5"}]
    assert "gateway" in body["errors"]
    assert "openclaw" not in body["errors"]


# ---------------------------------------------------------------------------
# GET /runs?stage=
# ---------------------------------------------------------------------------

async def test_list_runs_filters_by_stage(db, client):
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="run-prod", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
            triggered_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineRun(
            id="run-test", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="testing",
            triggered_at=datetime(2026, 1, 1),
        ))
        await session.commit()

    resp = await client.get("/runs?stage=testing")

    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()["runs"]]
    assert ids == ["run-test"]


async def test_list_runs_rejects_invalid_stage(db, client):
    resp = await client.get("/runs?stage=bogus")

    assert resp.status_code == 400
