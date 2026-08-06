"""Tests for owner-defined promotion readiness (SPEC-readiness-criteria.md):
GET /pipelines/{name}/promotion-readiness, POST .../preview, and the
pipeline-detail UI wiring."""
import yaml
import httpx
import pytest
from fastapi import FastAPI

import src.main as main
import src.ui.pipelines as ui_pipelines_module
from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, StepFeedback
from src.main import app as main_app
from src.models.pipeline import (
    PipelineConfig, ReadinessAccuracyConfig, ReadinessConfig,
    ReadinessOperationalConfig, StepConfig, TriggerConfig,
)
from src.ui import router as ui_router

ui_app = FastAPI()
ui_app.include_router(ui_router)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=main_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def ui_client():
    transport = httpx.ASGITransport(app=ui_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_readiness_cache():
    main._readiness_evidence_cache.clear()
    yield
    main._readiness_evidence_cache.clear()


def _pipeline(name: str, stage: str = "testing", step_name: str = "investigate",
              readiness: ReadinessConfig | None = None) -> PipelineConfig:
    return PipelineConfig(
        name=name, stage=stage, trigger=TriggerConfig(), readiness=readiness,
        steps=[StepConfig(name=step_name, executor="gateway", prompt_template="hi",
                           confidence_threshold=0.8)],
    )


async def _seed_run(sf, run_id: str, pipeline_name: str, stage: str) -> None:
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name=pipeline_name, source="test",
            normalised_context="{}", raw_payload="{}", stage=stage,
        ))
        await session.commit()


async def _seed_step(
    sf, run_id: str, pipeline_name: str, step_name: str, *, agent="a", model="m", provider="pr",
    prompt_hash=None, agent_version="v1", effective_confidence=0.9, step_feedback=None, index=0,
) -> str:
    async with sf() as session:
        step = PipelineStep(
            run_id=run_id, step_name=step_name, step_index=index, executor="gateway",
            agent=agent, model=model, provider=provider, prompt="p", status="completed",
            prompt_hash=prompt_hash, agent_version=agent_version,
            effective_confidence=effective_confidence,
        )
        session.add(step)
        await session.flush()
        step_id = step.id
        if step_feedback is not None:
            session.add(StepFeedback(
                step_id=step_id, run_id=run_id, pipeline_name=pipeline_name,
                step_name=step_name, outcome=step_feedback,
            ))
        await session.commit()
    return step_id


async def _seed_n(sf, pipeline_name: str, stage: str, step_name: str, n: int, outcome: str) -> None:
    for i in range(n):
        run_id = f"{pipeline_name}-{step_name}-{i}"
        await _seed_run(sf, run_id, pipeline_name, stage)
        await _seed_step(sf, run_id, pipeline_name, step_name, step_feedback=outcome)


# ---------------------------------------------------------------------------
# GET endpoint
# ---------------------------------------------------------------------------

async def test_endpoint_404_for_unknown_pipeline(db, monkeypatch, client):
    monkeypatch.setattr(main, "_pipelines", [])
    resp = await client.get("/pipelines/nonexistent/promotion-readiness")
    assert resp.status_code == 404


async def test_endpoint_200_for_testing_and_production_pipeline(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_n(sf, "pipeline-b", "testing", "investigate", 5, "correct")
    pipeline_b = _pipeline("pipeline-b", stage="testing",
                            readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)))
    pipeline_prod = _pipeline("pipeline-prod", stage="production")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b, pipeline_prod])

    resp_testing = await client.get("/pipelines/pipeline-b/promotion-readiness")
    resp_prod = await client.get("/pipelines/pipeline-prod/promotion-readiness")

    assert resp_testing.status_code == 200
    body = resp_testing.json()
    assert body["pipeline_name"] == "pipeline-b"
    assert body["criteria_source"] == "configured"
    assert body["steps"][0]["tiers"]["operational"]["verdict"] == "pass"

    assert resp_prod.status_code == 200
    assert resp_prod.json()["criteria_source"] == "none"


async def test_no_readiness_block_returns_none_source_but_still_shows_observed_evidence(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_n(sf, "pipeline-b", "testing", "investigate", 3, "correct")
    pipeline_b = _pipeline("pipeline-b")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    resp = await client.get("/pipelines/pipeline-b/promotion-readiness")
    body = resp.json()
    assert body["criteria_source"] == "none"
    assert body["verdict"] == "not_configured"
    step = body["steps"][0]
    assert all(t["verdict"] == "not_configured" for t in step["tiers"].values())
    assert step["observed_combos"], "observed evidence must still be populated (§11)"
    assert step["observed_combos"][0]["observed"]["total_n"] == 3


async def test_bin_width_n_min_query_params_affect_only_unconfigured_steps(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_n(sf, "pipeline-b", "testing", "investigate", 3, "correct")
    pipeline_b = _pipeline("pipeline-b")   # no calibration tier configured at all
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    resp_default = await client.get("/pipelines/pipeline-b/promotion-readiness")
    resp_lowered = await client.get("/pipelines/pipeline-b/promotion-readiness?n_min=3")

    default_combo = resp_default.json()["steps"][0]["observed_combos"][0]["observed"]
    lowered_combo = resp_lowered.json()["steps"][0]["observed_combos"][0]["observed"]
    assert default_combo["validated"] is False   # n=3 < default n_min=20
    assert lowered_combo["validated"] is True     # n=3 >= n_min=3


# ---------------------------------------------------------------------------
# POST preview endpoint
# ---------------------------------------------------------------------------

async def test_preview_valid_body_returns_yaml_that_round_trips(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_n(sf, "pipeline-b", "testing", "investigate", 5, "correct")
    pipeline_b = _pipeline("pipeline-b")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    resp = await client.post(
        "/pipelines/pipeline-b/promotion-readiness/preview",
        json={"readiness": {"operational": {"min_runs": 3}}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["steps"][0]["tiers"]["operational"]["verdict"] == "pass"
    assert body["scope"] == "pipeline"
    assert body["applied_to"] is None

    parsed = yaml.safe_load(body["yaml_snippet"])
    validated = ReadinessConfig.model_validate(parsed["readiness"])
    assert validated.operational.min_runs == 3


async def test_preview_invalid_body_422(db, monkeypatch, client):
    pipeline_b = _pipeline("pipeline-b")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    resp = await client.post(
        "/pipelines/pipeline-b/promotion-readiness/preview",
        json={"readiness": {"calibration": {"bin_width": 0.3}}},
    )
    assert resp.status_code == 422


async def test_preview_apply_to_unknown_step_400(db, monkeypatch, client):
    pipeline_b = _pipeline("pipeline-b")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    resp = await client.post(
        "/pipelines/pipeline-b/promotion-readiness/preview",
        json={"readiness": {"operational": {"min_runs": 3}}, "apply_to": ["nope"]},
    )
    assert resp.status_code == 400


async def test_preview_apply_to_specific_step_scopes_yaml(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_n(sf, "pipeline-b", "testing", "investigate", 5, "correct")
    pipeline_b = _pipeline("pipeline-b")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    resp = await client.post(
        "/pipelines/pipeline-b/promotion-readiness/preview",
        json={"readiness": {"operational": {"min_runs": 3}}, "apply_to": ["investigate"]},
    )
    body = resp.json()
    assert body["scope"] == "step"
    assert body["applied_to"] == ["investigate"]
    assert "investigate" in body["yaml_target_hint"]


async def test_preview_writes_nothing(db, monkeypatch, client):
    pipeline_b = _pipeline("pipeline-b")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    calls = []
    monkeypatch.setattr(main, "write_pipeline_yaml", lambda *a, **kw: calls.append(1))

    resp = await client.post(
        "/pipelines/pipeline-b/promotion-readiness/preview",
        json={"readiness": {"operational": {"min_runs": 3}}},
    )
    assert resp.status_code == 200
    assert calls == []


async def test_preview_and_get_agree_when_override_equals_configured_criteria(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_n(sf, "pipeline-b", "testing", "investigate", 5, "correct")
    pipeline_b = _pipeline("pipeline-b",
                            readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=3)))
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    get_resp = await client.get("/pipelines/pipeline-b/promotion-readiness")
    preview_resp = await client.post(
        "/pipelines/pipeline-b/promotion-readiness/preview",
        json={"readiness": {"operational": {"min_runs": 3}}},
    )
    assert get_resp.json()["steps"][0]["tiers"] == preview_resp.json()["steps"][0]["tiers"]


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------

async def test_ui_pipeline_detail_skips_readiness_for_production(tmp_path, db, monkeypatch, ui_client):
    calls = []

    async def _stub_gather(*args, **kwargs):
        calls.append(1)
        raise AssertionError("should not be called for a production pipeline")

    monkeypatch.setattr(ui_pipelines_module, "_gather_readiness_evidence", _stub_gather)

    pipeline = _pipeline("p", stage="production")
    ui_app.state.pipelines = [pipeline]
    ui_app.state.pipeline_dir = str(tmp_path)

    resp = await ui_client.get("/ui/pipelines/p")

    assert resp.status_code == 200
    assert calls == []
    assert '<h2 class="text-sm font-semibold text-zinc-200">Promotion readiness</h2>' not in resp.text


async def test_ui_pipeline_detail_populates_readiness_for_testing(tmp_path, db, ui_client):
    sf = get_session_factory()
    await _seed_n(sf, "p", "testing", "investigate", 5, "correct")

    pipeline = _pipeline("p", stage="testing",
                          readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=3)))
    ui_app.state.pipelines = [pipeline]
    ui_app.state.pipeline_dir = str(tmp_path)

    resp = await ui_client.get("/ui/pipelines/p")

    assert resp.status_code == 200
    assert '<h2 class="text-sm font-semibold text-zinc-200">Promotion readiness</h2>' in resp.text
    assert "How is this judged?" in resp.text
    assert "runs 5/3" in resp.text


async def test_ui_pipeline_detail_no_criteria_shows_neutral_state(tmp_path, db, ui_client):
    sf = get_session_factory()
    await _seed_n(sf, "p", "testing", "investigate", 3, "correct")

    pipeline = _pipeline("p", stage="testing")
    ui_app.state.pipelines = [pipeline]
    ui_app.state.pipeline_dir = str(tmp_path)

    resp = await ui_client.get("/ui/pipelines/p")

    assert resp.status_code == 200
    assert "no criteria configured" in resp.text
    assert "No criteria configured" in resp.text
