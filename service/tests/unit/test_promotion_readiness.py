"""Tests for calibration-based promotion readiness (SPEC-calibration-readiness.md):
compute_calibration_buckets(stage=...), CalibrationCache staying production-only,
get_pipeline_promotion_readiness, the GET /pipelines/{name}/promotion-readiness
endpoint, and the pipeline-detail UI wiring."""
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

import src.main as main
import src.ui as ui_module
from src.analytics import get_pipeline_promotion_readiness
from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, RunFeedback, StepFeedback
from src.main import app as main_app
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig
from src.pipeline.calibration import CalibrationCache, compute_calibration_buckets
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


def _pipeline(name: str, stage: str = "testing", step_name: str = "investigate") -> PipelineConfig:
    return PipelineConfig(
        name=name, stage=stage, trigger=TriggerConfig(),
        steps=[StepConfig(name=step_name, executor="gateway", confidence_threshold=0.8)],
    )


async def _seed_run(sf, run_id: str, pipeline_name: str, stage: str) -> None:
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name=pipeline_name, source="test",
            normalised_context="{}", raw_payload="{}", stage=stage,
        ))
        await session.commit()


async def _seed_step(
    sf, run_id: str, pipeline_name: str, step_name: str, *, agent=None, model=None, provider=None,
    prompt_hash=None, agent_version=None, effective_confidence=0.9, step_feedback=None, index=0,
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


async def _seed_combo(
    sf, *, pipeline_name: str, stage: str, step_name: str, agent: str, model: str,
    provider: str, prompt_hash: str, agent_version: str, n: int, outcome: str | None,
    effective_confidence: float = 0.95, run_prefix: str = "r",
) -> None:
    """Seed n marked (or unmarked, if outcome is None) step executions for one
    (step, agent, model, provider, prompt_hash, agent_version) combo."""
    for i in range(n):
        run_id = f"{run_prefix}-{pipeline_name}-{step_name}-{i}"
        await _seed_run(sf, run_id, pipeline_name, stage)
        await _seed_step(
            sf, run_id, pipeline_name, step_name, agent=agent, model=model, provider=provider,
            prompt_hash=prompt_hash, agent_version=agent_version,
            effective_confidence=effective_confidence, step_feedback=outcome, index=0,
        )


# ---------------------------------------------------------------------------
# compute_calibration_buckets(stage=...)
# ---------------------------------------------------------------------------

async def test_default_stage_excludes_testing_rows(db):
    sf = get_session_factory()
    await _seed_run(sf, "run-1", "p", "testing")
    await _seed_step(sf, "run-1", "p", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="correct")

    buckets = await compute_calibration_buckets(sf)

    assert buckets == {}


async def test_testing_stage_excludes_production_rows_for_same_key(db):
    sf = get_session_factory()
    await _seed_run(sf, "run-prod", "p", "production")
    await _seed_step(sf, "run-prod", "p", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="correct")
    await _seed_run(sf, "run-test", "p", "testing")
    await _seed_step(sf, "run-test", "p", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="partial", index=0)

    testing_buckets = await compute_calibration_buckets(sf, stage="testing")

    key = ("s", "a", "m", "pr", None, None)
    assert testing_buckets[key].total_n == 1
    assert testing_buckets[key].bins[9].mean_label == 0.5  # only the testing (partial) row


async def test_calibration_cache_stays_production_only(db):
    sf = get_session_factory()
    await _seed_run(sf, "run-prod", "p", "production")
    await _seed_step(sf, "run-prod", "p", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="correct")
    await _seed_run(sf, "run-test", "p", "testing")
    await _seed_step(sf, "run-test", "p", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="incorrect")

    cache = CalibrationCache(sf)
    bucket = await cache.get("s", "a", "m", "pr", None, None)

    assert bucket is not None
    assert bucket.total_n == 1
    assert bucket.bins[9].mean_label == 1.0  # only the production (correct) row


# ---------------------------------------------------------------------------
# get_pipeline_promotion_readiness
# ---------------------------------------------------------------------------

async def test_ready_from_production_evidence_alone(db):
    sf = get_session_factory()
    # Pipeline A (production): 20 clean, validated runs for the shared combo.
    await _seed_combo(
        sf, pipeline_name="pipeline-a", stage="production", step_name="investigate",
        agent="sre-investigation", model="claude-sonnet-5", provider="anthropic",
        prompt_hash="ph1", agent_version="v1", n=20, outcome="correct",
        effective_confidence=0.95,
    )
    # Pipeline B (testing): one unmarked execution of the identical combo, just to
    # make it an "observed" combo for pipeline B — no labelled testing history.
    await _seed_run(sf, "run-b-observe", "pipeline-b", "testing")
    await _seed_step(sf, "run-b-observe", "pipeline-b", "investigate",
                      agent="sre-investigation", model="claude-sonnet-5", provider="anthropic",
                      prompt_hash="ph1", agent_version="v1", effective_confidence=0.95)

    pipeline_b = _pipeline("pipeline-b")
    result = await get_pipeline_promotion_readiness(sf, pipeline_b)

    assert len(result["combos"]) == 1
    combo = result["combos"][0]
    assert combo["status"] == "ready"
    assert combo["testing"] is None
    assert combo["production"]["validated"] is True
    assert result["summary"] == {"ready": 1, "flagged": 0, "building": 0, "no_data": 0, "total": 1}


async def test_ready_from_testing_evidence_alone(db):
    sf = get_session_factory()
    await _seed_combo(
        sf, pipeline_name="pipeline-b", stage="testing", step_name="investigate",
        agent="brand-new-agent", model="brand-new-model", provider="openrouter",
        prompt_hash="ph2", agent_version="v2", n=20, outcome="correct",
        effective_confidence=0.95,
    )

    pipeline_b = _pipeline("pipeline-b")
    result = await get_pipeline_promotion_readiness(sf, pipeline_b)

    assert len(result["combos"]) == 1
    combo = result["combos"][0]
    assert combo["status"] == "ready"
    assert combo["production"] is None
    assert combo["testing"]["validated"] is True


async def test_flagged_when_validated_bucket_diverges(db):
    sf = get_session_factory()
    # High predicted confidence, but only half the marked outcomes are correct —
    # mean_label ~0.5 vs. midpoint ~0.95, a >=15pt divergence.
    for i in range(10):
        run_id = f"r-flag-{i}"
        await _seed_run(sf, run_id, "pipeline-b", "testing")
        await _seed_step(sf, run_id, "pipeline-b", "investigate",
                          agent="a", model="m", provider="pr", prompt_hash="ph", agent_version="v",
                          effective_confidence=0.95, step_feedback="correct" if i % 2 == 0 else "incorrect")

    pipeline_b = _pipeline("pipeline-b")
    result = await get_pipeline_promotion_readiness(sf, pipeline_b, n_min=10)

    assert len(result["combos"]) == 1
    combo = result["combos"][0]
    assert combo["status"] == "flagged"
    assert combo["testing"]["recommendation"] is not None


async def test_building_when_below_n_min_and_no_production(db):
    sf = get_session_factory()
    await _seed_combo(
        sf, pipeline_name="pipeline-b", stage="testing", step_name="investigate",
        agent="a", model="m", provider="pr", prompt_hash="ph", agent_version="v",
        n=5, outcome="correct", effective_confidence=0.9,
    )

    pipeline_b = _pipeline("pipeline-b")
    result = await get_pipeline_promotion_readiness(sf, pipeline_b)  # default n_min=20

    assert len(result["combos"]) == 1
    assert result["combos"][0]["status"] == "building"


async def test_unexercised_combo_not_reported(db):
    sf = get_session_factory()
    await _seed_combo(
        sf, pipeline_name="pipeline-b", stage="testing", step_name="investigate",
        agent="a", model="m", provider="pr", prompt_hash="ph", agent_version="v",
        n=5, outcome="correct",
    )

    pipeline_b = _pipeline("pipeline-b")
    result = await get_pipeline_promotion_readiness(sf, pipeline_b)

    keys = {(c["agent"], c["model"]) for c in result["combos"]}
    assert ("never-used-agent", "never-used-model") not in keys


async def test_pipeline_with_no_testing_runs_returns_empty(db):
    sf = get_session_factory()
    pipeline_b = _pipeline("pipeline-b")

    result = await get_pipeline_promotion_readiness(sf, pipeline_b)

    assert result["combos"] == []
    assert result["summary"] == {"ready": 0, "flagged": 0, "building": 0, "no_data": 0, "total": 0}


async def test_flagged_outranks_ready_when_production_diverges(db):
    sf = get_session_factory()
    # Production bucket: validated but diverging (flagged).
    for i in range(10):
        run_id = f"r-prod-{i}"
        await _seed_run(sf, run_id, "pipeline-a", "production")
        await _seed_step(sf, run_id, "pipeline-a", "investigate",
                          agent="a", model="m", provider="pr", prompt_hash="ph", agent_version="v",
                          effective_confidence=0.95, step_feedback="correct" if i % 2 == 0 else "incorrect")
    # Testing bucket for the same combo: validated and clean — would look "ready" alone.
    for i in range(10):
        run_id = f"r-test-{i}"
        await _seed_run(sf, run_id, "pipeline-b", "testing")
        await _seed_step(sf, run_id, "pipeline-b", "investigate",
                          agent="a", model="m", provider="pr", prompt_hash="ph", agent_version="v",
                          effective_confidence=0.95, step_feedback="correct")

    pipeline_b = _pipeline("pipeline-b")
    result = await get_pipeline_promotion_readiness(sf, pipeline_b, n_min=10)

    assert len(result["combos"]) == 1
    combo = result["combos"][0]
    assert combo["status"] == "flagged"
    assert combo["testing"]["validated"] is True
    assert combo["testing"]["recommendation"] is None
    assert combo["production"]["recommendation"] is not None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

async def test_endpoint_404_for_unknown_pipeline(db, monkeypatch, client):
    monkeypatch.setattr(main, "_pipelines", [])

    resp = await client.get("/pipelines/nonexistent/promotion-readiness")

    assert resp.status_code == 404


async def test_endpoint_200_for_testing_and_production_pipeline(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_combo(
        sf, pipeline_name="pipeline-b", stage="testing", step_name="investigate",
        agent="a", model="m", provider="pr", prompt_hash="ph", agent_version="v",
        n=20, outcome="correct", effective_confidence=0.95,
    )
    pipeline_b = _pipeline("pipeline-b", stage="testing")
    pipeline_prod = _pipeline("pipeline-prod", stage="production")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b, pipeline_prod])

    resp_testing = await client.get("/pipelines/pipeline-b/promotion-readiness")
    resp_prod = await client.get("/pipelines/pipeline-prod/promotion-readiness")

    assert resp_testing.status_code == 200
    body = resp_testing.json()
    assert body["pipeline_name"] == "pipeline-b"
    assert body["combos"][0]["status"] == "ready"

    assert resp_prod.status_code == 200
    assert resp_prod.json()["combos"] == []


async def test_endpoint_n_min_query_param_is_honoured(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_combo(
        sf, pipeline_name="pipeline-b", stage="testing", step_name="investigate",
        agent="a", model="m", provider="pr", prompt_hash="ph", agent_version="v",
        n=3, outcome="correct", effective_confidence=0.95,
    )
    pipeline_b = _pipeline("pipeline-b")
    monkeypatch.setattr(main, "_pipelines", [pipeline_b])

    resp_default = await client.get("/pipelines/pipeline-b/promotion-readiness")
    resp_lowered = await client.get("/pipelines/pipeline-b/promotion-readiness?n_min=3")

    assert resp_default.json()["combos"][0]["status"] == "building"
    assert resp_lowered.json()["combos"][0]["status"] == "ready"


# ---------------------------------------------------------------------------
# UI wiring — ui_pipeline_detail
# ---------------------------------------------------------------------------

async def test_ui_pipeline_detail_skips_readiness_for_production(tmp_path, db, monkeypatch, ui_client):
    calls = []

    async def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return {"pipeline_name": "p", "combos": [], "summary": {}}

    monkeypatch.setattr(ui_module, "_get_pipeline_promotion_readiness", _stub)

    pipeline = _pipeline("p", stage="production")
    ui_app.state.pipelines = [pipeline]
    ui_app.state.pipeline_dir = str(tmp_path)

    resp = await ui_client.get("/ui/pipelines/p")

    assert resp.status_code == 200
    assert calls == []
    # The HTML comment above the {% if %} always renders; the visible heading must not.
    assert '<h2 class="text-sm font-semibold text-zinc-200">Promotion readiness</h2>' not in resp.text


async def test_ui_pipeline_detail_populates_readiness_for_testing(tmp_path, db, monkeypatch, ui_client):
    calls = []

    async def _stub(sf, pipeline, **kwargs):
        calls.append(pipeline.name)
        return {
            "pipeline_name": pipeline.name,
            "combos": [{
                "step_name": "investigate", "agent": "a", "model": "m", "provider": "pr",
                "prompt_hash": "ph", "agent_version": "v", "confidence_threshold": 0.8,
                "status": "ready",
                "production": {"total_n": 20, "validated": True, "recommendation": None},
                "testing": None,
            }],
            "summary": {"ready": 1, "flagged": 0, "building": 0, "no_data": 0, "total": 1},
        }

    monkeypatch.setattr(ui_module, "_get_pipeline_promotion_readiness", _stub)

    pipeline = _pipeline("p", stage="testing")
    ui_app.state.pipelines = [pipeline]
    ui_app.state.pipeline_dir = str(tmp_path)

    resp = await ui_client.get("/ui/pipelines/p")

    assert resp.status_code == 200
    assert calls == ["p"]
    assert '<h2 class="text-sm font-semibold text-zinc-200">Promotion readiness</h2>' in resp.text
