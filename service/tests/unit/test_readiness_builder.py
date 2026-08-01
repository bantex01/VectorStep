"""Tests for the readiness criteria builder UI (SPEC-readiness-builder.md):
READINESS_KNOB_HELP completeness, builder_seed, and the pipeline-detail card."""
import httpx
import pytest
from fastapi import FastAPI

from src.models.pipeline import (
    FanOutConfig,
    FanOutGroupConfig,
    ParallelGroupConfig,
    ParallelGroupInner,
    ParallelStepConfig,
    PipelineConfig,
    ReadinessAccuracyConfig,
    ReadinessCalibrationConfig,
    ReadinessConfidenceConfig,
    ReadinessConfig,
    ReadinessOperationalConfig,
    StepConfig,
    TriggerConfig,
)
from src.readiness import READINESS_KNOB_HELP, builder_seed, step_specs
from src.ui import router as ui_router

app = FastAPI()
app.include_router(ui_router)

_TIER_MODELS = {
    "operational": ReadinessOperationalConfig,
    "confidence": ReadinessConfidenceConfig,
    "accuracy": ReadinessAccuracyConfig,
    "calibration": ReadinessCalibrationConfig,
}


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _pipeline(stage: str = "testing", readiness: ReadinessConfig | None = None,
              step_readiness: ReadinessConfig | None = None) -> PipelineConfig:
    return PipelineConfig(
        name="p", stage=stage, trigger=TriggerConfig(), readiness=readiness,
        steps=[StepConfig(name="investigate", executor="gateway", prompt_template="hi",
                           confidence_threshold=0.8, readiness=step_readiness)],
    )


# ---------------------------------------------------------------------------
# READINESS_KNOB_HELP
# ---------------------------------------------------------------------------

def test_knob_help_keys_exactly_match_tier_model_fields():
    expected = {
        f"{tier}.{field}"
        for tier, model in _TIER_MODELS.items()
        for field in model.model_fields
    }
    assert set(READINESS_KNOB_HELP.keys()) == expected


# ---------------------------------------------------------------------------
# builder_seed
# ---------------------------------------------------------------------------

def test_builder_seed_unconfigured_pipeline_has_every_tier_disabled_at_defaults():
    pipeline = _pipeline()
    seed = builder_seed(pipeline)

    for scope_key in ("__pipeline__", "investigate"):
        for tier, model in _TIER_MODELS.items():
            state = seed[scope_key][tier]
            assert state["enabled"] is False
            for name, info in model.model_fields.items():
                expected = None if info.is_required() else info.get_default(call_default_factory=True)
                assert state[name] == expected, (scope_key, tier, name)


def test_builder_seed_configured_pipeline_reflects_live_values_not_defaults():
    pipeline_level = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=42))
    step_level = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.75, min_marked=12))
    pipeline = _pipeline(readiness=pipeline_level, step_readiness=step_level)
    seed = builder_seed(pipeline)

    # __pipeline__ scope reflects the pipeline-level block only.
    assert seed["__pipeline__"]["operational"] == {
        "enabled": True, "min_runs": 42, "acceptable_statuses": ["completed"],
        "max_age_days": None, "require_current_config": False,
    }
    assert seed["__pipeline__"]["accuracy"]["enabled"] is False

    # step scope reflects the step-level override (accuracy) merged with the
    # inherited pipeline-level tier (operational) — the live effective criteria.
    assert seed["investigate"]["operational"]["enabled"] is True
    assert seed["investigate"]["operational"]["min_runs"] == 42
    assert seed["investigate"]["accuracy"] == {
        "enabled": True, "min_accuracy": 0.75, "min_marked": 12,
        "min_human_marked": None, "require_current_config": True,
    }


def test_builder_seed_includes_every_step_including_parallel_and_fan_out_groups():
    parallel = ParallelGroupConfig(parallel=ParallelGroupInner(
        name="g",
        steps=[
            ParallelStepConfig(name="a", executor="gateway", prompt_template="x"),
            ParallelStepConfig(name="b", executor="gateway", prompt_template="y"),
        ],
    ))
    fan_out = FanOutGroupConfig(fan_out=FanOutConfig(name="fo", executor="gateway", over="{{ items }}"))
    plain = StepConfig(name="s", executor="gateway", prompt_template="hi", confidence_threshold=0.8)
    pipeline = PipelineConfig(
        name="p", stage="testing", trigger=TriggerConfig(), steps=[plain, parallel, fan_out],
    )

    seed = builder_seed(pipeline)

    expected_keys = {"__pipeline__"} | {s.name for s in step_specs(pipeline)}
    assert expected_keys == {"__pipeline__", "s", "g", "fo"}
    assert set(seed.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Pipeline detail page — card rendering
# ---------------------------------------------------------------------------

async def test_builder_card_renders_on_testing_pipeline(tmp_path, db, client):
    pipeline = _pipeline()
    app.state.pipelines = [pipeline]
    app.state.pipeline_dir = str(tmp_path)

    resp = await client.get("/ui/pipelines/p")
    assert resp.status_code == 200

    assert "Criteria builder" in resp.text
    assert "readinessBuilder(" in resp.text
    assert '<option value="__pipeline__">' in resp.text
    assert '<option value="investigate">' in resp.text
    # Exactly one step -> exactly one per-step scope option besides __pipeline__.
    assert resp.text.count("<option value=") == 2 + 5  # __pipeline__ + step + 5 acceptable_statuses options


async def test_builder_card_absent_on_production_pipeline(tmp_path, db, client):
    pipeline = _pipeline(stage="production")
    app.state.pipelines = [pipeline]
    app.state.pipeline_dir = str(tmp_path)

    resp = await client.get("/ui/pipelines/p")
    assert resp.status_code == 200
    assert "<h2 class=\"text-sm font-semibold text-zinc-200\">Criteria builder</h2>" not in resp.text
    # The readinessBuilder() JS function definition ships on every page load (it's in
    # {% block scripts %}, unconditional); what must NOT appear is an x-data invocation of it.
    assert "x-data=\"readinessBuilder(" not in resp.text


async def test_builder_card_contains_help_text_for_every_knob(tmp_path, db, client):
    from markupsafe import escape

    pipeline = _pipeline()
    app.state.pipelines = [pipeline]
    app.state.pipeline_dir = str(tmp_path)

    resp = await client.get("/ui/pipelines/p")
    assert resp.status_code == 200
    for help_text in READINESS_KNOB_HELP.values():
        assert str(escape(help_text)) in resp.text


async def test_builder_card_states_nothing_is_saved(tmp_path, db, client):
    pipeline = _pipeline()
    app.state.pipelines = [pipeline]
    app.state.pipeline_dir = str(tmp_path)

    resp = await client.get("/ui/pipelines/p")
    assert resp.status_code == 200
    assert "nothing is saved" in resp.text


async def test_builder_card_references_no_write_endpoint(tmp_path, db, client):
    pipeline = _pipeline()
    app.state.pipelines = [pipeline]
    app.state.pipeline_dir = str(tmp_path)

    resp = await client.get("/ui/pipelines/p")
    assert resp.status_code == 200
    text = resp.text

    assert "'PUT'" not in text and '"PUT"' not in text
    assert "'DELETE'" not in text and '"DELETE"' not in text

    import re
    fetch_targets = re.findall(r"fetch\(\s*[`'\"]([^`'\"]*)", text)
    allowed_suffixes = ("/promotion-readiness/preview", "/run")
    for target in fetch_targets:
        assert target.endswith(allowed_suffixes), target
