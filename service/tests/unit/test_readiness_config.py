"""Tests for the readiness: config schema (SPEC-readiness-criteria.md §4) and the
tier-level merge (§5) — pure Pydantic, no DB."""
import pytest
from pydantic import ValidationError

from src.models.pipeline import (
    FanOutConfig,
    LibraryStepConfig,
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
from src.readiness import resolve_step_readiness

# ---------------------------------------------------------------------------
# Attachment points
# ---------------------------------------------------------------------------

def test_readiness_parses_on_pipeline_config():
    p = PipelineConfig(
        name="p", trigger=TriggerConfig(),
        readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)),
        steps=[StepConfig(name="s", executor="gateway")],
    )
    assert p.readiness.operational.min_runs == 5


def test_readiness_parses_on_step_config():
    s = StepConfig(name="s", executor="gateway",
                    readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)))
    assert s.readiness.operational.min_runs == 5


def test_readiness_parses_on_parallel_group_inner():
    g = ParallelGroupInner(
        name="g", steps=[ParallelStepConfig(name="a", executor="gateway", prompt_template="x")],
        readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)),
    )
    assert g.readiness.operational.min_runs == 5


def test_readiness_parses_on_fan_out_config():
    f = FanOutConfig(name="f", over="items", executor="gateway",
                      readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)))
    assert f.readiness.operational.min_runs == 5


def test_readiness_parses_on_library_step_config():
    ls = LibraryStepConfig(name="ls", executor="gateway",
                            readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)))
    assert ls.readiness.operational.min_runs == 5


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------

def test_operational_requires_min_runs():
    with pytest.raises(ValidationError):
        ReadinessOperationalConfig()


def test_calibration_defaults_are_all_valid():
    cal = ReadinessCalibrationConfig()
    assert cal.n_min == 20
    assert cal.bin_width == 0.1
    assert cal.max_divergence == 0.15
    assert cal.require_own_evidence is False
    assert cal.require_current_config is True


def test_bin_width_must_evenly_divide_one():
    with pytest.raises(ValidationError, match="must evenly divide 1.0"):
        ReadinessCalibrationConfig(bin_width=0.3)


def test_bin_width_rejected_at_config_load_not_request_time():
    """0.3 must fail at Pydantic-parse time (config load), not survive into a
    bare assert deep inside the evaluation engine."""
    with pytest.raises(ValidationError):
        ReadinessConfig.model_validate({"calibration": {"bin_width": 0.3}})


def test_acceptable_statuses_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ReadinessOperationalConfig(min_runs=1, acceptable_statuses=["done"])


def test_min_accuracy_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ReadinessAccuracyConfig(min_accuracy=1.5, min_marked=1)


def test_calibration_require_current_config_false_rejected():
    with pytest.raises(ValidationError, match="cannot be false"):
        ReadinessCalibrationConfig(require_current_config=False)


# ---------------------------------------------------------------------------
# Merge matrix — resolve_step_readiness (§5)
# ---------------------------------------------------------------------------

def test_merge_pipeline_only():
    pipeline_level = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=20))
    effective, source = resolve_step_readiness(pipeline_level, None, False)
    assert effective.operational.min_runs == 20
    assert source == {"operational": "pipeline"}


def test_merge_step_only():
    step_level = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.9, min_marked=10))
    effective, source = resolve_step_readiness(None, step_level, True)
    assert effective.accuracy.min_accuracy == 0.9
    assert source == {"accuracy": "step"}


def test_merge_pipeline_and_step_different_tiers_union():
    pipeline_level = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=20))
    step_level = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.9, min_marked=10))
    effective, source = resolve_step_readiness(pipeline_level, step_level, True)
    assert effective.operational.min_runs == 20
    assert effective.accuracy.min_accuracy == 0.9
    assert source == {"operational": "pipeline", "accuracy": "step"}


def test_merge_same_tier_step_wins_whole_tier_not_field_merged():
    pipeline_level = ReadinessConfig(operational=ReadinessOperationalConfig(
        min_runs=20, max_age_days=30, acceptable_statuses=["completed", "escalated"],
    ))
    step_level = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5))
    effective, source = resolve_step_readiness(pipeline_level, step_level, True)
    # Step's operational block REPLACES the pipeline's wholesale — max_age_days
    # and acceptable_statuses come back to their OWN defaults, not the pipeline's.
    assert effective.operational.min_runs == 5
    assert effective.operational.max_age_days is None
    assert effective.operational.acceptable_statuses == ["completed"]
    assert source == {"operational": "step"}


def test_merge_explicit_null_tier_removes_inherited_tier():
    pipeline_level = ReadinessConfig(
        operational=ReadinessOperationalConfig(min_runs=20),
        confidence=ReadinessConfidenceConfig(min_confidence=0.8),
    )
    step_level = ReadinessConfig.model_validate({"calibration": None})
    # calibration wasn't inherited anyway (pipeline never set it) — but the KEY
    # point is confidence, NOT mentioned by the step, is still inherited normally.
    effective, source = resolve_step_readiness(pipeline_level, step_level, True)
    assert effective.operational.min_runs == 20
    assert effective.confidence.min_confidence == 0.8
    assert effective.calibration is None


def test_merge_explicit_null_removes_a_tier_the_pipeline_did_set():
    pipeline_level = ReadinessConfig(
        operational=ReadinessOperationalConfig(min_runs=20),
        calibration=ReadinessCalibrationConfig(),
    )
    step_level = ReadinessConfig.model_validate({"calibration": None})
    effective, source = resolve_step_readiness(pipeline_level, step_level, True)
    assert effective.operational.min_runs == 20
    assert effective.calibration is None
    assert "calibration" not in source


def test_merge_whole_block_null_opts_step_out_entirely():
    pipeline_level = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=20))
    effective, source = resolve_step_readiness(pipeline_level, None, True)
    assert effective is None
    assert source == {}


def test_merge_neither_configured_returns_none():
    effective, source = resolve_step_readiness(None, None, False)
    assert effective is None
    assert source == {}
