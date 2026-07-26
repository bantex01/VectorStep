"""Tests for _step_config_summary — the plain-language "what is this step set up to
do" panel, complementing _confidence_narrative's "what happened this run"."""
from src.ui import _step_config_summary


def _trust(**overrides) -> dict:
    base = {
        "signals": {"S": 0.85, "V": None, "V_mode": None, "V_combination_strategy": None, "V_veto_floor": None},
        "gate": {"confidence_threshold": 0.75, "on_low_confidence": "escalate"},
        "grounding": None,
        "deterministic_checks": None,
        "calibration": None,
    }
    base.update(overrides)
    return base


def test_states_confidence_threshold_and_on_low_confidence():
    lines = _step_config_summary(_trust())

    assert any("75%" in line and "escalate" in line for line in lines)


def test_no_threshold_data_produces_no_threshold_line():
    lines = _step_config_summary(_trust(gate={}))

    assert not any("threshold" in line for line in lines)


def test_verifier_veto_strategy_names_the_floor():
    trust = _trust(signals={
        "S": 0.85, "V": 0.7, "V_mode": "independent",
        "V_combination_strategy": "veto", "V_veto_floor": 0.6,
    })
    lines = _step_config_summary(trust)

    assert any("independent" in line and "veto rule" in line and "60%" in line for line in lines)


def test_verifier_minimum_strategy_names_the_combination():
    trust = _trust(signals={
        "S": 0.85, "V": 0.7, "V_mode": "critic",
        "V_combination_strategy": "minimum", "V_veto_floor": None,
    })
    lines = _step_config_summary(trust)

    assert any("critic" in line and "minimum combination" in line for line in lines)


def test_no_verifier_produces_no_verifier_line():
    lines = _step_config_summary(_trust())

    assert not any("Verifier:" in line for line in lines)


def test_grounding_enforced_is_labelled_enforced():
    lines = _step_config_summary(_trust(grounding={"agent": "grounding-judge", "enforce": True}))

    assert any("enforced" in line and "grounding-judge" in line for line in lines)


def test_grounding_shadow_is_labelled_shadow():
    lines = _step_config_summary(_trust(grounding={"agent": "grounding-judge", "enforce": False}))

    assert any("shadow only" in line for line in lines)


def test_grounding_enforce_missing_says_not_recorded():
    lines = _step_config_summary(_trust(grounding={"agent": "grounding-judge"}))

    assert any("not recorded for this older run" in line for line in lines)


def test_deterministic_checks_listed_by_name_and_type():
    lines = _step_config_summary(_trust(deterministic_checks=[
        {"name": "still_breaching", "type": "shell", "passed": True, "detail": "", "duration_ms": 1},
        {"name": "sre_signoff", "type": "human", "passed": True, "detail": "", "duration_ms": 1},
    ]))

    assert any("still_breaching (shell)" in line and "sre_signoff (human)" in line for line in lines)


def test_calibration_enforced_states_n_min_and_policy():
    lines = _step_config_summary(_trust(calibration={
        "n": 5, "n_min": 20, "validated": False, "raw": 0.9, "calibrated": None,
        "on_uncalibrated": "proceed",
    }))

    assert any("20 marked results" in line and "proceed" in line for line in lines)


def test_calibration_bucket_reset_appends_reset_note():
    lines = _step_config_summary(_trust(calibration={
        "n": 4, "n_min": 20, "validated": False, "raw": 0.9, "calibrated": None,
        "on_uncalibrated": "proceed",
        "bucket_reset": {
            "reason": "prompt_changed", "previous_version_last_seen": "2026-07-03T16:02:51",
            "previous_validated_n": 47,
        },
    }))

    assert any("Reset:" in line and "prompt changed" in line and "47" in line for line in lines)


def test_calibration_no_bucket_reset_omits_reset_note():
    lines = _step_config_summary(_trust(calibration={
        "n": 20, "n_min": 20, "validated": True, "raw": 0.9, "calibrated": 0.8,
        "on_uncalibrated": "proceed",
    }))

    assert not any("Reset:" in line for line in lines)
