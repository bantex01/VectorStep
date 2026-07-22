"""Tests for _confidence_narrative — the per-run, plain-language walkthrough of how a
step's trust score was derived, shown behind the "How was this calculated?" button on
the run-detail page."""
from src.ui import _confidence_narrative


def _trust(**overrides) -> dict:
    base = {
        "signals": {"S": 0.85, "S_after_V": 0.85, "V": None, "V_mode": None, "G": None, "D": None},
        "calibration": None,
        "grounding": None,
        "deterministic_checks": None,
        "combined_trust": 0.85,
    }
    base.update(overrides)
    return base


def test_no_verifier_or_gates_just_states_self_report_and_outcome():
    lines = _confidence_narrative(_trust(), "completed")

    assert "85%" in lines[0]
    assert "completed normally" in lines[-1]
    assert "85%" in lines[-1]


def test_verifier_lowering_confidence_is_described():
    trust = _trust(signals={"S": 0.85, "S_after_V": 0.55, "V": 0.55, "V_mode": "critic", "G": None, "D": None})
    lines = _confidence_narrative(trust, "completed")

    assert any("55%" in line and "pulled" in line for line in lines)


def test_verifier_not_lowering_confidence_is_described_as_no_change():
    trust = _trust(signals={"S": 0.85, "S_after_V": 0.85, "V": 0.95, "V_mode": "independent", "G": None, "D": None})
    lines = _confidence_narrative(trust, "completed")

    assert any("blind" in line for line in lines)
    assert any("stood" in line for line in lines)


def test_calibration_validated_replaces_the_score():
    trust = _trust(
        calibration={
            "n": 20, "n_min": 20, "validated": True, "raw": 0.85, "calibrated": 0.62,
            "on_uncalibrated": "proceed",
        },
        combined_trust=0.62,
    )
    lines = _confidence_narrative(trust, "escalated")

    assert any("62%" in line and "track record" in line for line in lines)
    assert "62%" in lines[-1]
    assert "escalated" in lines[-1]


def test_calibration_unvalidated_escalate_forces_zero():
    trust = _trust(calibration={
        "n": 3, "n_min": 20, "validated": False, "raw": 0.85, "calibrated": None,
        "on_uncalibrated": "escalate",
    }, combined_trust=0.0)
    lines = _confidence_narrative(trust, "escalated")

    assert any("0%" in line and "play it safe" in line for line in lines)


def test_calibration_unvalidated_proceed_uses_raw_score():
    trust = _trust(calibration={
        "n": 3, "n_min": 20, "validated": False, "raw": 0.85, "calibrated": None,
        "on_uncalibrated": "proceed",
    })
    lines = _confidence_narrative(trust, "completed")

    assert any("as-is" in line for line in lines)


def test_grounding_enforced_and_capping():
    trust = _trust(grounding={"computed": True, "score": 0.4, "enforce": True}, combined_trust=0.4)
    lines = _confidence_narrative(trust, "escalated")

    assert any("40%" in line and "capped" in line for line in lines)


def test_grounding_enforced_but_no_effect():
    trust = _trust(grounding={"computed": True, "score": 0.95, "enforce": True})
    lines = _confidence_narrative(trust, "completed")

    assert any("95%" in line and "didn't change anything" in line for line in lines)


def test_grounding_shadow_only_is_labelled_as_not_affecting_outcome():
    trust = _trust(grounding={"computed": True, "score": 0.4, "enforce": False})
    lines = _confidence_narrative(trust, "completed")

    assert any("visibility only" in line for line in lines)


def test_grounding_not_computed_has_no_effect_note():
    trust = _trust(grounding={"computed": False, "reason": "error", "enforce": True})
    lines = _confidence_narrative(trust, "completed")

    assert any("couldn't complete" in line for line in lines)


def test_deterministic_check_failure_forces_zero():
    trust = _trust(
        deterministic_checks=[{"name": "still_breaching", "type": "shell", "passed": False, "detail": "x"}],
        combined_trust=0.0,
    )
    lines = _confidence_narrative(trust, "escalated")

    assert any("still_breaching" in line and "0%" in line for line in lines)


def test_deterministic_checks_all_passed_no_effect():
    trust = _trust(
        deterministic_checks=[{"name": "still_breaching", "type": "shell", "passed": True, "detail": "x"}],
    )
    lines = _confidence_narrative(trust, "completed")

    assert any("passed" in line and "no additional effect" in line for line in lines)
