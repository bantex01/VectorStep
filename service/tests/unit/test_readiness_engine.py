"""Pure tests for the readiness evaluation engine (SPEC-readiness-criteria.md §7d/§7e):
every tier x verdict, roll-up precedence, label precedence/provenance, and the
traps in §12 that live in evaluate_readiness. No DB fixture — ReadinessEvidence
is built directly in memory."""
from datetime import datetime, timedelta

from src.models.pipeline import (
    PipelineConfig,
    ReadinessAccuracyConfig,
    ReadinessCalibrationConfig,
    ReadinessConfidenceConfig,
    ReadinessConfig,
    ReadinessOperationalConfig,
    StepConfig,
    TriggerConfig,
)
from src.pipeline.calibration import LABEL_DETERMINISTIC, LABEL_HUMAN, LABEL_RUN_FALLBACK
from src.pipeline.versioning import prompt_hash as _compute_prompt_hash
from src.readiness import ComboEvidence, ReadinessEvidence, StepEvidence, evaluate_readiness

NOW = datetime(2026, 8, 1, 12, 0, 0)
_HASH = _compute_prompt_hash("hi")   # matches _pipeline()'s default prompt_template="hi"


def _row(run_id, *, status="completed", predicted=0.9, label=None, label_source=None,
         agent="a", model="m", provider="pr", prompt_hash=_HASH, agent_version="v1",
         executed_at=None):
    return {
        "run_id": run_id, "status": status, "executed_at": executed_at or NOW,
        "predicted": predicted, "label": label, "label_source": label_source,
        "agent": agent, "model": model, "provider": provider,
        "prompt_hash": prompt_hash, "agent_version": agent_version,
    }


def _combo(rows, *, agent="a", model="m", provider="pr", prompt_hash=_HASH, agent_version="v1"):
    samples = [(r["predicted"], r["label"], r["label_source"]) for r in rows if r["label"] is not None]
    return ComboEvidence(
        agent=agent, model=model, provider=provider, prompt_hash=prompt_hash, agent_version=agent_version,
        samples=samples, rows=len(rows), runs=len({r["run_id"] for r in rows}),
        last_seen_at=max((r["executed_at"] for r in rows), default=None),
    )


def _evidence(rows: list[dict], step_name="s") -> ReadinessEvidence:
    runs: dict[str, list] = {}
    own_combos: dict[tuple, ComboEvidence] = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
        key = (r["agent"], r["model"], r["provider"], r["prompt_hash"], r["agent_version"])
        own_combos.setdefault(key, []).append(r)
    se = StepEvidence(
        rows=rows, runs=runs,
        own_combos={key: _combo(group, prompt_hash=key[3]) for key, group in own_combos.items()},
        latest_agent_version="v1" if rows else None,
        latest_prompt_hash=_HASH if rows else None,
    )
    return ReadinessEvidence(pipeline_name="p", evidence_stage="testing", gathered_at=NOW, by_step={step_name: se})


def _pipeline(readiness: ReadinessConfig | None, prompt_template="hi") -> PipelineConfig:
    return PipelineConfig(
        name="p", stage="testing", trigger=TriggerConfig(),
        steps=[StepConfig(name="s", executor="gateway", prompt_template=prompt_template, readiness=readiness)],
    )


def _step_result(rows, readiness, prompt_template="hi"):
    evidence = _evidence(rows)
    pipeline = _pipeline(readiness, prompt_template=prompt_template)
    result = evaluate_readiness(evidence, pipeline)
    return result["steps"][0]


# ---------------------------------------------------------------------------
# operational tier: pass | insufficient_data — never fail (§12.4)
# ---------------------------------------------------------------------------

def test_operational_not_configured():
    step = _step_result([], ReadinessConfig())
    assert step["tiers"]["operational"]["verdict"] == "not_configured"


def test_operational_insufficient_data_below_min_runs():
    rows = [_row(f"r{i}") for i in range(3)]
    step = _step_result(rows, ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)))
    assert step["tiers"]["operational"]["verdict"] == "insufficient_data"


def test_operational_pass_at_min_runs():
    rows = [_row(f"r{i}") for i in range(5)]
    step = _step_result(rows, ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)))
    assert step["tiers"]["operational"]["verdict"] == "pass"


def test_operational_never_fails_despite_many_failures():
    rows = [_row(f"r{i}", status="completed") for i in range(5)] + \
           [_row(f"f{i}", status="failed") for i in range(50)]
    step = _step_result(rows, ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)))
    assert step["tiers"]["operational"]["verdict"] == "pass"
    assert step["tiers"]["operational"]["status_counts"]["failed"] == 50


def test_operational_counts_distinct_runs_not_rows_fan_out(_row=_row):
    # One run with a 5-branch fan-out -> 5 rows, but runs_acceptable must be 1.
    rows = [_row("r0") for _ in range(5)]
    step = _step_result(rows, ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=1)))
    op = step["tiers"]["operational"]
    assert op["runs_total"] == 1
    assert op["runs_acceptable"] == 1
    assert op["verdict"] == "pass"


def test_operational_min_runs_20_not_satisfied_by_single_20_branch_fan_out():
    rows = [_row("r0") for _ in range(20)]
    step = _step_result(rows, ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=20)))
    assert step["tiers"]["operational"]["verdict"] == "insufficient_data"


def test_operational_run_not_acceptable_if_any_branch_failed():
    rows = [_row("r0", status="completed"), _row("r0", status="failed")]
    step = _step_result(rows, ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=1)))
    op = step["tiers"]["operational"]
    assert op["runs_total"] == 1
    assert op["runs_acceptable"] == 0


def test_operational_max_age_days_excludes_older_rows():
    old = NOW - timedelta(days=40)
    rows = [_row("old", executed_at=old)] + [_row(f"r{i}") for i in range(3)]
    readiness = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=4, max_age_days=30))
    step = _step_result(rows, readiness)
    assert step["tiers"]["operational"]["runs_total"] == 3
    assert step["tiers"]["operational"]["verdict"] == "insufficient_data"


def test_operational_acceptable_statuses_laxer_with_escalated():
    rows = [_row(f"r{i}", status="escalated") for i in range(5)]
    readiness = ReadinessConfig(operational=ReadinessOperationalConfig(
        min_runs=5, acceptable_statuses=["completed", "escalated"],
    ))
    step = _step_result(rows, readiness)
    assert step["tiers"]["operational"]["verdict"] == "pass"


# ---------------------------------------------------------------------------
# confidence tier
# ---------------------------------------------------------------------------

def test_confidence_insufficient_data_no_runs():
    step = _step_result([], ReadinessConfig(confidence=ReadinessConfidenceConfig(min_confidence=0.8)))
    assert step["tiers"]["confidence"]["verdict"] == "insufficient_data"


def test_confidence_insufficient_data_below_min_runs():
    rows = [_row("r0", predicted=0.95)]
    readiness = ReadinessConfig(confidence=ReadinessConfidenceConfig(min_confidence=0.8, min_runs=5))
    step = _step_result(rows, readiness)
    assert step["tiers"]["confidence"]["verdict"] == "insufficient_data"


def test_confidence_fail_below_bar():
    rows = [_row(f"r{i}", predicted=0.5) for i in range(5)]
    readiness = ReadinessConfig(confidence=ReadinessConfidenceConfig(min_confidence=0.8, min_runs=1))
    step = _step_result(rows, readiness)
    assert step["tiers"]["confidence"]["verdict"] == "fail"


def test_confidence_pass_at_bar():
    rows = [_row(f"r{i}", predicted=0.95) for i in range(5)]
    readiness = ReadinessConfig(confidence=ReadinessConfidenceConfig(min_confidence=0.8, min_runs=1))
    step = _step_result(rows, readiness)
    assert step["tiers"]["confidence"]["verdict"] == "pass"


# ---------------------------------------------------------------------------
# accuracy tier
# ---------------------------------------------------------------------------

def test_accuracy_insufficient_data_below_min_marked():
    rows = [_row("r0", label=1.0, label_source=LABEL_HUMAN)]
    readiness = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.5, min_marked=5))
    step = _step_result(rows, readiness)
    assert step["tiers"]["accuracy"]["verdict"] == "insufficient_data"


def test_accuracy_weighting_2_correct_1_partial_1_incorrect():
    rows = [
        _row("r0", label=1.0, label_source=LABEL_HUMAN),
        _row("r1", label=1.0, label_source=LABEL_HUMAN),
        _row("r2", label=0.5, label_source=LABEL_HUMAN),
        _row("r3", label=0.0, label_source=LABEL_HUMAN),
    ]
    readiness = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.5, min_marked=4))
    step = _step_result(rows, readiness)
    acc = step["tiers"]["accuracy"]
    assert acc["accuracy"] == 2.5 / 4
    assert acc["verdict"] == "pass"


def test_accuracy_fail_below_bar():
    rows = [_row(f"r{i}", label=0.0, label_source=LABEL_HUMAN) for i in range(5)]
    readiness = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.5, min_marked=5))
    step = _step_result(rows, readiness)
    assert step["tiers"]["accuracy"]["verdict"] == "fail"


def test_accuracy_min_human_marked_unmet_is_insufficient_not_fail():
    rows = [_row(f"r{i}", label=1.0, label_source=LABEL_DETERMINISTIC) for i in range(5)]
    readiness = ReadinessConfig(accuracy=ReadinessAccuracyConfig(
        min_accuracy=0.1, min_marked=1, min_human_marked=1,
    ))
    step = _step_result(rows, readiness)
    assert step["tiers"]["accuracy"]["verdict"] == "insufficient_data"


def test_accuracy_provenance_sums_to_marked():
    rows = [
        _row("r0", label=1.0, label_source=LABEL_HUMAN),
        _row("r1", label=0.0, label_source=LABEL_DETERMINISTIC),
        _row("r2", label=0.0, label_source=LABEL_RUN_FALLBACK),
    ]
    readiness = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.1, min_marked=1))
    step = _step_result(rows, readiness)
    acc = step["tiers"]["accuracy"]
    assert sum(acc["provenance"].values()) == acc["marked"] == 3


def test_accuracy_warns_when_100pct_deterministic_failures_no_human():
    rows = [_row(f"r{i}", label=0.0, label_source=LABEL_DETERMINISTIC) for i in range(5)]
    readiness = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.1, min_marked=1))
    step = _step_result(rows, readiness)
    assert step["tiers"]["accuracy"]["warnings"]


def test_accuracy_warns_when_majority_run_feedback_fallback():
    rows = [_row(f"r{i}", label=0.0, label_source=LABEL_RUN_FALLBACK) for i in range(3)] + \
           [_row("h0", label=1.0, label_source=LABEL_HUMAN)]
    readiness = ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.1, min_marked=1))
    step = _step_result(rows, readiness)
    assert any("run-level rating" in w for w in step["tiers"]["accuracy"]["warnings"])


# ---------------------------------------------------------------------------
# calibration tier
# ---------------------------------------------------------------------------

def test_calibration_insufficient_data_no_evidence():
    step = _step_result([], ReadinessConfig(calibration=ReadinessCalibrationConfig()))
    assert step["tiers"]["calibration"]["verdict"] == "insufficient_data"


def test_calibration_pass_when_validated_clean():
    rows = [_row(f"r{i}", predicted=0.95, label=1.0, label_source=LABEL_HUMAN) for i in range(20)]
    readiness = ReadinessConfig(calibration=ReadinessCalibrationConfig(n_min=20))
    step = _step_result(rows, readiness)
    assert step["tiers"]["calibration"]["verdict"] == "pass"


def test_calibration_fail_when_validated_and_diverging():
    # predicted ~0.95, but only half correct -> mean_label ~0.5, way off midpoint.
    rows = [
        _row(f"r{i}", predicted=0.95, label=1.0 if i % 2 == 0 else 0.0, label_source=LABEL_HUMAN)
        for i in range(20)
    ]
    readiness = ReadinessConfig(calibration=ReadinessCalibrationConfig(n_min=20))
    step = _step_result(rows, readiness)
    assert step["tiers"]["calibration"]["verdict"] == "fail"


def test_calibration_n_min_is_per_bin_not_total():
    # 100 marked results split evenly across bins of width 0.5 (2 bins) -> 50 each,
    # but with n_min=60 neither bin validates even though total is 100.
    rows = (
        [_row(f"lo{i}", predicted=0.2, label=0.0, label_source=LABEL_HUMAN) for i in range(50)]
        + [_row(f"hi{i}", predicted=0.8, label=1.0, label_source=LABEL_HUMAN) for i in range(50)]
    )
    readiness = ReadinessConfig(calibration=ReadinessCalibrationConfig(n_min=60, bin_width=0.5))
    step = _step_result(rows, readiness)
    assert step["tiers"]["calibration"]["verdict"] == "insufficient_data"


def test_calibration_only_current_config_combo_evaluated():
    current = [_row(f"c{i}", predicted=0.95, label=1.0, label_source=LABEL_HUMAN) for i in range(20)]
    stale = [_row(f"s{i}", predicted=0.5, label=0.0, label_source=LABEL_HUMAN, prompt_hash="OLD") for i in range(20)]
    evidence = _evidence(current + stale)
    pipeline = _pipeline(ReadinessConfig(calibration=ReadinessCalibrationConfig(n_min=20)))
    result = evaluate_readiness(evidence, pipeline)
    step = result["steps"][0]
    combos = step["tiers"]["calibration"]["combos"]
    stale_combo = next(c for c in combos if c["prompt_hash"] == "OLD")
    assert stale_combo["is_current_config"] is False
    assert stale_combo["verdict"] == "not_current_config"
    assert step["tiers"]["calibration"]["verdict"] == "pass"  # from the current-config combo alone


# ---------------------------------------------------------------------------
# roll-ups (§7e)
# ---------------------------------------------------------------------------

def test_step_rollup_all_not_configured():
    step = _step_result([], ReadinessConfig())
    assert step["verdict"] == "not_configured"


def test_step_rollup_fail_outranks_everything():
    rows = [_row(f"r{i}", predicted=0.5) for i in range(5)]
    readiness = ReadinessConfig(
        operational=ReadinessOperationalConfig(min_runs=1),
        confidence=ReadinessConfidenceConfig(min_confidence=0.9, min_runs=1),
    )
    step = _step_result(rows, readiness)
    assert step["tiers"]["operational"]["verdict"] == "pass"
    assert step["tiers"]["confidence"]["verdict"] == "fail"
    assert step["verdict"] == "not_ready"


def test_step_rollup_building_when_no_fail_but_insufficient():
    rows = [_row("r0", predicted=0.95)]
    readiness = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5))
    step = _step_result(rows, readiness)
    assert step["verdict"] == "building"


def test_step_rollup_no_data_when_zero_rows():
    readiness = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5))
    step = _step_result([], readiness)
    assert step["verdict"] == "no_data"


def test_step_rollup_ready_when_pass_and_no_fail():
    rows = [_row(f"r{i}") for i in range(5)]
    readiness = ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5))
    step = _step_result(rows, readiness)
    assert step["verdict"] == "ready"


def test_pipeline_rollup_not_ready_if_any_step_not_ready():
    evidence = ReadinessEvidence(
        pipeline_name="p", evidence_stage="testing", gathered_at=NOW,
        by_step={
            "ok": StepEvidence(rows=[_row(f"r{i}") for i in range(5)],
                                runs={f"r{i}": [_row(f"r{i}")] for i in range(5)}),
            "bad": StepEvidence(rows=[_row(f"b{i}", predicted=0.1) for i in range(5)],
                                 runs={f"b{i}": [_row(f"b{i}", predicted=0.1)] for i in range(5)}),
        },
    )
    pipeline = PipelineConfig(
        name="p", stage="testing", trigger=TriggerConfig(),
        readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5)),
        steps=[
            StepConfig(name="ok", executor="gateway", prompt_template="x",
                       readiness=ReadinessConfig(operational=ReadinessOperationalConfig(min_runs=5))),
            StepConfig(name="bad", executor="gateway", prompt_template="y",
                       readiness=ReadinessConfig(
                           operational=ReadinessOperationalConfig(min_runs=5),
                           confidence=ReadinessConfidenceConfig(min_confidence=0.9, min_runs=1),
                       )),
        ],
    )
    result = evaluate_readiness(evidence, pipeline)
    assert result["verdict"] == "not_ready"
    assert result["summary"]["not_ready"] == 1
    assert result["summary"]["ready"] == 1
