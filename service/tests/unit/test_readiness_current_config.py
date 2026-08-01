"""Tests for require_current_config semantics (SPEC-readiness-criteria.md §7f) —
the "current config" filter, its edge cases, and the "prompt changed" note."""
from datetime import datetime

from src.models.pipeline import (
    FanOutConfig,
    FanOutGroupConfig,
    ParallelGroupConfig,
    ParallelGroupInner,
    ParallelStepConfig,
    PipelineConfig,
    ReadinessAccuracyConfig,
    ReadinessConfig,
    StepConfig,
    TriggerConfig,
)
from src.pipeline.calibration import LABEL_HUMAN
from src.pipeline.versioning import prompt_hash as _compute_prompt_hash
from src.readiness import ComboEvidence, ReadinessEvidence, StepEvidence, evaluate_readiness

NOW = datetime(2026, 8, 1, 12, 0, 0)


def _row(run_id, *, prompt_hash, agent_version="v1", label=1.0, executed_at=None):
    return {
        "run_id": run_id, "status": "completed", "executed_at": executed_at or NOW,
        "predicted": 0.9, "label": label, "label_source": LABEL_HUMAN,
        "agent": "a", "model": "m", "provider": "pr",
        "prompt_hash": prompt_hash, "agent_version": agent_version,
    }


def _evidence(rows, step_name="s", latest_agent_version="v1") -> ReadinessEvidence:
    runs: dict[str, list] = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    se = StepEvidence(rows=rows, runs=runs, latest_agent_version=latest_agent_version if rows else None)
    return ReadinessEvidence(pipeline_name="p", evidence_stage="testing", gathered_at=NOW, by_step={step_name: se})


def _pipeline_with_step(step) -> PipelineConfig:
    return PipelineConfig(name="p", stage="testing", trigger=TriggerConfig(), steps=[step])


def test_prompt_edited_excludes_evidence_and_emits_note():
    old_hash = "old-hash-value"
    new_template = "brand new prompt"
    rows = [_row(f"r{i}", prompt_hash=old_hash) for i in range(5)]
    evidence = _evidence(rows)
    step = StepConfig(
        name="s", executor="gateway", prompt_template=new_template,
        readiness=ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.1, min_marked=1)),
    )
    result = evaluate_readiness(evidence, _pipeline_with_step(step))
    step_result = result["steps"][0]

    assert step_result["current_config"]["prompt_hash_matches_history"] is False
    assert step_result["tiers"]["accuracy"]["marked"] == 0
    assert step_result["tiers"]["accuracy"]["verdict"] == "insufficient_data"
    assert step_result["notes"], "expected a note explaining the exclusion"
    assert "5" in step_result["notes"][0]


def test_require_current_config_false_includes_stale_evidence():
    old_hash = "old-hash-value"
    rows = [_row(f"r{i}", prompt_hash=old_hash) for i in range(5)]
    evidence = _evidence(rows)
    step = StepConfig(
        name="s", executor="gateway", prompt_template="brand new prompt",
        readiness=ReadinessConfig(accuracy=ReadinessAccuracyConfig(
            min_accuracy=0.1, min_marked=1, require_current_config=False,
        )),
    )
    result = evaluate_readiness(evidence, _pipeline_with_step(step))
    step_result = result["steps"][0]
    assert step_result["tiers"]["accuracy"]["marked"] == 5
    assert step_result["notes"] == []


def test_non_llm_step_filter_is_inert():
    rows = [_row(f"r{i}", prompt_hash=None) for i in range(5)]
    evidence = _evidence(rows)
    step = StepConfig(
        name="s", executor="notify",
        readiness=ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.1, min_marked=1)),
    )
    result = evaluate_readiness(evidence, _pipeline_with_step(step))
    step_result = result["steps"][0]
    assert step_result["current_config"]["prompt_hash"] is None
    assert step_result["current_config"]["prompt_hash_matches_history"] is True
    assert step_result["tiers"]["accuracy"]["marked"] == 5


def test_two_agent_versions_only_newest_counts():
    older = NOW.replace(year=2025)
    rows = (
        [_row(f"old{i}", prompt_hash=_compute_prompt_hash("hi"), agent_version="v-old", executed_at=older)
         for i in range(3)]
        + [_row(f"new{i}", prompt_hash=_compute_prompt_hash("hi"), agent_version="v-new") for i in range(4)]
    )
    runs: dict[str, list] = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    # latest_agent_version must be derived from the most recent executed_at, not seed order.
    se = StepEvidence(rows=rows, runs=runs, latest_agent_version="v-new")
    evidence = ReadinessEvidence(pipeline_name="p", evidence_stage="testing", gathered_at=NOW, by_step={"s": se})

    step = StepConfig(
        name="s", executor="gateway", prompt_template="hi",
        readiness=ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.1, min_marked=1)),
    )
    result = evaluate_readiness(evidence, _pipeline_with_step(step))
    step_result = result["steps"][0]
    assert step_result["tiers"]["accuracy"]["marked"] == 4


def test_no_history_agent_version_source_unknown():
    evidence = _evidence([])
    step = StepConfig(name="s", executor="gateway", prompt_template="hi", readiness=ReadinessConfig(
        accuracy=ReadinessAccuracyConfig(min_accuracy=0.1, min_marked=1),
    ))
    result = evaluate_readiness(evidence, _pipeline_with_step(step))
    step_result = result["steps"][0]
    assert step_result["current_config"]["agent_version_source"] == "unknown"
    assert step_result["current_config"]["agent_version"] is None


def test_parallel_group_prompt_hash_not_applicable():
    rows = [_row(f"r{i}", prompt_hash=None) for i in range(5)]
    evidence = _evidence(rows, step_name="g")
    group = ParallelGroupConfig(parallel=ParallelGroupInner(
        name="g",
        readiness=ReadinessConfig(accuracy=ReadinessAccuracyConfig(min_accuracy=0.1, min_marked=1)),
        steps=[
            ParallelStepConfig(name="a", executor="gateway", prompt_template="x"),
            ParallelStepConfig(name="b", executor="gateway", prompt_template="y"),
        ],
    ))
    pipeline = PipelineConfig(name="p", stage="testing", trigger=TriggerConfig(), steps=[group])
    result = evaluate_readiness(evidence, pipeline)
    step_result = result["steps"][0]
    assert step_result["current_config"]["prompt_hash"] is None
    assert step_result["current_config"]["prompt_hash_source"] == "not_applicable_parallel_group"
