"""Tests for the calibration loop (SPEC-calibration.md): compute_calibration_buckets
(label precedence, binning, fan-out collapse), CalibrationCache TTL behaviour, gate
integration (the critical tests), TrustReport shape, and the UI recommendation helper."""
from datetime import datetime

import pytest

from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, StepFeedback, RunFeedback
from src.models.context import NormalisedContext
from src.models.llm import LLMOutput
from src.models.pipeline import CalibrationConfig, GroundingConfig, PipelineConfig, StepConfig, TriggerConfig
from src.pipeline import calibration as calibration_module
from src.pipeline.calibration import (
    LABEL_DETERMINISTIC,
    LABEL_HUMAN,
    LABEL_RUN_FALLBACK,
    CalibrationBin,
    CalibrationBucket,
    CalibrationCache,
    calibration_recommendation,
    compute_calibration_buckets,
    resolve_label,
)
from src.pipeline.runner import PipelineRunner
from src.pipeline.versioning import prompt_hash as prompt_hash_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubExecutor:
    def __init__(self, output):
        self._output = output

    async def execute(self, step, ctx):
        return self._output


def _make_normalised(**kwargs) -> NormalisedContext:
    defaults = dict(
        source="test", pipeline="test-pipeline", severity="warning",
        summary="test alert", labels={}, metadata={}, raw={},
        received_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    defaults.update(kwargs)
    return NormalisedContext(**defaults)


def _make_output(confidence=0.9, model=None, provider=None, agent_version=None) -> LLMOutput:
    return LLMOutput(
        confidence=confidence, summary="ok", next_step_context="",
        raw_response={}, model=model, provider=provider, agent_version=agent_version,
    )


async def _init():
    return get_session_factory()


async def _seed_run(sf, run_id: str, stage: str = "production"):
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name="p", source="test",
            normalised_context="{}", raw_payload="{}", stage=stage,
        ))
        await session.commit()


async def _seed_step(
    sf, run_id: str, step_name: str, *, agent=None, model=None, provider=None,
    prompt_hash=None, agent_version=None,
    effective_confidence=0.9, deterministic_passed=None, step_feedback=None, index=0,
) -> str:
    async with sf() as session:
        step = PipelineStep(
            run_id=run_id, step_name=step_name, step_index=index, executor="gateway",
            agent=agent, model=model, provider=provider, prompt="p", status="completed",
            prompt_hash=prompt_hash, agent_version=agent_version,
            effective_confidence=effective_confidence, deterministic_passed=deterministic_passed,
        )
        session.add(step)
        await session.flush()
        step_id = step.id
        if step_feedback is not None:
            session.add(StepFeedback(
                step_id=step_id, run_id=run_id, pipeline_name="p", step_name=step_name,
                outcome=step_feedback,
            ))
        await session.commit()
    return step_id


async def _seed_run_feedback(sf, run_id: str, outcome: str):
    async with sf() as session:
        session.add(RunFeedback(run_id=run_id, pipeline_name="p", outcome=outcome))
        await session.commit()


# ---------------------------------------------------------------------------
# compute_calibration_buckets — label precedence
# ---------------------------------------------------------------------------

async def test_human_feedback_used_as_label(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_step(sf, "r1", "investigate", effective_confidence=0.9, step_feedback="correct")

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    bucket = buckets[("investigate", None, None, None, None, None)]
    assert bucket.total_n == 1
    bin_ = bucket.lookup(0.9)
    assert bin_.n == 1
    assert bin_.mean_label == 1.0


async def test_deterministic_failure_labels_zero_without_step_feedback(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_run(sf, "r2")
    await _seed_step(sf, "r1", "investigate", effective_confidence=0.9, deterministic_passed=False, index=0)
    # A passing check with no other label is excluded — not a positive label.
    await _seed_step(sf, "r2", "investigate", effective_confidence=0.9, deterministic_passed=True, index=0)

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    bucket = buckets[("investigate", None, None, None, None, None)]
    assert bucket.total_n == 1
    assert bucket.lookup(0.9).mean_label == 0.0


async def test_run_feedback_fallback_used_only_when_no_step_level_label(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_run(sf, "r2")
    # r1's step has neither StepFeedback nor a failed deterministic check -> falls back to RunFeedback.
    await _seed_step(sf, "r1", "investigate", effective_confidence=0.9, index=0)
    await _seed_run_feedback(sf, "r1", "incorrect")
    # r2's step HAS a StepFeedback row that conflicts with its RunFeedback -> human wins.
    await _seed_step(sf, "r2", "investigate", effective_confidence=0.9, step_feedback="correct", index=0)
    await _seed_run_feedback(sf, "r2", "incorrect")

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    bucket = buckets[("investigate", None, None, None, None, None)]
    labels = sorted(pair for b in bucket.bins for pair in [b.mean_label] * b.n if b.n)
    # r1 -> 0.0 (run-level fallback), r2 -> 1.0 (human overrides conflicting run feedback)
    assert bucket.total_n == 2
    assert bucket.lookup(0.9).n == 2
    assert bucket.lookup(0.9).mean_label == 0.5


async def test_step_execution_with_no_label_source_is_excluded(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_step(sf, "r1", "investigate", effective_confidence=0.9, index=0)  # no feedback of any kind

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    assert ("investigate", None, None, None, None, None) not in buckets


# ---------------------------------------------------------------------------
# compute_calibration_buckets — binning + validation boundary
# ---------------------------------------------------------------------------

async def test_binning_computes_correct_n_and_mean_per_bin(db):
    sf = await _init()
    for i, (predicted, outcome) in enumerate([(0.2, "correct"), (0.3, "incorrect"), (0.7, "correct")]):
        run_id = f"r{i}"
        await _seed_run(sf, run_id)
        await _seed_step(sf, run_id, "investigate", effective_confidence=predicted, step_feedback=outcome, index=0)

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)
    bucket = buckets[("investigate", None, None, None, None, None)]

    assert len(bucket.bins) == 2
    lo_bin, hi_bin = bucket.bins
    assert lo_bin.lo == 0.0 and lo_bin.hi == 0.5
    assert lo_bin.n == 2 and lo_bin.mean_label == 0.5   # correct(1.0) + incorrect(0.0)
    assert hi_bin.lo == 0.5 and hi_bin.hi == 1.0
    assert hi_bin.n == 1 and hi_bin.mean_label == 1.0


async def test_validated_flips_at_n_min_boundary(db):
    sf = await _init()
    for i in range(5):
        run_id = f"r{i}"
        await _seed_run(sf, run_id)
        await _seed_step(sf, run_id, "investigate", effective_confidence=0.9, step_feedback="correct", index=0)

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=5)
    bin_at_5 = buckets[("investigate", None, None, None, None, None)].lookup(0.9)
    assert bin_at_5.n == 5
    assert bin_at_5.validated is True

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=6)
    bin_at_5_of_6 = buckets[("investigate", None, None, None, None, None)].lookup(0.9)
    assert bin_at_5_of_6.n == 5
    assert bin_at_5_of_6.validated is False


async def test_fan_out_branches_collapse_into_one_bucket(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_step(sf, "r1", "triage/0", effective_confidence=0.9, step_feedback="correct", index=0)
    await _seed_step(sf, "r1", "triage/1", effective_confidence=0.8, step_feedback="incorrect", index=1)

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    assert ("triage", None, None, None, None, None) in buckets
    assert buckets[("triage", None, None, None, None, None)].total_n == 2
    assert not any(k[0] == "triage/0" or k[0] == "triage/1" for k in buckets)


async def test_bin_width_not_evenly_dividing_one_raises(db):
    sf = await _init()
    with pytest.raises(AssertionError):
        await compute_calibration_buckets(sf, bin_width=0.3, n_min=1)


# ---------------------------------------------------------------------------
# compute_calibration_buckets — prompt_hash / agent_version key dimensions
# (SPEC-prompt-versioning.md §4g)
# ---------------------------------------------------------------------------

async def test_different_prompt_hash_produces_separate_buckets(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_run(sf, "r2")
    await _seed_step(
        sf, "r1", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-old", effective_confidence=0.9, step_feedback="correct", index=0,
    )
    await _seed_step(
        sf, "r2", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-new", effective_confidence=0.9, step_feedback="incorrect", index=0,
    )

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    old_bucket = buckets[("investigate", "a", "m", "p", "hash-old", None)]
    new_bucket = buckets[("investigate", "a", "m", "p", "hash-new", None)]
    assert old_bucket.total_n == 1 and old_bucket.lookup(0.9).mean_label == 1.0
    assert new_bucket.total_n == 1 and new_bucket.lookup(0.9).mean_label == 0.0


async def test_different_agent_version_produces_separate_buckets(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_run(sf, "r2")
    await _seed_step(
        sf, "r1", "investigate", agent="a", model="m", provider="p",
        agent_version="v-old", effective_confidence=0.9, step_feedback="correct", index=0,
    )
    await _seed_step(
        sf, "r2", "investigate", agent="a", model="m", provider="p",
        agent_version="v-new", effective_confidence=0.9, step_feedback="incorrect", index=0,
    )

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    old_bucket = buckets[("investigate", "a", "m", "p", None, "v-old")]
    new_bucket = buckets[("investigate", "a", "m", "p", None, "v-new")]
    assert old_bucket.total_n == 1 and old_bucket.lookup(0.9).mean_label == 1.0
    assert new_bucket.total_n == 1 and new_bucket.lookup(0.9).mean_label == 0.0


async def test_null_prompt_hash_is_never_a_wildcard(db):
    """Regression lock for SPEC-prompt-versioning.md §2: a row with prompt_hash
    IS NULL (pre-migration / non-LLM steps) must never be treated as matching
    a row with a real hash — NULL is its own bucket dimension, not a grace
    period. Do not "fix" this by adding NULL-as-wildcard matching."""
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_run(sf, "r2")
    await _seed_step(
        sf, "r1", "investigate", agent="a", model="m", provider="p",
        prompt_hash=None, effective_confidence=0.9, step_feedback="correct", index=0,
    )
    await _seed_step(
        sf, "r2", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-real", effective_confidence=0.9, step_feedback="correct", index=0,
    )

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    null_bucket = buckets[("investigate", "a", "m", "p", None, None)]
    real_bucket = buckets[("investigate", "a", "m", "p", "hash-real", None)]
    assert null_bucket.total_n == 1
    assert real_bucket.total_n == 1
    assert null_bucket is not real_bucket


async def test_reverting_to_previous_prompt_hash_rejoins_original_bucket(db):
    sf = await _init()
    await _seed_run(sf, "r1")  # original version
    await _seed_run(sf, "r2")  # edited version
    await _seed_run(sf, "r3")  # reverted back to original — same hash as r1
    await _seed_step(
        sf, "r1", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-v1", effective_confidence=0.9, step_feedback="correct", index=0,
    )
    await _seed_step(
        sf, "r2", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-v2", effective_confidence=0.9, step_feedback="incorrect", index=0,
    )
    await _seed_step(
        sf, "r3", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-v1", effective_confidence=0.9, step_feedback="correct", index=0,
    )

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    v1_bucket = buckets[("investigate", "a", "m", "p", "hash-v1", None)]
    v2_bucket = buckets[("investigate", "a", "m", "p", "hash-v2", None)]
    assert v1_bucket.total_n == 2  # r1 and r3 pooled together
    assert v2_bucket.total_n == 1


async def test_bucket_validated_under_one_version_not_validated_under_another(db):
    sf = await _init()
    for i in range(5):
        run_id = f"old{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent="a", model="m", provider="p",
            prompt_hash="hash-old", effective_confidence=0.9, step_feedback="correct", index=0,
        )
    # Only 2 labelled runs under the new prompt version — below n_min.
    for i in range(2):
        run_id = f"new{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent="a", model="m", provider="p",
            prompt_hash="hash-new", effective_confidence=0.9, step_feedback="correct", index=0,
        )

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=5)

    old_bin = buckets[("investigate", "a", "m", "p", "hash-old", None)].lookup(0.9)
    new_bin = buckets[("investigate", "a", "m", "p", "hash-new", None)].lookup(0.9)
    assert old_bin.validated is True
    assert new_bin.validated is False


# ---------------------------------------------------------------------------
# CalibrationCache
# ---------------------------------------------------------------------------

async def test_cache_reuses_within_ttl(db, monkeypatch):
    sf = await _init()
    call_count = 0
    original = calibration_module.compute_calibration_buckets

    async def _counting(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(calibration_module, "compute_calibration_buckets", _counting)
    cache = CalibrationCache(sf, ttl_seconds=1000)

    await cache.get("investigate", None, None, None, None, None)
    await cache.get("investigate", None, None, None, None, None)

    assert call_count == 1


async def test_cache_refetches_after_ttl_expires(db, monkeypatch):
    sf = await _init()
    call_count = 0
    original = calibration_module.compute_calibration_buckets

    async def _counting(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(calibration_module, "compute_calibration_buckets", _counting)
    current_time = [1000.0]
    monkeypatch.setattr(calibration_module.time, "time", lambda: current_time[0])

    cache = CalibrationCache(sf, ttl_seconds=100)
    await cache.get("investigate", None, None, None, None, None)
    assert call_count == 1

    current_time[0] = 1050.0  # within TTL
    await cache.get("investigate", None, None, None, None, None)
    assert call_count == 1

    current_time[0] = 1200.0  # past TTL
    await cache.get("investigate", None, None, None, None, None)
    assert call_count == 2


async def test_cache_get_missing_bucket_returns_none(db):
    sf = await _init()
    cache = CalibrationCache(sf, ttl_seconds=1000)

    result = await cache.get("does-not-exist", None, None, None, None, None)

    assert result is None


# ---------------------------------------------------------------------------
# Gate integration (the critical tests — do not skip)
# ---------------------------------------------------------------------------

async def test_enforced_validated_bucket_overrides_raw_trust_and_escalates(db):
    sf = await _init()
    agent_key = "gateway:sre-investigation"
    for i, label in enumerate(["incorrect", "incorrect", "partial"]):
        run_id = f"hist{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent=agent_key, model="claude-sonnet-5", provider="anthropic",
            effective_confidence=0.9, step_feedback=label, index=0,
        )
    # bin [0.5, 1.0) mean_label = (0.0 + 0.0 + 0.5) / 3 = 1/6

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=3, calibration_bin_width=0.5,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="", confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert result.trust_report["calibration"]["validated"] is True
    assert result.trust_report["combined_trust"] == pytest.approx(1 / 6)
    assert result.trust_report["combined_trust"] != result.effective_confidence
    assert result.status == "escalated"


async def test_enforced_unvalidated_bucket_default_proceed_is_advisory_only(db):
    sf = await _init()  # no history at all for this bucket

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=20, calibration_bin_width=0.1,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="", confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),  # on_uncalibrated defaults to "proceed"
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert result.trust_report["calibration"]["validated"] is False
    assert result.trust_report["combined_trust"] == result.effective_confidence == 0.9
    assert result.status == "completed"


async def test_enforced_unvalidated_bucket_escalate_policy_forces_zero(db):
    sf = await _init()  # no history at all for this bucket

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=20, calibration_bin_width=0.1,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="", confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True, on_uncalibrated="escalate"),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert result.trust_report["combined_trust"] == 0.0
    assert result.status == "escalated"


async def test_no_calibration_block_is_byte_for_byte_unchanged(db):
    sf = await _init()

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)}, session_factory=sf,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="", confidence_threshold=0.75, on_low_confidence="escalate",
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert result.effective_confidence == 0.9
    assert result.trust_report is None
    assert result.status == "completed"


async def test_calibration_seeds_combined_trust_before_grounding_min(db):
    sf = await _init()
    agent_key = "gateway:sre-investigation"
    for i, label in enumerate(["incorrect", "partial", "correct"]):
        run_id = f"hist{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent=agent_key, model="claude-sonnet-5", provider="anthropic",
            effective_confidence=0.9, step_feedback=label, index=0,
        )
    # bin [0.5, 1.0) mean_label = (0.0 + 0.5 + 1.0) / 3 = 0.5

    primary_output = _make_output(confidence=0.95, model="claude-sonnet-5", provider="anthropic")
    grounding_output = LLMOutput(confidence=0.8, summary="mostly supported", next_step_context="", raw_response={})
    runner = PipelineRunner(
        executors={
            "gateway": lambda: _StubExecutor(primary_output),
            "grounding_stub": lambda: _StubExecutor(grounding_output),
        },
        session_factory=sf, calibration_n_min=3, calibration_bin_width=0.5,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="", confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),
        grounding=GroundingConfig(executor="grounding_stub", enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    # If grounding had capped the RAW 0.95 instead, min(0.95, 0.8) == 0.8 — but calibration
    # seeds first, so grounding's min() applies on top of the calibrated 0.5, not raw.
    assert result.trust_report["combined_trust"] == 0.5


# ---------------------------------------------------------------------------
# TrustReport shape
# ---------------------------------------------------------------------------

async def test_trust_report_calibration_shape_for_enforced_step(db):
    sf = await _init()
    agent_key = "gateway:sre-investigation"
    for i in range(3):
        run_id = f"hist{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent=agent_key, model="claude-sonnet-5", provider="anthropic",
            effective_confidence=0.9, step_feedback="correct", index=0,
        )

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=3, calibration_bin_width=0.5,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="", confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    calib = result.trust_report["calibration"]
    assert set(calib.keys()) == {
        "bucket", "bin", "n", "n_min", "validated", "raw", "calibrated", "on_uncalibrated",
    }
    assert calib["bucket"] == {
        "step_name": "investigate", "agent": agent_key,
        "model": "claude-sonnet-5", "provider": "anthropic",
    }
    assert calib["n"] == 3
    assert calib["n_min"] == 3
    assert calib["validated"] is True
    assert calib["raw"] == 0.9
    assert calib["calibrated"] == 1.0
    assert calib["on_uncalibrated"] == "proceed"


async def test_trust_report_calibration_absent_for_non_enforced_step(db):
    sf = await _init()

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    grounding_output = LLMOutput(confidence=0.9, summary="ok", next_step_context="", raw_response={})
    runner = PipelineRunner(
        executors={
            "gateway": lambda: _StubExecutor(primary_output),
            "grounding_stub": lambda: _StubExecutor(grounding_output),
        },
        session_factory=sf,
    )
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        confidence_threshold=0.75, on_low_confidence="escalate",
        grounding=GroundingConfig(executor="grounding_stub", enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert result.trust_report is not None
    assert result.trust_report["calibration"] is None


# ---------------------------------------------------------------------------
# Gate integration — prompt_hash / agent_version scoping (SPEC-prompt-versioning.md §4g)
# ---------------------------------------------------------------------------

async def test_gate_uses_history_matching_current_prompt_hash(db):
    sf = await _init()
    agent_key = "gateway:sre-investigation"
    template = "Investigate this alert.\nBe concise."
    h = prompt_hash_fn(template)
    for i, label in enumerate(["incorrect", "incorrect", "partial"]):
        run_id = f"hist{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent=agent_key, model="claude-sonnet-5", provider="anthropic",
            prompt_hash=h, effective_confidence=0.9, step_feedback=label, index=0,
        )

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=3, calibration_bin_width=0.5,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template=template, confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert result.trust_report["calibration"]["validated"] is True


async def test_gate_edited_prompt_does_not_inherit_old_prompts_history(db):
    """The bug this whole spec exists to fix: editing prompt_template must NOT let
    the step keep drawing on the old prompt's calibration history."""
    sf = await _init()
    agent_key = "gateway:sre-investigation"
    old_template = "Investigate this alert.\nBe concise."
    old_hash = prompt_hash_fn(old_template)
    for i, label in enumerate(["incorrect", "incorrect", "partial"]):
        run_id = f"hist{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent=agent_key, model="claude-sonnet-5", provider="anthropic",
            prompt_hash=old_hash, effective_confidence=0.9, step_feedback=label, index=0,
        )

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=3, calibration_bin_width=0.5,
    )
    new_template = "Investigate this alert.\nBe VERY thorough and check every dependency."
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template=new_template, confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),  # on_uncalibrated defaults to "proceed"
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert result.trust_report["calibration"]["validated"] is False
    assert result.trust_report["calibration"]["n"] == 0


async def test_gate_changed_agent_version_does_not_inherit_old_versions_history(db):
    """Same invariant as above, for the Gateway-reported agent_version dimension."""
    sf = await _init()
    agent_key = "gateway:sre-investigation"
    for i, label in enumerate(["incorrect", "incorrect", "partial"]):
        run_id = f"hist{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent=agent_key, model="claude-sonnet-5", provider="anthropic",
            agent_version="v-old", effective_confidence=0.9, step_feedback=label, index=0,
        )

    # This run's agent reports a DIFFERENT version — soul.md/agent.yaml changed on
    # the Gateway since the historical rows were recorded.
    primary_output = _make_output(
        confidence=0.9, model="claude-sonnet-5", provider="anthropic", agent_version="v-new",
    )
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=3, calibration_bin_width=0.5,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="", confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert result.trust_report["calibration"]["validated"] is False
    assert result.trust_report["calibration"]["n"] == 0
    bucket_reset = result.trust_report["calibration"]["bucket_reset"]
    assert bucket_reset["reason"] == "agent_changed"
    assert bucket_reset["previous_validated_n"] == 3
    assert bucket_reset["previous_version_last_seen"] is not None


async def test_gate_edited_prompt_bucket_reset_reports_prompt_changed(db):
    sf = await _init()
    agent_key = "gateway:sre-investigation"
    old_template = "Investigate this alert.\nBe concise."
    old_hash = prompt_hash_fn(old_template)
    for i, label in enumerate(["incorrect", "incorrect", "partial"]):
        run_id = f"hist{i}"
        await _seed_run(sf, run_id)
        await _seed_step(
            sf, run_id, "investigate", agent=agent_key, model="claude-sonnet-5", provider="anthropic",
            prompt_hash=old_hash, effective_confidence=0.9, step_feedback=label, index=0,
        )

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=3, calibration_bin_width=0.5,
    )
    new_template = "Investigate this alert.\nBe VERY thorough."
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template=new_template, confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    bucket_reset = result.trust_report["calibration"]["bucket_reset"]
    assert bucket_reset["reason"] == "prompt_changed"
    assert bucket_reset["previous_validated_n"] == 3
    assert bucket_reset["previous_version_last_seen"] is not None


async def test_gate_new_step_with_no_history_at_all_has_no_bucket_reset(db):
    """A step with genuinely no prior history (not a reset — just never run before)
    must not emit bucket_reset."""
    sf = await _init()  # no history at all

    primary_output = _make_output(confidence=0.9, model="claude-sonnet-5", provider="anthropic")
    runner = PipelineRunner(
        executors={"gateway": lambda: _StubExecutor(primary_output)},
        session_factory=sf, calibration_n_min=20, calibration_bin_width=0.1,
    )
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="", confidence_threshold=0.75, on_low_confidence="escalate",
        calibration=CalibrationConfig(enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="new-run", step_outputs={}, run_log=[],
    )

    assert "bucket_reset" not in result.trust_report["calibration"]


# ---------------------------------------------------------------------------
# CalibrationCache.previous_versions_for (SPEC-prompt-versioning.md §4h)
# ---------------------------------------------------------------------------

async def test_previous_versions_for_finds_other_version_of_same_combo(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_run(sf, "r2")
    await _seed_step(
        sf, "r1", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-old", effective_confidence=0.9, step_feedback="correct", index=0,
    )
    await _seed_step(
        sf, "r2", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-new", effective_confidence=0.9, step_feedback="incorrect", index=0,
    )
    cache = CalibrationCache(sf, bin_width=0.5, n_min=1)
    await cache.get("investigate", "a", "m", "p", "hash-new", None)  # warms the cache

    previous = cache.previous_versions_for("investigate", "a", "m", "p")

    assert len(previous) == 2  # both hash-old's and hash-new's buckets match the 4-component prefix
    assert {b.prompt_hash for b in previous} == {"hash-old", "hash-new"}


async def test_previous_versions_for_excludes_different_step_or_agent(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_run(sf, "r2")
    await _seed_step(
        sf, "r1", "investigate", agent="a", model="m", provider="p",
        prompt_hash="hash-a", effective_confidence=0.9, step_feedback="correct", index=0,
    )
    await _seed_step(
        sf, "r2", "other-step", agent="a", model="m", provider="p",
        prompt_hash="hash-b", effective_confidence=0.9, step_feedback="correct", index=0,
    )
    cache = CalibrationCache(sf, bin_width=0.5, n_min=1)
    await cache.get("investigate", "a", "m", "p", "hash-a", None)

    previous = cache.previous_versions_for("investigate", "a", "m", "p")

    assert len(previous) == 1
    assert previous[0].step_name == "investigate"


# ---------------------------------------------------------------------------
# calibration_recommendation
# ---------------------------------------------------------------------------

def _bucket_with_bins(bins: list[CalibrationBin]) -> CalibrationBucket:
    return CalibrationBucket(
        step_name="s", agent=None, model=None, provider=None, prompt_hash=None, agent_version=None,
        bins=bins, total_n=sum(b.n for b in bins), last_seen_at=None,
    )


def test_recommendation_none_when_no_validated_bins():
    bucket = _bucket_with_bins([
        CalibrationBin(lo=0.0, hi=0.5, n=2, mean_label=0.0, validated=False),
    ])
    assert calibration_recommendation(bucket) is None


def test_recommendation_none_when_validated_bins_close_to_diagonal():
    bucket = _bucket_with_bins([
        CalibrationBin(lo=0.5, hi=1.0, n=20, mean_label=0.72, validated=True),  # midpoint 0.75, diff 0.03
    ])
    assert calibration_recommendation(bucket) is None


def test_recommendation_returned_for_divergent_validated_bin():
    bucket = _bucket_with_bins([
        CalibrationBin(lo=0.5, hi=0.9, n=40, mean_label=0.5, validated=True),  # midpoint 0.7, diff 0.2
    ])
    msg = calibration_recommendation(bucket)
    assert msg is not None
    assert "70%" in msg
    assert "50%" in msg
    assert "40 marked" in msg


# ---------------------------------------------------------------------------
# calibration_recommendation — max_divergence / n_min overrides
# (SPEC-readiness-criteria.md §6b, extended for the readiness engine)
# ---------------------------------------------------------------------------

def test_max_divergence_override_suppresses_a_flag_that_fires_at_default():
    bucket = _bucket_with_bins([
        CalibrationBin(lo=0.5, hi=0.9, n=40, mean_label=0.5, validated=True),  # midpoint 0.7, diff 0.2
    ])
    assert calibration_recommendation(bucket) is not None
    assert calibration_recommendation(bucket, max_divergence=0.30) is None


def test_n_min_override_validates_a_bin_whose_baked_flag_is_false():
    bucket = _bucket_with_bins([
        CalibrationBin(lo=0.5, hi=0.9, n=5, mean_label=0.5, validated=False),  # baked False at some other n_min
    ])
    assert calibration_recommendation(bucket) is None            # respects the baked flag
    msg = calibration_recommendation(bucket, n_min=5)             # override treats n=5 as validated
    assert msg is not None


def test_resolve_label_precedence_table():
    # Human feedback beats a failed deterministic check beats a run-level fallback.
    assert resolve_label("correct", False, "incorrect") == (1.0, LABEL_HUMAN)
    assert resolve_label(None, False, "incorrect") == (0.0, LABEL_DETERMINISTIC)
    assert resolve_label(None, None, "partial") == (0.5, LABEL_RUN_FALLBACK)
    assert resolve_label(None, True, None) is None   # a PASSING check produces no label
    assert resolve_label(None, None, None) is None


# ---------------------------------------------------------------------------
# compute_calibration_buckets(stage=...) / CalibrationCache stage scoping
# Moved here from test_promotion_readiness.py (SPEC-readiness-criteria.md §14) —
# calibration tests that had landed in that file by history.
# ---------------------------------------------------------------------------

async def test_default_stage_excludes_testing_rows(db):
    sf = await _init()
    await _seed_run(sf, "run-1", "testing")
    await _seed_step(sf, "run-1", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="correct")

    buckets = await compute_calibration_buckets(sf)

    assert buckets == {}


async def test_testing_stage_excludes_production_rows_for_same_key(db):
    sf = await _init()
    await _seed_run(sf, "run-prod", "production")
    await _seed_step(sf, "run-prod", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="correct")
    await _seed_run(sf, "run-test", "testing")
    await _seed_step(sf, "run-test", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="partial", index=0)

    testing_buckets = await compute_calibration_buckets(sf, stage="testing")

    key = ("s", "a", "m", "pr", None, None)
    assert testing_buckets[key].total_n == 1
    assert testing_buckets[key].bins[9].mean_label == 0.5  # only the testing (partial) row


async def test_calibration_cache_stays_production_only(db):
    sf = await _init()
    await _seed_run(sf, "run-prod", "production")
    await _seed_step(sf, "run-prod", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="correct")
    await _seed_run(sf, "run-test", "testing")
    await _seed_step(sf, "run-test", "s", agent="a", model="m", provider="pr",
                      effective_confidence=0.9, step_feedback="incorrect")

    cache = CalibrationCache(sf)
    bucket = await cache.get("s", "a", "m", "pr", None, None)

    assert bucket is not None
    assert bucket.total_n == 1
    assert bucket.bins[9].mean_label == 1.0  # only the production (correct) row
