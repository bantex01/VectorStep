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
from src.pipeline.calibration import CalibrationBin, CalibrationBucket, CalibrationCache, compute_calibration_buckets
from src.pipeline.runner import PipelineRunner
from src.ui import _calibration_recommendation


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


def _make_output(confidence=0.9, model=None, provider=None) -> LLMOutput:
    return LLMOutput(
        confidence=confidence, summary="ok", next_step_context="",
        raw_response={}, model=model, provider=provider,
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
    effective_confidence=0.9, deterministic_passed=None, step_feedback=None, index=0,
) -> str:
    async with sf() as session:
        step = PipelineStep(
            run_id=run_id, step_name=step_name, step_index=index, executor="gateway",
            agent=agent, model=model, provider=provider, prompt="p", status="completed",
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

    bucket = buckets[("investigate", None, None, None)]
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

    bucket = buckets[("investigate", None, None, None)]
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

    bucket = buckets[("investigate", None, None, None)]
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

    assert ("investigate", None, None, None) not in buckets


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
    bucket = buckets[("investigate", None, None, None)]

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
    bin_at_5 = buckets[("investigate", None, None, None)].lookup(0.9)
    assert bin_at_5.n == 5
    assert bin_at_5.validated is True

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=6)
    bin_at_5_of_6 = buckets[("investigate", None, None, None)].lookup(0.9)
    assert bin_at_5_of_6.n == 5
    assert bin_at_5_of_6.validated is False


async def test_fan_out_branches_collapse_into_one_bucket(db):
    sf = await _init()
    await _seed_run(sf, "r1")
    await _seed_step(sf, "r1", "triage/0", effective_confidence=0.9, step_feedback="correct", index=0)
    await _seed_step(sf, "r1", "triage/1", effective_confidence=0.8, step_feedback="incorrect", index=1)

    buckets = await compute_calibration_buckets(sf, bin_width=0.5, n_min=1)

    assert ("triage", None, None, None) in buckets
    assert buckets[("triage", None, None, None)].total_n == 2
    assert not any(k[0] == "triage/0" or k[0] == "triage/1" for k in buckets)


async def test_bin_width_not_evenly_dividing_one_raises(db):
    sf = await _init()
    with pytest.raises(AssertionError):
        await compute_calibration_buckets(sf, bin_width=0.3, n_min=1)


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

    await cache.get("investigate", None, None, None)
    await cache.get("investigate", None, None, None)

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
    await cache.get("investigate", None, None, None)
    assert call_count == 1

    current_time[0] = 1050.0  # within TTL
    await cache.get("investigate", None, None, None)
    assert call_count == 1

    current_time[0] = 1200.0  # past TTL
    await cache.get("investigate", None, None, None)
    assert call_count == 2


async def test_cache_get_missing_bucket_returns_none(db):
    sf = await _init()
    cache = CalibrationCache(sf, ttl_seconds=1000)

    result = await cache.get("does-not-exist", None, None, None)

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
# UI — _calibration_recommendation
# ---------------------------------------------------------------------------

def _bucket_with_bins(bins: list[CalibrationBin]) -> CalibrationBucket:
    return CalibrationBucket(step_name="s", agent=None, model=None, provider=None, bins=bins, total_n=sum(b.n for b in bins))


def test_recommendation_none_when_no_validated_bins():
    bucket = _bucket_with_bins([
        CalibrationBin(lo=0.0, hi=0.5, n=2, mean_label=0.0, validated=False),
    ])
    assert _calibration_recommendation(bucket) is None


def test_recommendation_none_when_validated_bins_close_to_diagonal():
    bucket = _bucket_with_bins([
        CalibrationBin(lo=0.5, hi=1.0, n=20, mean_label=0.72, validated=True),  # midpoint 0.75, diff 0.03
    ])
    assert _calibration_recommendation(bucket) is None


def test_recommendation_returned_for_divergent_validated_bin():
    bucket = _bucket_with_bins([
        CalibrationBin(lo=0.5, hi=0.9, n=40, mean_label=0.5, validated=True),  # midpoint 0.7, diff 0.2
    ])
    msg = _calibration_recommendation(bucket)
    assert msg is not None
    assert "70%" in msg
    assert "50%" in msg
    assert "40 marked" in msg
