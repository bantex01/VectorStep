"""DB-backed tests for readiness.gather_readiness_evidence (SPEC-readiness-criteria.md
§7c) and the golden test proving compute_calibration_buckets' output is byte-identical
before/after the §6 calibration.py refactor — the safety net for touching the live-gate
path."""
from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, StepFeedback
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig
from src.pipeline.calibration import compute_calibration_buckets
from src.readiness import evaluate_readiness, gather_readiness_evidence


async def _seed_run(sf, run_id: str, pipeline_name: str, stage: str) -> None:
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name=pipeline_name, source="test",
            normalised_context="{}", raw_payload="{}", stage=stage,
        ))
        await session.commit()


async def _seed_step(
    sf, run_id: str, pipeline_name: str, step_name: str, *, agent=None, model=None, provider=None,
    prompt_hash=None, agent_version=None, effective_confidence=0.9, status="completed",
    step_feedback=None, executor="gateway", index=0,
) -> str:
    async with sf() as session:
        step = PipelineStep(
            run_id=run_id, step_name=step_name, step_index=index, executor=executor,
            agent=agent, model=model, provider=provider, prompt="p", status=status,
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


def _pipeline(name: str, stage: str, step_name: str = "s") -> PipelineConfig:
    return PipelineConfig(
        name=name, stage=stage, trigger=TriggerConfig(),
        steps=[StepConfig(name=step_name, executor="gateway", prompt_template="hi")],
    )


# ---------------------------------------------------------------------------
# gather_readiness_evidence
# ---------------------------------------------------------------------------

async def test_own_stage_query_is_pipeline_scoped(db):
    sf = get_session_factory()
    await _seed_run(sf, "r-a", "pipeline-a", "testing")
    await _seed_step(sf, "r-a", "pipeline-a", "s", agent="ag", model="m", provider="pr",
                      step_feedback="correct")
    await _seed_run(sf, "r-b", "pipeline-b", "testing")
    await _seed_step(sf, "r-b", "pipeline-b", "s", agent="ag", model="m", provider="pr",
                      step_feedback="correct")

    evidence = await gather_readiness_evidence(sf, _pipeline("pipeline-a", "testing"))

    se = evidence.by_step["s"]
    assert len(se.rows) == 1
    assert se.rows[0]["run_id"] == "r-a"


async def test_production_query_is_cross_pipeline(db):
    sf = get_session_factory()
    await _seed_run(sf, "r-a", "pipeline-a", "production")
    await _seed_step(sf, "r-a", "pipeline-a", "s", agent="ag", model="m", provider="pr",
                      step_feedback="correct")

    # pipeline-b never ran this step at all — but shares the library step "s".
    pipeline_b = _pipeline("pipeline-b", "testing")
    evidence = await gather_readiness_evidence(sf, pipeline_b)

    se = evidence.by_step["s"]
    assert len(se.rows) == 0                     # own-stage: nothing for pipeline-b
    key = ("ag", "m", "pr", None, None)
    assert key in se.prod_combos                  # production: cross-pipeline evidence visible
    assert se.prod_combos[key].rows == 1
    assert se.prod_pipelines[key] == {"pipeline-a"}


async def test_require_confidence_false_surfaces_notify_step(db):
    sf = get_session_factory()
    await _seed_run(sf, "r-a", "p", "testing")
    async with sf() as session:
        step = PipelineStep(
            run_id="r-a", step_name="notify-oncall", step_index=0, executor="notify",
            prompt="", status="completed", effective_confidence=None,
        )
        session.add(step)
        await session.commit()

    pipeline = PipelineConfig(
        name="p", stage="testing", trigger=TriggerConfig(),
        steps=[StepConfig(name="notify-oncall", executor="notify")],
    )
    evidence = await gather_readiness_evidence(sf, pipeline)
    se = evidence.by_step["notify-oncall"]
    assert len(se.rows) == 1
    assert se.rows[0]["predicted"] is None


async def test_labeled_row_with_no_confidence_is_excluded_from_calibration_samples(db):
    """A row can be LABELED (human feedback, a failed deterministic check, or a
    run-level outcome fallback) while effective_confidence is NULL — e.g. a run
    that errored before the model returned, or a non-confidence-bearing executor.
    own_rows is deliberately fetched with require_confidence=False so operational/
    accuracy still see it, but it must not become a calibration SAMPLE (predicted,
    label) — bins_from_samples compares `lo <= predicted < hi`, which crashes on
    a None predicted. This reproduces the crash a real testing-stage pipeline hit
    on GET /ui/pipelines/{name} the moment one such row existed."""
    sf = get_session_factory()
    await _seed_run(sf, "r-a", "p", "testing")
    await _seed_step(sf, "r-a", "p", "s", agent="a", model="m", provider="pr",
                      effective_confidence=None, step_feedback="correct")

    pipeline = _pipeline("p", "testing")
    evidence = await gather_readiness_evidence(sf, pipeline)

    combo = evidence.by_step["s"].own_combos[("a", "m", "pr", None, None)]
    assert combo.samples == []          # the labeled-but-unconfident row must not be a sample

    # evaluate_readiness always computes observed_combos (the "Observed (service
    # defaults)" fallback), regardless of whether any readiness: tier is configured —
    # this must not raise.
    result = evaluate_readiness(evidence, pipeline)
    assert result["steps"][0]["observed_combos"][0]["rows"] == 1
    assert result["steps"][0]["observed_combos"][0]["observed"]["bins"][0]["n"] == 0


async def test_step_names_narrowing_includes_fan_out_and_parallel_branches(db):
    sf = get_session_factory()
    await _seed_run(sf, "r-a", "p", "testing")
    await _seed_step(sf, "r-a", "p", "fanout/0", agent="a", model="m", provider="pr", step_feedback="correct")
    await _seed_run(sf, "r-b", "p", "testing")
    await _seed_step(sf, "r-b", "p", "fanout/1", agent="a", model="m", provider="pr", step_feedback="correct")

    pipeline = PipelineConfig(
        name="p", stage="testing", trigger=TriggerConfig(),
        steps=[StepConfig(name="fanout", executor="gateway", prompt_template="x")],
    )
    evidence = await gather_readiness_evidence(sf, pipeline)
    se = evidence.by_step["fanout"]
    assert len(se.rows) == 2
    assert len(se.runs) == 2


async def test_production_stage_pipeline_gathers_production_evidence(db):
    sf = get_session_factory()
    await _seed_run(sf, "r-a", "p", "production")
    await _seed_step(sf, "r-a", "p", "s", agent="a", model="m", provider="pr", step_feedback="correct")

    evidence = await gather_readiness_evidence(sf, _pipeline("p", "production"))
    assert evidence.evidence_stage == "production"
    se = evidence.by_step["s"]
    assert len(se.rows) == 1


async def test_exactly_two_db_round_trips_per_gather(db, monkeypatch):
    sf = get_session_factory()
    await _seed_run(sf, "r-a", "p", "testing")
    await _seed_step(sf, "r-a", "p", "s", agent="a", model="m", provider="pr", step_feedback="correct")

    import src.readiness as readiness_module
    calls = []
    real_fetch = readiness_module.fetch_label_rows

    async def _counting_fetch(*args, **kwargs):
        calls.append(1)
        return await real_fetch(*args, **kwargs)

    monkeypatch.setattr(readiness_module, "fetch_label_rows", _counting_fetch)
    await gather_readiness_evidence(sf, _pipeline("p", "testing"))
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Golden test: compute_calibration_buckets byte-identical pre/post §6 refactor
# ---------------------------------------------------------------------------

async def test_compute_calibration_buckets_golden_output(db):
    """A fixed, seeded dataset exercising every branch of the label-precedence
    chain and the fan-out collapse — asserts the exact shape and values
    compute_calibration_buckets returned BEFORE the §6 extraction. Any refactor
    that changes this output has broken the live production gate."""
    sf = get_session_factory()

    # Human feedback wins over a failed deterministic check.
    await _seed_run(sf, "r1", "p", "production")
    async with sf() as session:
        step = PipelineStep(
            run_id="r1", step_name="investigate", step_index=0, executor="gateway",
            agent="ag", model="m", provider="pr", prompt="p", status="completed",
            prompt_hash="h1", agent_version="v1", effective_confidence=0.92,
            deterministic_passed=False,
        )
        session.add(step)
        await session.flush()
        session.add(StepFeedback(step_id=step.id, run_id="r1", pipeline_name="p",
                                  step_name="investigate", outcome="correct"))
        await session.commit()

    # Deterministic failure, no human feedback -> labelled 0.0.
    await _seed_run(sf, "r2", "p", "production")
    await _seed_step(sf, "r2", "p", "investigate", agent="ag", model="m", provider="pr",
                      prompt_hash="h1", agent_version="v1", effective_confidence=0.15,
                      step_feedback=None)
    async with sf() as session:
        rows = (await session.execute(
            __import__("sqlalchemy").select(PipelineStep).where(PipelineStep.run_id == "r2")
        )).scalars().all()
        rows[0].deterministic_passed = False
        await session.commit()

    # Fan-out branches collapse into the group bucket.
    await _seed_run(sf, "r3", "p", "production")
    await _seed_step(sf, "r3", "p", "triage/0", agent="ag", model="m", provider="pr",
                      prompt_hash="h1", agent_version="v1", effective_confidence=0.8,
                      step_feedback="partial", index=0)
    await _seed_step(sf, "r3", "p", "triage/1", agent="ag", model="m", provider="pr",
                      prompt_hash="h1", agent_version="v1", effective_confidence=0.85,
                      step_feedback="incorrect", index=1)

    # A passing deterministic check with no other label produces NO sample at all.
    await _seed_run(sf, "r4", "p", "production")
    await _seed_step(sf, "r4", "p", "investigate", agent="ag", model="m", provider="pr",
                      prompt_hash="h1", agent_version="v1", effective_confidence=0.5,
                      step_feedback=None)

    buckets = await compute_calibration_buckets(sf, bin_width=0.1, n_min=1, stage="production")

    investigate = buckets[("investigate", "ag", "m", "pr", "h1", "v1")]
    assert investigate.total_n == 2
    assert investigate.lookup(0.92).mean_label == 1.0
    assert investigate.lookup(0.15).mean_label == 0.0

    triage = buckets[("triage", "ag", "m", "pr", "h1", "v1")]
    assert triage.total_n == 2
    # 0.8 (partial=0.5) and 0.85 (incorrect=0.0) both fall in the same [0.8, 0.9) bin.
    assert triage.lookup(0.8).n == 2
    assert triage.lookup(0.8).mean_label == 0.25

    assert ("investigate", "ag", "m", "pr", "h1", "v1") in buckets
    assert len(buckets) == 2
