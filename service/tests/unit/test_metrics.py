"""Tests for pork_pipeline_tokens_total, pork_human_approval_decisions_total,
pork_pipeline_feedback_total, and pork_human_approvals_pending."""
from datetime import datetime

from src.db.database import create_tables, init_db, get_session_factory
from src.db.models import PipelineRun, PipelineStep, RunFeedback
from src.executors import human
from src.metrics import MetricsData, PorkCollector, fetch_metrics_data


def _empty_metrics_data(
    token_usage: list[tuple] | None = None,
    human_decisions=None,
    feedback_counts=None,
    step_feedback_counts=None,
    grounding_scores=None,
    deterministic_check_counts=None,
) -> MetricsData:
    return MetricsData(
        run_counts=[],
        runs_in_progress=0,
        step_counts=[],
        step_durations=[],
        verifier_counts=[],
        token_usage=token_usage or [],
        human_decisions=human_decisions or [],
        feedback_counts=feedback_counts or [],
        step_feedback_counts=step_feedback_counts or [],
        grounding_scores=grounding_scores or [],
        deterministic_check_counts=deterministic_check_counts or [],
    )


def _find_family(families, sample_name):
    return next(f for f in families if any(s.name == sample_name for s in f.samples))


def test_collect_emits_pork_pipeline_tokens_total_with_expected_labels():
    data = _empty_metrics_data([
        ("payments", "alert-triage", "triage", "gateway", "sre-triage", "claude-sonnet-4-6", "anthropic", 100, 50),
    ])
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_pipeline_tokens_total")

    assert family.samples[0].labels == {
        "team": "payments", "pipeline": "alert-triage", "step_name": "triage", "executor": "gateway",
        "agent": "sre-triage", "model": "claude-sonnet-4-6", "provider": "anthropic", "direction": "input",
    }
    assert family.samples[0].value == 100
    assert family.samples[1].labels["direction"] == "output"
    assert family.samples[1].value == 50


def test_collect_buckets_null_team_and_model_as_empty_string():
    data = _empty_metrics_data([
        (None, "scheduled-pipeline", "triage", "gateway", "agent-x", None, None, 10, 5),
    ])
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_pipeline_tokens_total")

    assert family.samples[0].labels["team"] == ""
    assert family.samples[0].labels["model"] == ""
    assert family.samples[0].labels["provider"] == ""


async def test_fetch_metrics_data_excludes_steps_without_tokens(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    session_factory = get_session_factory()

    async with session_factory() as session:
        session.add(PipelineRun(
            id="run-1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="payments",
        ))
        session.add(PipelineStep(
            id="step-1", run_id="run-1", step_name="s1", step_index=0,
            executor="gateway", agent="a", model="m", prompt="", status="completed",
            executed_at=datetime(2026, 1, 1), input_tokens=100, output_tokens=50,
        ))
        session.add(PipelineStep(
            id="step-2", run_id="run-1", step_name="s2", step_index=1,
            executor="openclaw", agent="b", model=None, prompt="", status="completed",
            executed_at=datetime(2026, 1, 1), input_tokens=None, output_tokens=None,
        ))
        await session.commit()

    metrics_data = await fetch_metrics_data(session_factory)

    assert len(metrics_data.token_usage) == 1
    team, pipeline, step_name, executor, agent, model, provider, input_sum, output_sum = metrics_data.token_usage[0]
    assert (team, pipeline, step_name, executor, agent, model) == ("payments", "p", "s1", "gateway", "a", "m")
    assert (input_sum, output_sum) == (100, 50)


async def test_fetch_metrics_data_sums_across_steps(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    session_factory = get_session_factory()

    async with session_factory() as session:
        session.add(PipelineRun(
            id="run-1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="payments",
        ))
        for i in range(2):
            session.add(PipelineStep(
                id=f"step-{i}", run_id="run-1", step_name="s", step_index=i,
                executor="gateway", agent="a", model="m", prompt="", status="completed",
                executed_at=datetime(2026, 1, 1), input_tokens=10, output_tokens=5,
            ))
        await session.commit()

    metrics_data = await fetch_metrics_data(session_factory)

    assert len(metrics_data.token_usage) == 1
    _, _, _, _, _, _, _, input_sum, output_sum = metrics_data.token_usage[0]
    assert (input_sum, output_sum) == (20, 10)


# ---------------------------------------------------------------------------
# pork_human_approval_decisions_total
# ---------------------------------------------------------------------------

def test_collect_emits_human_approval_decisions():
    data = _empty_metrics_data(human_decisions=[
        ("barkham", "approval-test", "approved", 3),
        ("barkham", "approval-test", "rejected", 1),
    ])
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_human_approval_decisions_total")

    by_decision = {s.labels["decision"]: s.value for s in family.samples}
    assert by_decision == {"approved": 3, "rejected": 1}
    assert all(s.labels["team"] == "barkham" for s in family.samples)
    assert all(s.labels["pipeline"] == "approval-test" for s in family.samples)


def test_collect_buckets_null_team_in_human_decisions():
    data = _empty_metrics_data(human_decisions=[(None, "p", "approved", 1)])
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_human_approval_decisions_total")
    assert family.samples[0].labels["team"] == ""


async def test_fetch_metrics_data_derives_decision_from_primary_confidence(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    session_factory = get_session_factory()

    async with session_factory() as session:
        session.add(PipelineRun(
            id="run-1", pipeline_name="approval-test", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="barkham",
        ))
        session.add(PipelineStep(
            id="step-approved", run_id="run-1", step_name="human-approval", step_index=0,
            executor="human", agent=None, model=None, prompt="", status="completed",
            executed_at=datetime(2026, 1, 1), primary_confidence=1.0,
        ))
        session.add(PipelineStep(
            id="step-rejected", run_id="run-1", step_name="human-approval-2", step_index=1,
            executor="human", agent=None, model=None, prompt="", status="aborted",
            executed_at=datetime(2026, 1, 1), primary_confidence=0.0,
        ))
        # Timeout case — no output at all, primary_confidence NULL — must be excluded.
        session.add(PipelineStep(
            id="step-timeout", run_id="run-1", step_name="human-approval-3", step_index=2,
            executor="human", agent=None, model=None, prompt="", status="failed",
            executed_at=datetime(2026, 1, 1), primary_confidence=None,
        ))
        await session.commit()

    metrics_data = await fetch_metrics_data(session_factory)

    assert set(metrics_data.human_decisions) == {
        ("barkham", "approval-test", "approved", 1),
        ("barkham", "approval-test", "rejected", 1),
    }


# ---------------------------------------------------------------------------
# pork_pipeline_feedback_total
# ---------------------------------------------------------------------------

def test_collect_emits_pipeline_feedback_total():
    data = _empty_metrics_data(feedback_counts=[
        ("alert-triage", "correct", 5),
        ("alert-triage", "incorrect", 2),
    ])
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_pipeline_feedback_total")

    by_outcome = {s.labels["outcome"]: s.value for s in family.samples}
    assert by_outcome == {"correct": 5, "incorrect": 2}


async def test_fetch_metrics_data_counts_feedback_by_pipeline_and_outcome(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    session_factory = get_session_factory()

    async with session_factory() as session:
        for run_id in ("run-1", "run-2", "run-3"):
            session.add(PipelineRun(
                id=run_id, pipeline_name="p", source="generic", status="completed",
                normalised_context="{}", raw_payload="{}",
            ))
        session.add(RunFeedback(id="fb-1", run_id="run-1", pipeline_name="p", outcome="correct"))
        session.add(RunFeedback(id="fb-2", run_id="run-2", pipeline_name="p", outcome="correct"))
        session.add(RunFeedback(id="fb-3", run_id="run-3", pipeline_name="p", outcome="partial"))
        await session.commit()

    metrics_data = await fetch_metrics_data(session_factory)

    assert set(metrics_data.feedback_counts) == {("p", "correct", 2), ("p", "partial", 1)}


# ---------------------------------------------------------------------------
# pork_human_approvals_pending — live gauge, not derived from MetricsData
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# stage=testing exclusion — every DB-derived series in fetch_metrics_data
# ---------------------------------------------------------------------------

async def test_fetch_metrics_data_excludes_testing_runs(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    session_factory = get_session_factory()

    async with session_factory() as session:
        session.add(PipelineRun(
            id="run-prod", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="payments", stage="production",
        ))
        session.add(PipelineRun(
            id="run-test", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="payments", stage="testing",
        ))
        for run_id, suffix in (("run-prod", "a"), ("run-test", "b")):
            session.add(PipelineStep(
                id=f"step-{suffix}", run_id=run_id, step_name=f"s-{suffix}", step_index=0,
                executor="gateway", agent="agent-x", model="m", prompt="", status="completed",
                executed_at=datetime(2026, 1, 1), input_tokens=100, output_tokens=50,
                duration_ms=1000, verifier_confidence=0.9, primary_confidence=1.0,
                effective_confidence=0.5,
            ))
            session.add(PipelineStep(
                id=f"human-{suffix}", run_id=run_id, step_name=f"h-{suffix}", step_index=1,
                executor="human", agent=None, model=None, prompt="", status="completed",
                executed_at=datetime(2026, 1, 1), primary_confidence=1.0,
            ))
        session.add(RunFeedback(id="fb-prod", run_id="run-prod", pipeline_name="p", outcome="correct"))
        session.add(RunFeedback(id="fb-test", run_id="run-test", pipeline_name="p", outcome="incorrect"))
        await session.commit()

    metrics_data = await fetch_metrics_data(session_factory)

    assert metrics_data.run_counts == [("p", "completed", 1)]
    assert len(metrics_data.token_usage) == 1
    assert metrics_data.token_usage[0][:2] == ("payments", "p")
    assert len(metrics_data.human_decisions) == 1
    assert metrics_data.human_decisions[0][:2] == ("payments", "p")
    assert metrics_data.feedback_counts == [("p", "correct", 1)]
    # Step-only aggregates (no independent stage awareness pre-join) also exclude testing.
    assert sum(row[-1] for row in metrics_data.step_counts) == 2  # gateway + human, prod run only
    assert len(metrics_data.step_durations) == 1
    assert len(metrics_data.verifier_counts) == 1
    assert metrics_data.verifier_counts[0][1] == 1  # one verified step, from the prod run only


def test_pending_approvals_gauge_excludes_testing():
    human._pending_meta.clear()
    human._pending_meta["tok-prod"] = {
        "team": "barkham", "pipeline": "p", "step": "s", "run_id": "r",
        "message": "m", "stage": "production", "created_at": human.utc_now(),
    }
    human._pending_meta["tok-test"] = {
        "team": "barkham", "pipeline": "p", "step": "s", "run_id": "r",
        "message": "m", "stage": "testing", "created_at": human.utc_now(),
    }
    try:
        data = _empty_metrics_data()
        families = list(PorkCollector(data).collect())
        family = _find_family(families, "pork_human_approvals_pending")

        by_team = {s.labels["team"]: s.value for s in family.samples}
        assert by_team == {"barkham": 1}
    finally:
        human._pending_meta.clear()


def test_pending_approvals_gauge_zero_when_nothing_pending():
    human._pending_meta.clear()
    data = _empty_metrics_data()
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_human_approvals_pending")

    assert family.samples[0].value == 0
    assert family.samples[0].labels["team"] == ""


def test_pending_approvals_gauge_grouped_by_team():
    human._pending_meta.clear()
    human._pending_meta["tok-1"] = {"team": "barkham", "pipeline": "p", "step": "s",
                                     "run_id": "r", "message": "m", "created_at": human.utc_now()}
    human._pending_meta["tok-2"] = {"team": "barkham", "pipeline": "p", "step": "s",
                                     "run_id": "r", "message": "m", "created_at": human.utc_now()}
    human._pending_meta["tok-3"] = {"team": None, "pipeline": "p", "step": "s",
                                     "run_id": "r", "message": "m", "created_at": human.utc_now()}
    try:
        data = _empty_metrics_data()
        families = list(PorkCollector(data).collect())
        family = _find_family(families, "pork_human_approvals_pending")

        by_team = {s.labels["team"]: s.value for s in family.samples}
        assert by_team == {"barkham": 2, "": 1}
    finally:
        human._pending_meta.clear()
