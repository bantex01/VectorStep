"""Tests for the pork_pipeline_tokens_total metric (team/pipeline/executor/agent/model/direction)."""
from datetime import datetime

from src.db.database import create_tables, init_db, get_session_factory
from src.db.models import PipelineRun, PipelineStep
from src.metrics import MetricsData, PorkCollector, fetch_metrics_data


def _empty_metrics_data(token_usage: list[tuple]) -> MetricsData:
    return MetricsData(
        run_counts=[],
        runs_in_progress=0,
        step_counts=[],
        step_durations=[],
        verifier_counts=[],
        token_usage=token_usage,
    )


def _find_family(families, sample_name):
    return next(f for f in families if any(s.name == sample_name for s in f.samples))


def test_collect_emits_pork_pipeline_tokens_total_with_expected_labels():
    data = _empty_metrics_data([
        ("payments", "alert-triage", "gateway", "sre-triage", "anthropic/claude-sonnet-4-6", 100, 50),
    ])
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_pipeline_tokens_total")

    assert family.samples[0].labels == {
        "team": "payments", "pipeline": "alert-triage", "executor": "gateway",
        "agent": "sre-triage", "model": "anthropic/claude-sonnet-4-6", "direction": "input",
    }
    assert family.samples[0].value == 100
    assert family.samples[1].labels["direction"] == "output"
    assert family.samples[1].value == 50


def test_collect_buckets_null_team_and_model_as_empty_string():
    data = _empty_metrics_data([
        (None, "scheduled-pipeline", "gateway", "agent-x", None, 10, 5),
    ])
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_pipeline_tokens_total")

    assert family.samples[0].labels["team"] == ""
    assert family.samples[0].labels["model"] == ""


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
    team, pipeline, executor, agent, model, input_sum, output_sum = metrics_data.token_usage[0]
    assert (team, pipeline, executor, agent, model) == ("payments", "p", "gateway", "a", "m")
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
                id=f"step-{i}", run_id="run-1", step_name=f"s{i}", step_index=i,
                executor="gateway", agent="a", model="m", prompt="", status="completed",
                executed_at=datetime(2026, 1, 1), input_tokens=10, output_tokens=5,
            ))
        await session.commit()

    metrics_data = await fetch_metrics_data(session_factory)

    assert len(metrics_data.token_usage) == 1
    _, _, _, _, _, input_sum, output_sum = metrics_data.token_usage[0]
    assert (input_sum, output_sum) == (20, 10)
