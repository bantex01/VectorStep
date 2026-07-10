"""Prometheus metrics derived from pipeline_runs / pipeline_steps.

All counters are computed from cumulative, all-time SQL aggregates rather than
incremented in-process — rows are never deleted, so the totals are monotonically
non-decreasing and safe to expose as Prometheus counters even though they're
recomputed on every scrape.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily
from prometheus_client.registry import Collector
from prometheus_client.utils import floatToGoString
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from .db.models import PipelineRun, PipelineStep, RunFeedback
from .executors.human import list_pending as _list_pending_approvals

# Bucket upper bounds in seconds — spans a quick webhook call up to the default
# 1200s step timeout.
_DURATION_BUCKETS: tuple[float, ...] = (1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, float("inf"))


@dataclass
class MetricsData:
    run_counts: list[tuple[str, str, int]]
    runs_in_progress: int
    step_counts: list[tuple[str, str | None, str, int]]
    step_durations: list[tuple[str, str | None, float]]
    verifier_counts: list[tuple[str | None, int, int]]
    token_usage: list[tuple[str | None, str, str, str | None, str | None, int, int]]
    # (team, pipeline, executor, agent, model, input_tokens_sum, output_tokens_sum)
    human_decisions: list[tuple[str | None, str, str, int]]
    # (team, pipeline, decision["approved"|"rejected"], count) — human steps only,
    # derived from primary_confidence (1.0 = approved, 0.0 = rejected — see
    # executors/human.py). Timeouts leave primary_confidence NULL and are excluded.
    feedback_counts: list[tuple[str, str, int]]
    # (pipeline, outcome["correct"|"partial"|"incorrect"], count) — from RunFeedback


async def fetch_metrics_data(session_factory: async_sessionmaker) -> MetricsData:
    # Every query is scoped to stage=production — a testing pipeline (default stage,
    # see PipelineConfig.stage) contributes nothing to any /metrics series. Queries
    # that don't otherwise touch pipeline_runs pick up a join purely for this filter.
    async with session_factory() as session:
        rows = await session.execute(
            select(PipelineRun.pipeline_name, PipelineRun.status, func.count())
            .where(PipelineRun.stage == "production")
            .group_by(PipelineRun.pipeline_name, PipelineRun.status)
        )
        run_counts = rows.all()

        runs_in_progress = await session.scalar(
            select(func.count()).where(
                PipelineRun.status == "running", PipelineRun.stage == "production"
            )
        )

        rows = await session.execute(
            select(PipelineStep.executor, PipelineStep.agent, PipelineStep.status, func.count())
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.stage == "production")
            .group_by(PipelineStep.executor, PipelineStep.agent, PipelineStep.status)
        )
        step_counts = rows.all()

        rows = await session.execute(
            select(PipelineStep.executor, PipelineStep.agent, PipelineStep.duration_ms)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.duration_ms.is_not(None), PipelineRun.stage == "production")
        )
        step_durations = [(executor, agent, ms / 1000.0) for executor, agent, ms in rows.all()]

        rows = await session.execute(
            select(
                PipelineStep.agent,
                func.count(),
                func.sum(case(
                    (PipelineStep.effective_confidence < PipelineStep.primary_confidence, 1),
                    else_=0,
                )),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.verifier_confidence.is_not(None), PipelineRun.stage == "production")
            .group_by(PipelineStep.agent)
        )
        verifier_counts = rows.all()

        # Only steps that actually report tokens (gateway executor) contribute —
        # openclaw/human/webhook steps leave input_tokens NULL and are excluded
        # rather than padding the metric with spurious zero-token series.
        rows = await session.execute(
            select(
                PipelineRun.team,
                PipelineRun.pipeline_name,
                PipelineStep.executor,
                PipelineStep.agent,
                PipelineStep.model,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None), PipelineRun.stage == "production")
            .group_by(
                PipelineRun.team, PipelineRun.pipeline_name,
                PipelineStep.executor, PipelineStep.agent, PipelineStep.model,
            )
        )
        token_usage = list(rows.all())

        decision_case = case(
            (PipelineStep.primary_confidence >= 0.5, "approved"),
            else_="rejected",
        )
        rows = await session.execute(
            select(
                PipelineRun.team,
                PipelineRun.pipeline_name,
                decision_case.label("decision"),
                func.count(),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(
                PipelineStep.executor == "human",
                PipelineStep.primary_confidence.is_not(None),
                PipelineRun.stage == "production",
            )
            .group_by(PipelineRun.team, PipelineRun.pipeline_name, decision_case)
        )
        human_decisions = list(rows.all())

        rows = await session.execute(
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count())
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(PipelineRun.stage == "production")
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        )
        feedback_counts = list(rows.all())

    return MetricsData(
        run_counts=list(run_counts),
        runs_in_progress=runs_in_progress or 0,
        step_counts=list(step_counts),
        step_durations=step_durations,
        verifier_counts=list(verifier_counts),
        token_usage=token_usage,
        human_decisions=human_decisions,
        feedback_counts=list(feedback_counts),
    )


class PorkCollector(Collector):
    """One-shot collector populated with data fetched just before a scrape."""

    def __init__(self, data: MetricsData):
        self._data = data

    def collect(self):
        data = self._data

        runs_total = CounterMetricFamily(
            "pork_pipeline_runs_total",
            "Total pipeline runs by pipeline and terminal status",
            labels=["pipeline", "status"],
        )
        for pipeline, status, count in data.run_counts:
            runs_total.add_metric([pipeline, status], count)
        yield runs_total

        yield GaugeMetricFamily(
            "pork_pipeline_runs_in_progress",
            "Pipeline runs currently in status=running",
            value=data.runs_in_progress,
        )

        steps_total = CounterMetricFamily(
            "pork_pipeline_steps_total",
            "Total pipeline steps by executor, agent, and status",
            labels=["executor", "agent", "status"],
        )
        for executor, agent, status, count in data.step_counts:
            steps_total.add_metric([executor, agent or "", status], count)
        yield steps_total

        yield self._duration_histogram(data.step_durations)

        verifier_runs = CounterMetricFamily(
            "pork_verifier_runs_total",
            "Total steps where a verifier ran, by primary agent",
            labels=["agent"],
        )
        verifier_overrides = CounterMetricFamily(
            "pork_verifier_overrides_total",
            "Total verifier runs where the verifier lowered the primary's effective confidence",
            labels=["agent"],
        )
        for agent, runs, overrides in data.verifier_counts:
            verifier_runs.add_metric([agent or ""], runs)
            verifier_overrides.add_metric([agent or ""], overrides or 0)
        yield verifier_runs
        yield verifier_overrides

        token_totals = CounterMetricFamily(
            "pork_pipeline_tokens_total",
            "Cumulative LLM tokens consumed, by team, pipeline, executor, agent, model, and direction",
            labels=["team", "pipeline", "executor", "agent", "model", "direction"],
        )
        for team, pipeline, executor, agent, model, input_sum, output_sum in data.token_usage:
            base = [team or "", pipeline, executor, agent or "", model or ""]
            token_totals.add_metric([*base, "input"], input_sum)
            token_totals.add_metric([*base, "output"], output_sum)
        yield token_totals

        decisions_total = CounterMetricFamily(
            "pork_human_approval_decisions_total",
            "Total human-in-the-loop approve/reject decisions, by team, pipeline, and decision",
            labels=["team", "pipeline", "decision"],
        )
        for team, pipeline, decision, count in data.human_decisions:
            decisions_total.add_metric([team or "", pipeline, decision], count)
        yield decisions_total

        feedback_total = CounterMetricFamily(
            "pork_pipeline_feedback_total",
            "Total human accuracy feedback submissions, by pipeline and outcome",
            labels=["pipeline", "outcome"],
        )
        for pipeline, outcome, count in data.feedback_counts:
            feedback_total.add_metric([pipeline, outcome], count)
        yield feedback_total

        yield self._pending_approvals_gauge()

    @staticmethod
    def _pending_approvals_gauge() -> GaugeMetricFamily:
        """Pending human approvals aren't persisted (in-memory only — see
        executors/human.py), so unlike everything else in this collector this reads
        live process state directly rather than the pre-fetched MetricsData.
        Testing-stage approvals are excluded, same as every other series here."""
        pending_by_team: dict[str, int] = defaultdict(int)
        for item in _list_pending_approvals():
            if item.get("stage") == "testing":
                continue
            pending_by_team[item.get("team") or ""] += 1

        gauge = GaugeMetricFamily(
            "pork_human_approvals_pending",
            "Currently pending human-in-the-loop approvals, by team",
            labels=["team"],
        )
        if pending_by_team:
            for team, count in pending_by_team.items():
                gauge.add_metric([team], count)
        else:
            gauge.add_metric([""], 0)
        return gauge

    @staticmethod
    def _duration_histogram(durations: list[tuple[str, str | None, float]]) -> HistogramMetricFamily:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for executor, agent, seconds in durations:
            grouped[(executor, agent or "")].append(seconds)

        hist = HistogramMetricFamily(
            "pork_pipeline_step_duration_seconds",
            "Pipeline step execution duration in seconds",
            labels=["executor", "agent"],
        )
        for (executor, agent), values in grouped.items():
            buckets = [
                (floatToGoString(le), sum(1 for v in values if v <= le))
                for le in _DURATION_BUCKETS
            ]
            hist.add_metric([executor, agent], buckets, sum_value=sum(values))
        return hist
