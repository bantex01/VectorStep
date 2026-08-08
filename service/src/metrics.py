"""Prometheus metrics derived from pipeline_runs / pipeline_steps.

All counters are computed from cumulative, all-time SQL aggregates rather than
incremented in-process — rows are never deleted, so the totals are monotonically
non-decreasing and safe to expose as Prometheus counters even though they're
recomputed on every scrape.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily
from prometheus_client.registry import Collector
from prometheus_client.utils import floatToGoString
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from . import live_pricing, pricing
from .analytics import get_team_month_to_date_spend
from .db.models import PipelineRun, PipelineStep, RunFeedback, StepFeedback
from .executors.human import list_pending as _list_pending_approvals

# Bucket upper bounds in seconds — spans a quick webhook call up to the default
# 1200s step timeout.
_DURATION_BUCKETS: tuple[float, ...] = (1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, float("inf"))


@dataclass
class MetricsData:
    run_counts: list[tuple[str, str, int]]
    runs_in_progress: int
    step_counts: list[tuple[str, str, str, str | None, str | None, str | None, str, int]]
    # (pipeline, step_name, executor, agent, model, provider, status, count) — pipeline/
    # step_name/model/provider let Grafana reconstruct the per-step and per-model
    # breakdowns the Steps/Agents/Pipelines Insights UI pages compute from the DB directly.
    step_durations: list[tuple[str, str, str, str | None, str | None, str | None, float]]
    # (pipeline, step_name, executor, agent, model, provider, seconds)
    verifier_counts: list[tuple[str | None, int, int]]
    token_usage: list[tuple[str | None, str, str, str, str | None, str | None, str | None, int, int]]
    # (team, pipeline, step_name, executor, agent, model, provider, input_tokens_sum, output_tokens_sum)
    cost_usage: list[tuple[str | None, str, str, str, str | None, str | None, str | None, float]]
    # (team, pipeline, step_name, executor, agent, model, provider, cost_sum) — unpriced (NULL-cost)
    # steps excluded, same as token_usage excludes steps with no token data.
    approx_cost_usage: list[tuple[str | None, str, str, str, str | None, str | None, str | None, float]]
    # Same shape as cost_usage, but for steps with NO real cost, fuzzy-matched against
    # OpenRouter's live catalog (SPEC-live-pricing.md) — empty unless pricing.live_pricing
    # is enabled. Always a SEPARATE metric from cost_usage, never blended into it.
    team_budget_ratios: list[tuple[str, float]]
    # (team, month_to_date_spend / pricing.team_budgets[team]) — only teams with a
    # configured budget get a row; advisory only, see get_team_month_to_date_spend.
    human_decisions: list[tuple[str | None, str, str, int]]
    # (team, pipeline, decision["approved"|"rejected"], count) — human steps only,
    # derived from primary_confidence (1.0 = approved, 0.0 = rejected — see
    # executors/human.py). Timeouts leave primary_confidence NULL and are excluded.
    feedback_counts: list[tuple[str, str, int]]
    # (pipeline, outcome["correct"|"partial"|"incorrect"], count) — from RunFeedback
    step_feedback_counts: list[tuple[str, str, str | None, str | None, str | None, str, int]]
    # (pipeline, step_name, agent, model, provider, outcome, count) — from StepFeedback
    grounding_scores: list[tuple[str, str, str | None, str | None, str | None, float]]
    # (pipeline, step_name, agent, model, provider, grounding_score) — from pipeline_steps, G non-null
    deterministic_check_counts: list[tuple[str, str, str, int]]
    # (pipeline, step_name, outcome["passed"|"failed"], count) — from pipeline_steps,
    # deterministic_passed IS NOT NULL
    runs_resumed: list[tuple[str, int]] = field(default_factory=list)
    # (pipeline, count) — from pipeline_runs, resumed_at IS NOT NULL (SPEC-durable-runs.md).
    # Defaulted (unlike every field above) so existing MetricsData(...) call sites in
    # tests that predate this field don't all need updating.


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
            select(PipelineRun.pipeline_name, func.count())
            .where(PipelineRun.resumed_at.is_not(None), PipelineRun.stage == "production")
            .group_by(PipelineRun.pipeline_name)
        )
        runs_resumed = rows.all()

        rows = await session.execute(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.executor, PipelineStep.agent, PipelineStep.model,
                PipelineStep.provider, PipelineStep.status, func.count(),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.stage == "production")
            .group_by(
                PipelineRun.pipeline_name, PipelineStep.step_name, PipelineStep.executor,
                PipelineStep.agent, PipelineStep.model, PipelineStep.provider, PipelineStep.status,
            )
        )
        step_counts = rows.all()

        rows = await session.execute(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.executor, PipelineStep.agent, PipelineStep.model,
                PipelineStep.provider, PipelineStep.duration_ms,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.duration_ms.is_not(None), PipelineRun.stage == "production")
        )
        step_durations = [
            (pipeline, step_name, executor, agent, model, provider, ms / 1000.0)
            for pipeline, step_name, executor, agent, model, provider, ms in rows.all()
        ]

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
                PipelineStep.step_name,
                PipelineStep.executor,
                PipelineStep.agent,
                PipelineStep.model,
                PipelineStep.provider,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None), PipelineRun.stage == "production")
            .group_by(
                PipelineRun.team, PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.executor, PipelineStep.agent, PipelineStep.model, PipelineStep.provider,
            )
        )
        token_usage = list(rows.all())

        # Only steps with a resolved cost contribute — unpriced (NULL) steps are
        # excluded rather than padding the metric with a spurious 0, same
        # reasoning as token_usage excluding no-usage steps.
        rows = await session.execute(
            select(
                PipelineRun.team,
                PipelineRun.pipeline_name,
                PipelineStep.step_name,
                PipelineStep.executor,
                PipelineStep.agent,
                PipelineStep.model,
                PipelineStep.provider,
                func.coalesce(func.sum(PipelineStep.cost), 0.0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.cost.is_not(None), PipelineRun.stage == "production")
            .group_by(
                PipelineRun.team, PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.executor, PipelineStep.agent, PipelineStep.model, PipelineStep.provider,
            )
        )
        cost_usage = list(rows.all())

        # Best-effort OpenRouter approximation for currently-unpriced steps
        # (SPEC-live-pricing.md) — a separate metric, never blended with cost_usage.
        # Fuzzy-matching can't be expressed in SQL, so this fetches raw rows and
        # aggregates in Python; only fetched when live_pricing is enabled, since
        # otherwise there's nothing to match against.
        approx_cost_usage: list[tuple] = []
        _pricing_table = pricing.get_table()
        if _pricing_table and _pricing_table.live_pricing.enabled:
            rows = await session.execute(
                select(
                    PipelineRun.team,
                    PipelineRun.pipeline_name,
                    PipelineStep.step_name,
                    PipelineStep.executor,
                    PipelineStep.agent,
                    PipelineStep.model,
                    PipelineStep.provider,
                    PipelineStep.input_tokens,
                    PipelineStep.output_tokens,
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.cost.is_(None), PipelineRun.stage == "production")
            )
            catalog = live_pricing.get_catalog()
            approx_sums: dict[tuple, float] = defaultdict(float)
            for team, pipeline, step_name, executor, agent, model, provider, in_tok, out_tok in rows.all():
                rate = live_pricing.resolve_approx_rate(catalog, provider, model)
                approx = live_pricing.approx_step_cost(rate, in_tok, out_tok)
                if approx is not None:
                    key = (team, pipeline, step_name, executor, agent, model, provider)
                    approx_sums[key] += approx
            approx_cost_usage = [(*key, total) for key, total in approx_sums.items()]

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

        rows = await session.execute(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.agent, PipelineStep.model, PipelineStep.provider,
                StepFeedback.outcome, func.count(),
            )
            .join(PipelineStep, StepFeedback.step_id == PipelineStep.id)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.stage == "production")
            .group_by(
                PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.agent, PipelineStep.model, PipelineStep.provider,
                StepFeedback.outcome,
            )
        )
        step_feedback_counts = list(rows.all())

        rows = await session.execute(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.agent, PipelineStep.model, PipelineStep.provider,
                PipelineStep.grounding_score,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.grounding_score.is_not(None), PipelineRun.stage == "production")
        )
        grounding_scores = list(rows.all())

        outcome_case = case(
            (PipelineStep.deterministic_passed.is_(True), "passed"),
            else_="failed",
        )
        rows = await session.execute(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name,
                outcome_case.label("outcome"), func.count(),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.deterministic_passed.is_not(None), PipelineRun.stage == "production")
            .group_by(PipelineRun.pipeline_name, PipelineStep.step_name, outcome_case)
        )
        deterministic_check_counts = list(rows.all())

    team_spend = await get_team_month_to_date_spend(session_factory)
    team_budget_ratios = [
        (team, row["ratio"]) for team, row in team_spend.items() if row["ratio"] is not None
    ]

    return MetricsData(
        run_counts=list(run_counts),
        runs_in_progress=runs_in_progress or 0,
        runs_resumed=list(runs_resumed),
        step_counts=list(step_counts),
        step_durations=step_durations,
        verifier_counts=list(verifier_counts),
        token_usage=token_usage,
        cost_usage=cost_usage,
        approx_cost_usage=approx_cost_usage,
        team_budget_ratios=team_budget_ratios,
        human_decisions=human_decisions,
        feedback_counts=list(feedback_counts),
        step_feedback_counts=step_feedback_counts,
        grounding_scores=grounding_scores,
        deterministic_check_counts=deterministic_check_counts,
    )


class VectorStepCollector(Collector):
    """One-shot collector populated with data fetched just before a scrape."""

    def __init__(self, data: MetricsData):
        self._data = data

    def collect(self):
        data = self._data

        runs_total = CounterMetricFamily(
            "vectorstep_pipeline_runs_total",
            "Total pipeline runs by pipeline and terminal status",
            labels=["pipeline", "status"],
        )
        for pipeline, status, count in data.run_counts:
            runs_total.add_metric([pipeline, status], count)
        yield runs_total

        yield GaugeMetricFamily(
            "vectorstep_pipeline_runs_in_progress",
            "Pipeline runs currently in status=running",
            value=data.runs_in_progress,
        )

        runs_resumed = CounterMetricFamily(
            "vectorstep_runs_resumed_total",
            "Total runs resumed after a restart (SPEC-durable-runs.md), by pipeline",
            labels=["pipeline"],
        )
        for pipeline, count in data.runs_resumed:
            runs_resumed.add_metric([pipeline], count)
        yield runs_resumed

        steps_total = CounterMetricFamily(
            "vectorstep_pipeline_steps_total",
            "Total pipeline steps by pipeline, step, executor, agent, model, provider, and status",
            labels=["pipeline", "step_name", "executor", "agent", "model", "provider", "status"],
        )
        for pipeline, step_name, executor, agent, model, provider, status, count in data.step_counts:
            steps_total.add_metric(
                [pipeline, step_name, executor, agent or "", model or "", provider or "", status], count,
            )
        yield steps_total

        yield self._duration_histogram(data.step_durations)

        yield self._grounding_histogram(data.grounding_scores)

        verifier_runs = CounterMetricFamily(
            "vectorstep_verifier_runs_total",
            "Total steps where a verifier ran, by primary agent",
            labels=["agent"],
        )
        verifier_overrides = CounterMetricFamily(
            "vectorstep_verifier_overrides_total",
            "Total verifier runs where the verifier lowered the primary's effective confidence",
            labels=["agent"],
        )
        for agent, runs, overrides in data.verifier_counts:
            verifier_runs.add_metric([agent or ""], runs)
            verifier_overrides.add_metric([agent or ""], overrides or 0)
        yield verifier_runs
        yield verifier_overrides

        token_totals = CounterMetricFamily(
            "vectorstep_pipeline_tokens_total",
            "Cumulative LLM tokens consumed, by team, pipeline, step, executor, agent, model, provider, and direction",
            labels=["team", "pipeline", "step_name", "executor", "agent", "model", "provider", "direction"],
        )
        for team, pipeline, step_name, executor, agent, model, provider, input_sum, output_sum in data.token_usage:
            base = [team or "", pipeline, step_name, executor, agent or "", model or "", provider or ""]
            token_totals.add_metric([*base, "input"], input_sum)
            token_totals.add_metric([*base, "output"], output_sum)
        yield token_totals

        cost_total = CounterMetricFamily(
            "vectorstep_pipeline_cost_total",
            f"Cumulative step cost in {pricing.get_table().currency if pricing.get_table() else 'USD'} "
            "(unpriced steps excluded), by pipeline, team, model, provider",
            labels=["pipeline", "team", "model", "provider"],
        )
        # cost_usage is grouped by step_name/executor/agent too (finer than this
        # metric's labels) — collapse those dimensions here rather than emitting
        # duplicate label-sets, which Prometheus exposition rejects outright.
        cost_by_label: dict[tuple[str, str, str, str], float] = defaultdict(float)
        for team, pipeline, step_name, executor, agent, model, provider, cost_sum in data.cost_usage:
            cost_by_label[(pipeline, team or "", model or "", provider or "")] += cost_sum
        for (pipeline, team, model, provider), cost_sum in cost_by_label.items():
            cost_total.add_metric([pipeline, team, model, provider], cost_sum)
        yield cost_total

        approx_cost_total = CounterMetricFamily(
            "vectorstep_pipeline_approx_cost_total",
            f"Best-effort OpenRouter reference cost in {pricing.get_table().currency if pricing.get_table() else 'USD'} "
            "for steps with no real (manual) price — APPROXIMATE, never blended with "
            "vectorstep_pipeline_cost_total; empty unless pricing.live_pricing is enabled. "
            "By pipeline, team, model, provider.",
            labels=["pipeline", "team", "model", "provider"],
        )
        approx_cost_by_label: dict[tuple[str, str, str, str], float] = defaultdict(float)
        for team, pipeline, step_name, executor, agent, model, provider, cost_sum in data.approx_cost_usage:
            approx_cost_by_label[(pipeline, team or "", model or "", provider or "")] += cost_sum
        for (pipeline, team, model, provider), cost_sum in approx_cost_by_label.items():
            approx_cost_total.add_metric([pipeline, team, model, provider], cost_sum)
        yield approx_cost_total

        budget_ratio = GaugeMetricFamily(
            "vectorstep_team_budget_ratio",
            "Month-to-date spend / pricing.team_budgets for the team, UTC calendar month — "
            "advisory only, not enforced. Only teams with a configured budget appear.",
            labels=["team"],
        )
        for team, ratio in data.team_budget_ratios:
            budget_ratio.add_metric([team], ratio)
        yield budget_ratio

        decisions_total = CounterMetricFamily(
            "vectorstep_human_approval_decisions_total",
            "Total human-in-the-loop approve/reject decisions, by team, pipeline, and decision",
            labels=["team", "pipeline", "decision"],
        )
        for team, pipeline, decision, count in data.human_decisions:
            decisions_total.add_metric([team or "", pipeline, decision], count)
        yield decisions_total

        feedback_total = CounterMetricFamily(
            "vectorstep_pipeline_feedback_total",
            "Total human accuracy feedback submissions, by pipeline and outcome",
            labels=["pipeline", "outcome"],
        )
        for pipeline, outcome, count in data.feedback_counts:
            feedback_total.add_metric([pipeline, outcome], count)
        yield feedback_total

        step_feedback_total = CounterMetricFamily(
            "vectorstep_step_feedback_total",
            "Total human accuracy feedback submissions per step, by pipeline, step, agent, model, provider, and outcome",
            labels=["pipeline", "step_name", "agent", "model", "provider", "outcome"],
        )
        for pipeline, step_name, agent, model, provider, outcome, count in data.step_feedback_counts:
            step_feedback_total.add_metric(
                [pipeline, step_name, agent or "", model or "", provider or "", outcome], count
            )
        yield step_feedback_total

        deterministic_total = CounterMetricFamily(
            "vectorstep_step_deterministic_check_total",
            "Total deterministic-check step outcomes, by pipeline, step, and outcome",
            labels=["pipeline", "step_name", "outcome"],
        )
        for pipeline, step_name, outcome, count in data.deterministic_check_counts:
            deterministic_total.add_metric([pipeline, step_name, outcome], count)
        yield deterministic_total

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
            "vectorstep_human_approvals_pending",
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
    def _duration_histogram(
        durations: list[tuple[str, str, str, str | None, str | None, str | None, float]],
    ) -> HistogramMetricFamily:
        grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
        for pipeline, step_name, executor, agent, model, provider, seconds in durations:
            grouped[(pipeline, step_name, executor, agent or "", model or "", provider or "")].append(seconds)

        hist = HistogramMetricFamily(
            "vectorstep_pipeline_step_duration_seconds",
            "Pipeline step execution duration in seconds, by pipeline, step, executor, agent, model, and provider",
            labels=["pipeline", "step_name", "executor", "agent", "model", "provider"],
        )
        for (pipeline, step_name, executor, agent, model, provider), values in grouped.items():
            buckets = [
                (floatToGoString(le), sum(1 for v in values if v <= le))
                for le in _DURATION_BUCKETS
            ]
            hist.add_metric(
                [pipeline, step_name, executor, agent, model, provider], buckets, sum_value=sum(values),
            )
        return hist

    @staticmethod
    def _grounding_histogram(
        scores: list[tuple[str, str, str | None, str | None, str | None, float]],
    ) -> HistogramMetricFamily:
        grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
        for pipeline, step_name, agent, model, provider, g in scores:
            grouped[(pipeline, step_name, agent or "", model or "", provider or "")].append(g)
        _BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf"))
        hist = HistogramMetricFamily(
            "vectorstep_step_grounding_score",
            "Shadow-mode grounding score (G) distribution per step, by pipeline, step, agent, model, provider",
            labels=["pipeline", "step_name", "agent", "model", "provider"],
        )
        for (pipeline, step_name, agent, model, provider), values in grouped.items():
            buckets = [(floatToGoString(le), sum(1 for v in values if v <= le)) for le in _BUCKETS]
            hist.add_metric([pipeline, step_name, agent, model, provider], buckets, sum_value=sum(values))
        return hist
