"""Shared rollup/aggregation queries for pipeline and step operational stats and
judged (human feedback) accuracy.

This is the single source of truth for every number surfaced by both the
`/ui/insights/*` HTML pages (ui.py) and the JSON `/stats` endpoints (main.py) —
see SPEC-pork-service-mcp.md §6. Extracting the queries here means the two
surfaces can't drift: they call the same functions.

Two distinct notions of "accuracy" (see SPEC-pork-service-mcp.md §3):
  - operational outcome: derived from `PipelineRun.status` / `PipelineStep.status`
    and token/duration columns — "how many failed, how long did it take".
  - judged accuracy: derived from `RunFeedback.outcome` / `StepFeedback.outcome`,
    a human grading a run/step as correct/partial/incorrect.
Every rollup here is production-scoped by default and accepts an explicit
`stage` to widen or narrow that (see `_scope_stage`).
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from .db.models import PipelineRun, PipelineStep, RunFeedback, StepFeedback
from .utils import utc_now

# Every value ever assigned to PipelineRun.status. "deduplicated" is deliberately
# NOT included here even though it's a status a webhook caller can see in the
# /webhook response: that path (main.py:_trigger_run, on a duplicate match)
# returns the *existing* duplicate run's status and never inserts a new row, so
# "deduplicated" never actually lands in the `pipeline_runs.status` column.
# "running" / "completed" / "failed" / "aborted" / "escalated" / "stopped" are
# set throughout pipeline/runner.py; "interrupted" is set by
# db/database.py:mark_interrupted_runs when a run is found still "running" after
# a crash/restart.
ALL_RUN_STATUSES = ("completed", "failed", "aborted", "escalated", "stopped", "running", "interrupted")

# Every value ever assigned to PipelineStep.status — see pipeline/runner.py's
# StepResult.status Literal. Steps have no "running"/"interrupted" state: a step
# row is only written once its executor call has returned.
ALL_STEP_STATUSES = ("completed", "failed", "aborted", "escalated", "stopped")

_TIME_RANGES = {
    "24h": (timedelta(hours=24), "24 hours"),
    "7d": (timedelta(days=7), "7 days"),
    "30d": (timedelta(days=30), "30 days"),
}


def _time_range_cutoff(time_range: str) -> tuple[datetime | None, str]:
    """Map 24h|7d|30d|all to (cutoff_datetime_or_None, label)."""
    delta_label = _TIME_RANGES.get(time_range)
    if delta_label is None:
        return None, "all time"
    delta, label = delta_label
    return utc_now() - delta, label


def _scope_stage(q, stage: str):
    """Scope a PipelineRun-touching query by pipeline stage.

    stage="all" leaves the query unfiltered; "production" / "testing" filter
    to that stage. Every stats/insights rollup defaults to "production" —
    see `_production_only` below, kept as the pre-existing name ui.py's
    call sites already use.
    """
    if stage == "all":
        return q
    return q.where(PipelineRun.stage == stage)


def _production_only(q):
    """Scope a PipelineRun-touching query to stage=production — applied to every
    aggregate/rollup surface (dashboard cards, insights, success/accuracy bars,
    /stats endpoints). Browse/list surfaces (the runs list, recent-runs tables)
    stay unfiltered and show a stage badge instead."""
    return _scope_stage(q, "production")


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (numpy's default method) over an
    already-sorted list. Returns None for an empty input."""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (pct / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _empty_accuracy() -> dict:
    return {"correct": 0, "partial": 0, "incorrect": 0, "total": 0, "correct_pct": None}


def _accuracy_from_counts(correct: int, partial: int, incorrect: int) -> dict:
    total = correct + partial + incorrect
    return {
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "total": total,
        "correct_pct": round(correct / total * 100, 1) if total else None,
    }


# ── Pipeline rollups ──────────────────────────────────────────────────────────

async def _pipeline_rollup(
    session_factory: async_sessionmaker,
    time_range: str = "7d",
    stage: str = "production",
    pipeline_name: str | None = None,
) -> dict[str, dict]:
    """Core per-pipeline aggregation. Returns {pipeline_name: stats_dict} for
    every pipeline with at least one run in range, or just `pipeline_name` if
    given. Used by both get_pipeline_stats and list_pipeline_stats."""
    cutoff, _ = _time_range_cutoff(time_range)

    def _scope(q):
        q = _scope_stage(q, stage)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        if pipeline_name:
            q = q.where(PipelineRun.pipeline_name == pipeline_name)
        return q

    async with session_factory() as session:
        status_rows = (await session.execute(_scope(
            select(PipelineRun.pipeline_name, PipelineRun.status, func.count().label("n"))
            .group_by(PipelineRun.pipeline_name, PipelineRun.status)
        ))).all()

        token_rows = (await session.execute(_scope(
            select(
                PipelineRun.pipeline_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineRun.pipeline_name)
        ))).all()

        teams_rows = (await session.execute(_scope(
            select(PipelineRun.pipeline_name, PipelineRun.team).distinct()
        ))).all()

        duration_rows = (await session.execute(_scope(
            select(PipelineRun.pipeline_name, PipelineRun.triggered_at, PipelineRun.completed_at)
            .where(PipelineRun.completed_at.is_not(None))
        ))).all()

        feedback_rows = (await session.execute(_scope(
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        ))).all()

    status_counts: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in ALL_RUN_STATUSES})
    names: set[str] = set()
    for name, status, n in status_rows:
        names.add(name)
        status_counts[name][status] = status_counts[name].get(status, 0) + n

    token_totals = {name: (i, o) for name, i, o in token_rows}
    names.update(token_totals)

    teams_by_pipeline: dict[str, set[str]] = defaultdict(set)
    for name, team in teams_rows:
        teams_by_pipeline[name].add(team or "Unattributed")
    names.update(teams_by_pipeline)

    durations_by_pipeline: dict[str, list[float]] = defaultdict(list)
    for name, triggered_at, completed_at in duration_rows:
        secs = (completed_at.replace(tzinfo=None) - triggered_at.replace(tzinfo=None)).total_seconds()
        if secs >= 0:
            durations_by_pipeline[name].append(secs)
    names.update(durations_by_pipeline)

    feedback_by_pipeline: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0})
    for name, outcome, n in feedback_rows:
        feedback_by_pipeline[name][outcome] = n
    names.update(feedback_by_pipeline)

    result: dict[str, dict] = {}
    for name in names:
        sc = status_counts.get(name) or {s: 0 for s in ALL_RUN_STATUSES}
        runs_total = sum(sc.values())
        terminal = runs_total - sc.get("running", 0)
        success_rate = round(sc.get("completed", 0) / terminal, 4) if terminal else None

        inp, out = token_totals.get(name, (0, 0))

        secs = sorted(durations_by_pipeline.get(name, []))
        avg_duration = (sum(secs) / len(secs)) if secs else None
        p95_duration = _percentile(secs, 95)

        fb = feedback_by_pipeline.get(name, {})

        result[name] = {
            "pipeline_name": name,
            "time_range": time_range,
            "stage": stage,
            "runs_total": runs_total,
            "status_counts": dict(sc),
            "success_rate": success_rate,
            "tokens": {"input": inp, "output": out, "total": inp + out},
            "duration_seconds": {"avg": avg_duration, "p95": p95_duration},
            "accuracy": _accuracy_from_counts(fb.get("correct", 0), fb.get("partial", 0), fb.get("incorrect", 0)),
            "teams": sorted(teams_by_pipeline.get(name, set())),
        }
    return result


def _empty_pipeline_stats(pipeline_name: str, time_range: str, stage: str) -> dict:
    return {
        "pipeline_name": pipeline_name,
        "time_range": time_range,
        "stage": stage,
        "runs_total": 0,
        "status_counts": {s: 0 for s in ALL_RUN_STATUSES},
        "success_rate": None,
        "tokens": {"input": 0, "output": 0, "total": 0},
        "duration_seconds": {"avg": None, "p95": None},
        "accuracy": _empty_accuracy(),
        "teams": [],
    }


async def get_pipeline_stats(
    session_factory: async_sessionmaker,
    pipeline_name: str,
    time_range: str = "7d",
    stage: str = "production",
) -> dict:
    """Operational + judged-accuracy rollup for one pipeline over `time_range`.
    Does not check whether `pipeline_name` is a real/loaded pipeline — callers
    (the /pipelines/{name}/stats endpoint) 404 against the pipeline registry
    first. Returns an all-zero payload if the pipeline has no runs in range."""
    rollup = await _pipeline_rollup(session_factory, time_range, stage, pipeline_name=pipeline_name)
    return rollup.get(pipeline_name) or _empty_pipeline_stats(pipeline_name, time_range, stage)


async def list_pipeline_stats(
    session_factory: async_sessionmaker,
    time_range: str = "7d",
    stage: str = "production",
) -> list[dict]:
    """The same per-pipeline payload as get_pipeline_stats, for every pipeline
    that has at least one run in range — the JSON behind /ui/insights/pipelines.
    Sorted by runs_total descending, matching that page's table ordering."""
    rollup = await _pipeline_rollup(session_factory, time_range, stage)
    return sorted(rollup.values(), key=lambda r: r["runs_total"], reverse=True)


# ── Step rollups ──────────────────────────────────────────────────────────────

async def _step_rollup(
    session_factory: async_sessionmaker,
    time_range: str = "7d",
    stage: str = "production",
    step_name: str | None = None,
) -> dict[str, dict]:
    """Core per-library-step aggregation, keyed by PipelineStep.step_name.
    Mirrors _pipeline_rollup but scoped to steps — see get_step_stats."""
    cutoff, _ = _time_range_cutoff(time_range)

    def _scope(q):
        q = _scope_stage(q, stage)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        if step_name:
            q = q.where(PipelineStep.step_name == step_name)
        return q

    async with session_factory() as session:
        status_rows = (await session.execute(_scope(
            select(PipelineStep.step_name, PipelineStep.status, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineStep.step_name, PipelineStep.status)
        ))).all()

        token_rows = (await session.execute(_scope(
            select(
                PipelineStep.step_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineStep.step_name)
        ))).all()

        duration_rows = (await session.execute(_scope(
            select(PipelineStep.step_name, PipelineStep.duration_ms)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.duration_ms.is_not(None))
        ))).all()

        feedback_rows = (await session.execute(_scope(
            select(PipelineStep.step_name, StepFeedback.outcome, func.count().label("n"))
            .join(StepFeedback, StepFeedback.step_id == PipelineStep.id)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineStep.step_name, StepFeedback.outcome)
        ))).all()

    status_counts: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in ALL_STEP_STATUSES})
    names: set[str] = set()
    for name, status, n in status_rows:
        names.add(name)
        status_counts[name][status] = status_counts[name].get(status, 0) + n

    token_totals = {name: (i, o) for name, i, o in token_rows}
    names.update(token_totals)

    durations_by_step: dict[str, list[float]] = defaultdict(list)
    for name, duration_ms in duration_rows:
        durations_by_step[name].append(duration_ms / 1000)
    names.update(durations_by_step)

    feedback_by_step: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0})
    for name, outcome, n in feedback_rows:
        feedback_by_step[name][outcome] = n
    names.update(feedback_by_step)

    result: dict[str, dict] = {}
    for name in names:
        sc = status_counts.get(name) or {s: 0 for s in ALL_STEP_STATUSES}
        runs_total = sum(sc.values())
        success_rate = round(sc.get("completed", 0) / runs_total, 4) if runs_total else None

        inp, out = token_totals.get(name, (0, 0))

        secs = sorted(durations_by_step.get(name, []))
        avg_duration = (sum(secs) / len(secs)) if secs else None
        p95_duration = _percentile(secs, 95)

        fb = feedback_by_step.get(name, {})

        result[name] = {
            "step_name": name,
            "time_range": time_range,
            "stage": stage,
            "runs_total": runs_total,
            "status_counts": dict(sc),
            "success_rate": success_rate,
            "tokens": {"input": inp, "output": out, "total": inp + out},
            "avg_input_tokens": round(inp / runs_total) if runs_total else None,
            "avg_output_tokens": round(out / runs_total) if runs_total else None,
            "duration_seconds": {"avg": avg_duration, "p95": p95_duration},
            "accuracy": _accuracy_from_counts(fb.get("correct", 0), fb.get("partial", 0), fb.get("incorrect", 0)),
        }
    return result


def _empty_step_stats(step_name: str, time_range: str, stage: str) -> dict:
    return {
        "step_name": step_name,
        "time_range": time_range,
        "stage": stage,
        "runs_total": 0,
        "status_counts": {s: 0 for s in ALL_STEP_STATUSES},
        "success_rate": None,
        "tokens": {"input": 0, "output": 0, "total": 0},
        "avg_input_tokens": None,
        "avg_output_tokens": None,
        "duration_seconds": {"avg": None, "p95": None},
        "accuracy": _empty_accuracy(),
    }


async def get_step_stats(
    session_factory: async_sessionmaker,
    step_name: str,
    time_range: str = "7d",
    stage: str = "production",
) -> dict:
    """Operational + judged-accuracy rollup for one library step over
    `time_range`. Does not check the step library — the endpoint 404s against
    it first. Returns an all-zero payload if the step has no executions in range."""
    rollup = await _step_rollup(session_factory, time_range, stage, step_name=step_name)
    return rollup.get(step_name) or _empty_step_stats(step_name, time_range, stage)
