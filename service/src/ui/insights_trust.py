from fastapi import APIRouter
from .. import pricing
from ..analytics import _production_only
from ..analytics import _step_rollup
from ..analytics import _time_range_cutoff
from ..db.database import get_session_factory
from ..db.models import PipelineRun
from ..db.models import PipelineStep
from ..db.models import StepFeedback
from ..utils import utc_now
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from sqlalchemy import select

from . import helpers
from .helpers import templates
from . import insights


router = APIRouter()


# --- lines 1964-1974 ---
def _buckets_matching(calibration_buckets: dict, step_name, agent, model, provider) -> list:
    """Every calibration bucket matching the first four key components, regardless
    of prompt_hash/agent_version. See SPEC-prompt-versioning.md §4g — the bucket key
    grew to 6 components, but several call sites (like the ones below) only have the
    original 4 to key off of."""
    return [
        b for (s, a, m, p, _ph, _av), b in calibration_buckets.items()
        if (s, a, m, p) == (step_name, agent, model, provider)
    ]



# --- lines 1975-1984 ---
def _largest_bucket_matching(calibration_buckets: dict, step_name, agent, model, provider):
    """The matching bucket with the most samples — used for the Calibration bins
    display, so a brand-new (tiny) version doesn't blank out a rich, informative
    history the moment a prompt or agent changes."""
    matches = _buckets_matching(calibration_buckets, step_name, agent, model, provider)
    if not matches:
        return None
    return max(matches, key=lambda b: b.total_n)



# --- lines 1985-1998 ---
def _most_recent_bucket_matching(calibration_buckets: dict, step_name, agent, model, provider):
    """The matching bucket most recently added to — used for the Version chips.
    Deliberately NOT the same choice as _largest_bucket_matching: right after a
    prompt/agent edit, the legacy or previous-version bucket is almost always
    still the *largest* (it's had longer to accumulate), but the Version column's
    whole job is to say what's CURRENT, and showing the largest bucket's hash
    there would keep pointing at a stale version for a long time. A bucket with
    no last_seen_at (shouldn't happen post-migration, but defensive) sorts last."""
    matches = _buckets_matching(calibration_buckets, step_name, agent, model, provider)
    if not matches:
        return None
    return max(matches, key=lambda b: b.last_seen_at or datetime.min)



# --- lines 1999-2253 ---
@router.get("/insights/steps", response_class=HTMLResponse)
async def ui_insights_steps(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    # Per-step run/token/duration rollup — shared with the /steps/{name}/stats
    # JSON endpoint (analytics.py) so this page and that endpoint can never
    # disagree.
    rollup = await _step_rollup(sf, time_range, "production")
    run_counts = {name: r["runs_total"] for name, r in rollup.items()}
    failed_counts = {name: r["status_counts"].get("failed", 0) for name, r in rollup.items()}
    token_totals = {name: (r["tokens"]["input"], r["tokens"]["output"]) for name, r in rollup.items()}
    avg_duration_by_step = {
        name: r["duration_seconds"]["avg"] for name, r in rollup.items()
        if r["duration_seconds"]["avg"] is not None
    }
    status_counts_by_step: dict[str, dict[str, int]] = {name: dict(r["status_counts"]) for name, r in rollup.items()}

    # Per-step-execution rows and the finer (step, agent, model, provider) feedback
    # breakdown below stay as page-specific queries — granularity the shared
    # rollup (aggregated to step_name only) doesn't expose.
    async with sf() as session:
        # All step executions in range — used for per-step timeseries and recent-list
        q = _production_only(
            select(
                PipelineStep.step_name, PipelineRun.pipeline_name, PipelineStep.run_id,
                PipelineStep.status, PipelineStep.executed_at, PipelineStep.duration_ms,
                PipelineStep.input_tokens, PipelineStep.output_tokens,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_steps_raw = rows.all()

        q = _production_only(
            select(
                PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.model, PipelineStep.provider,
                StepFeedback.outcome, func.count().label("n"),
            )
            .join(PipelineStep, StepFeedback.step_id == PipelineStep.id)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(
                PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.model, PipelineStep.provider, StepFeedback.outcome,
            )
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        step_feedback_rows = (await session.execute(q)).all()

    _all_step_durations_secs = [r.duration_ms / 1000 for r in all_steps_raw if r.duration_ms is not None]
    overall_avg_duration_secs = (
        sum(_all_step_durations_secs) / len(_all_step_durations_secs) if _all_step_durations_secs else None
    )

    # step_combo: (pipeline_name, step_name, agent, qualified_model) -> counters, shared
    # with the Pipelines Insights drilldown (see insights._fetch_step_agent_model_combo).
    step_combo = await insights._fetch_step_agent_model_combo(cutoff)

    from ..pipeline.calibration import calibration_recommendation, compute_calibration_buckets

    calibration_buckets = await compute_calibration_buckets(sf)

    feedback_by_step: dict[str, dict] = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0})
    feedback_by_combo: dict[tuple, dict] = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0})
    for step_name, agent, model, provider, outcome, n in step_feedback_rows:
        feedback_by_step[step_name][outcome] += n
        feedback_by_combo[(step_name, agent, helpers._qualified_model(provider, model))][outcome] += n

    def _acc(d: dict) -> dict:
        total = d["correct"] + d["partial"] + d["incorrect"]
        return {**d, "total": total,
                "accuracy_pct": round(d["correct"] / total * 100) if total else None}

    feedback_by_step = {k: _acc(v) for k, v in feedback_by_step.items()}
    feedback_by_combo = {k: _acc(v) for k, v in feedback_by_combo.items()}

    # ── Per-step aggregates ────────────────────────────────────────────────────

    now = utc_now()
    resolution = helpers._ts_resolution(time_range)
    if all_steps_raw:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_steps_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = helpers._ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_step: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_step: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_step: dict[str, list] = defaultdict(list)

    for row in all_steps_raw:
        bucket = helpers._ts_bucket(row.executed_at, resolution)
        runs_by_bucket_step[row.step_name][bucket] += 1
        if row.duration_ms is not None:
            durations_by_bucket_step[row.step_name][bucket].append(row.duration_ms / 1000)
        if len(recent_by_step[row.step_name]) < 5:
            recent_by_step[row.step_name].append(row)

    # ── Per-step pipeline/agent/model breakdown ────────────────────────────────
    # Re-aggregates the shared step_combo indexed by step instead of by pipeline (compare
    # ui_insights_pipelines's step_breakdown_by_pipeline, built from the same combo).
    pipeline_breakdown_by_step: dict[str, list[dict]] = defaultdict(list)
    distinct_pipelines_by_step: dict[str, set[str]] = defaultdict(set)
    for (_team, pipeline_name, step_name, agent, provider, model), c in step_combo.items():
        distinct_pipelines_by_step[step_name].add(pipeline_name)
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        fb = feedback_by_combo.get((step_name, agent, helpers._qualified_model(provider, model)))
        # step_combo isn't scoped by prompt_hash/agent_version, but calibration_buckets
        # now is (SPEC-prompt-versioning.md §4g) — one (step, agent, model, provider)
        # combo can match several buckets (one per prompt/agent version). The
        # Calibration column shows the LARGEST bucket (most informative — a brand-new
        # version's near-empty bins shouldn't blank out a rich history), but the
        # Version column shows the MOST RECENT bucket's hash (what's actually current
        # right now) — using the largest bucket for both would keep pointing the
        # Version chip at a stale prompt for a long time after every edit. §6a's
        # prompt-history view is where every version gets its own row regardless.
        bins_bucket = _largest_bucket_matching(calibration_buckets, step_name, agent, model, provider)
        version_bucket = _most_recent_bucket_matching(calibration_buckets, step_name, agent, model, provider)
        calibration_summary = calibration_recommendation(bins_bucket) if bins_bucket else None
        pipeline_breakdown_by_step[step_name].append({
            "pipeline_name": pipeline_name,
            "agent": agent,
            "model": helpers._qualified_model(provider, model),
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": helpers._format_seconds(avg_duration_secs),
            "accuracy_pct": fb["accuracy_pct"] if fb else None,
            "marked_total": fb["total"] if fb else 0,
            "calibration_bins": [
                {"lo": b.lo, "hi": b.hi, "n": b.n, "mean_label": b.mean_label, "validated": b.validated}
                for b in bins_bucket.bins
            ] if bins_bucket else [],
            "calibration_recommendation": calibration_summary,
            "prompt_hash": version_bucket.prompt_hash if version_bucket else None,
            "agent_version": version_bucket.agent_version if version_bucket else None,
        })
    for rows_ in pipeline_breakdown_by_step.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────

    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_step.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_step.get(name)
        inp, out = token_totals.get(name, (0, 0))

        runs_ts_data = [runs_by_bucket_step[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_step[name][b]) / len(durations_by_bucket_step[name][b]))
            if durations_by_bucket_step[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_step.get(name, []):
            recent.append({
                "id": str(r.run_id),
                "pipeline_name": r.pipeline_name,
                "status": r.status,
                "ago": helpers._format_ago(r.executed_at),
                "duration": helpers._format_seconds(r.duration_ms / 1000) if r.duration_ms is not None else None,
            })

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": helpers._format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "pipeline_breakdown": pipeline_breakdown_by_step.get(name, []),
            "accuracy_pct": feedback_by_step.get(name, {}).get("accuracy_pct"),
            "marked_total": feedback_by_step.get(name, {}).get("total", 0),
            "feedback_breakdown": {
                k: feedback_by_step.get(name, {}).get(k, 0)
                for k in ("correct", "partial", "incorrect")
            },
        }

    # ── Headline stats ────────────────────────────────────────────────────────

    total_step_executions = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_step_executions - total_failed) / total_step_executions * 100) if total_step_executions else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())

    # ── Chart data ────────────────────────────────────────────────────────────

    step_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        step_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_step.get(name),
            "input_tokens": token_totals.get(name, (0, 0))[0],
            "output_tokens": token_totals.get(name, (0, 0))[1],
            "pipelines": sorted(distinct_pipelines_by_step.get(name, [])),
        })

    duration_chart_rows = sorted(avg_duration_by_step.items(), key=lambda t: t[1], reverse=True)
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    steps_ts = helpers._build_ts(
        [(r.executed_at, r.step_name) for r in all_steps_raw], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = helpers._build_ts(
        [(r.executed_at, r.step_name, (r.input_tokens or 0) + (r.output_tokens or 0)) for r in all_steps_raw],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_steps.html", {
        "time_range": time_range,
        "range_label": range_label,
        "step_rows": step_rows,
        "duration_chart": duration_chart,
        "steps_ts": steps_ts,
        "tokens_ts": tokens_ts,
        "total_step_count": len(run_counts),
        "total_step_executions": total_step_executions,
        "overall_success_rate": overall_success_rate,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "drilldown_data": drilldown_data,
        "active_page": "insights_steps",
    })



# --- lines 2254-2506 ---
@router.get("/insights/agents", response_class=HTMLResponse)
async def ui_insights_agents(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    # Every query below is a rollup for this insights page — scoped to production.
    # PipelineStep.agent already carries the executor prefix (f"{executor}:{agent}" —
    # see runner.py), so it alone is a stable, unique key without also grouping by executor.
    async with sf() as session:
        q = _production_only(
            select(PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        run_counts = dict(rows.all())

        q = _production_only(
            select(PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None), PipelineStep.status == "failed")
            .group_by(PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        failed_counts = dict(rows.all())

        q = _production_only(
            select(
                PipelineStep.agent,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        token_totals = {name: (i, o) for name, i, o in rows.all()}

        q = _production_only(
            select(PipelineStep.agent, func.avg(PipelineStep.duration_ms))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        avg_duration_by_agent = {name: ms / 1000 for name, ms in rows.all() if ms is not None}

        q = _production_only(
            select(func.avg(PipelineStep.duration_ms))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        overall_avg_duration_ms = (await session.execute(q)).scalar()
        overall_avg_duration_secs = overall_avg_duration_ms / 1000 if overall_avg_duration_ms is not None else None

        q = _production_only(
            select(PipelineStep.agent, PipelineStep.status, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent, PipelineStep.status)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        status_counts_by_agent: dict[str, dict[str, int]] = defaultdict(dict)
        for name, status, n in rows.all():
            status_counts_by_agent[name][status] = n

        # All steps in range — used for per-agent timeseries and recent-list
        q = _production_only(
            select(
                PipelineStep.agent, PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.run_id, PipelineStep.status, PipelineStep.executed_at,
                PipelineStep.duration_ms, PipelineStep.input_tokens, PipelineStep.output_tokens,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_steps_raw = rows.all()

    # step_combo: (pipeline_name, step_name, agent, qualified_model) -> counters, shared
    # with the Pipelines/Steps Insights drilldowns (see insights._fetch_step_agent_model_combo).
    step_combo = await insights._fetch_step_agent_model_combo(cutoff)

    # ── Per-agent aggregates ──────────────────────────────────────────────────

    now = utc_now()
    resolution = helpers._ts_resolution(time_range)
    if all_steps_raw:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_steps_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = helpers._ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_agent: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_agent: dict[str, list] = defaultdict(list)

    for row in all_steps_raw:
        bucket = helpers._ts_bucket(row.executed_at, resolution)
        runs_by_bucket_agent[row.agent][bucket] += 1
        if row.duration_ms is not None:
            durations_by_bucket_agent[row.agent][bucket].append(row.duration_ms / 1000)
        if len(recent_by_agent[row.agent]) < 5:
            recent_by_agent[row.agent].append(row)

    # ── Per-agent step/pipeline/model breakdown ────────────────────────────────
    # Re-aggregates the shared step_combo indexed by agent instead of by pipeline/step
    # (compare ui_insights_pipelines / ui_insights_steps, built from the same combo).
    step_breakdown_by_agent: dict[str, list[dict]] = defaultdict(list)
    models_by_agent: dict[str, set[str]] = defaultdict(set)
    for (_team, pipeline_name, step_name, agent, provider, model), c in step_combo.items():
        if not agent:
            continue
        qualified_model = helpers._qualified_model(provider, model)
        models_by_agent[agent].add(qualified_model)
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        step_breakdown_by_agent[agent].append({
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "model": qualified_model,
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": helpers._format_seconds(avg_duration_secs),
        })
    for rows_ in step_breakdown_by_agent.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────

    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_agent.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_agent.get(name)
        inp, out = token_totals.get(name, (0, 0))

        runs_ts_data = [runs_by_bucket_agent[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_agent[name][b]) / len(durations_by_bucket_agent[name][b]))
            if durations_by_bucket_agent[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_agent.get(name, []):
            recent.append({
                "id": str(r.run_id),
                "pipeline_name": r.pipeline_name,
                "step_name": r.step_name,
                "status": r.status,
                "ago": helpers._format_ago(r.executed_at),
                "duration": helpers._format_seconds(r.duration_ms / 1000) if r.duration_ms is not None else None,
            })

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": helpers._format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "step_breakdown": step_breakdown_by_agent.get(name, []),
        }

    # ── Headline stats ────────────────────────────────────────────────────────

    total_step_executions = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_step_executions - total_failed) / total_step_executions * 100) if total_step_executions else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())

    # ── Chart data ────────────────────────────────────────────────────────────

    agent_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        executor, _, bare_name = name.partition(":")
        agent_rows.append({
            "name": name,
            "executor": executor,
            "bare_name": bare_name or name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_agent.get(name),
            "input_tokens": token_totals.get(name, (0, 0))[0],
            "output_tokens": token_totals.get(name, (0, 0))[1],
            "models": sorted(models_by_agent.get(name, [])),
        })

    duration_chart_rows = sorted(avg_duration_by_agent.items(), key=lambda t: t[1], reverse=True)
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    steps_ts = helpers._build_ts(
        [(r.executed_at, r.agent) for r in all_steps_raw], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = helpers._build_ts(
        [(r.executed_at, r.agent, (r.input_tokens or 0) + (r.output_tokens or 0)) for r in all_steps_raw],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_agents.html", {
        "time_range": time_range,
        "range_label": range_label,
        "agent_rows": agent_rows,
        "duration_chart": duration_chart,
        "steps_ts": steps_ts,
        "tokens_ts": tokens_ts,
        "total_agent_count": len(run_counts),
        "total_step_executions": total_step_executions,
        "overall_success_rate": overall_success_rate,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "drilldown_data": drilldown_data,
        "active_page": "insights_agents",
    })



# --- lines 2507-2730 ---
@router.get("/insights/providers", response_class=HTMLResponse)
async def ui_insights_providers(request: Request, time_range: str = "7d"):
    """Token/call spend grouped by LLM provider (gateway executor only). Unlike every
    other Insights page, this one falls back to a best-effort provider guess from the
    model string (see helpers._provider_from_model) when the `provider` column is NULL — this
    page's entire purpose is grouping by provider, so a best-effort bucket for
    pre-migration rows beats losing that history from the page entirely. Contrast with
    helpers._qualified_model (used for per-row display elsewhere), which never guesses."""
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.provider, PipelineStep.model, PipelineStep.status,
                PipelineStep.run_id, PipelineStep.executed_at, PipelineStep.duration_ms,
                PipelineStep.input_tokens, PipelineStep.output_tokens, PipelineStep.cost,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.executor == "gateway", PipelineStep.model.is_not(None))
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_rows = rows.all()

    def eff_provider(provider: str | None, model: str | None) -> str:
        return provider or helpers._provider_from_model(model)

    run_counts: dict[str, int] = defaultdict(int)
    failed_counts: dict[str, int] = defaultdict(int)
    token_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cost_totals: dict[str, list] = defaultdict(lambda: [0.0, 0])
    duration_sum: dict[str, float] = defaultdict(float)
    duration_n: dict[str, int] = defaultdict(int)
    status_counts_by_provider: dict[str, dict[str, int]] = defaultdict(dict)
    models_by_provider: dict[str, set[str]] = defaultdict(set)
    breakdown_combo: dict[tuple[str, str, str, str | None, str], dict] = {}

    for r in all_rows:
        provider = eff_provider(r.provider, r.model)
        run_counts[provider] += 1
        if r.status == "failed":
            failed_counts[provider] += 1
        token_totals[provider][0] += r.input_tokens or 0
        token_totals[provider][1] += r.output_tokens or 0
        cost_totals[provider][0] += r.cost or 0.0
        cost_totals[provider][1] += 1 if r.cost is None else 0
        if r.duration_ms is not None:
            duration_sum[provider] += r.duration_ms
            duration_n[provider] += 1
        status_counts_by_provider[provider][r.status] = status_counts_by_provider[provider].get(r.status, 0) + 1
        models_by_provider[provider].add(r.model)

        bkey = (provider, r.pipeline_name, r.step_name, r.agent, r.model)
        bc = breakdown_combo.setdefault(bkey, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "duration_sum_ms": 0.0, "duration_n": 0,
        })
        bc["total"] += 1
        if r.status == "failed":
            bc["failed"] += 1
        bc["input_tokens"] += r.input_tokens or 0
        bc["output_tokens"] += r.output_tokens or 0
        if r.duration_ms is not None:
            bc["duration_sum_ms"] += r.duration_ms
            bc["duration_n"] += 1

    avg_duration_by_provider = {p: duration_sum[p] / duration_n[p] / 1000 for p in duration_n if duration_n[p]}
    overall_avg_duration_secs = (
        sum(duration_sum.values()) / sum(duration_n.values()) / 1000 if sum(duration_n.values()) else None
    )

    # ── Timeseries + recent-list buckets ───────────────────────────────────────
    now = utc_now()
    resolution = helpers._ts_resolution(time_range)
    if all_rows:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_rows)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = helpers._ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_provider: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_provider: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_provider: dict[str, list] = defaultdict(list)

    for r in all_rows:
        provider = eff_provider(r.provider, r.model)
        bucket = helpers._ts_bucket(r.executed_at, resolution)
        runs_by_bucket_provider[provider][bucket] += 1
        if r.duration_ms is not None:
            durations_by_bucket_provider[provider][bucket].append(r.duration_ms / 1000)
        if len(recent_by_provider[provider]) < 5:
            recent_by_provider[provider].append(r)

    # ── Per-provider pipeline/step/agent/model breakdown ───────────────────────
    breakdown_by_provider: dict[str, list[dict]] = defaultdict(list)
    for (provider, pipeline_name, step_name, agent, model), c in breakdown_combo.items():
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        breakdown_by_provider[provider].append({
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "agent": agent,
            "model": model,
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": helpers._format_seconds(avg_duration_secs),
        })
    for rows_ in breakdown_by_provider.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────
    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_provider.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_provider.get(name)
        inp, out = token_totals.get(name, [0, 0])

        runs_ts_data = [runs_by_bucket_provider[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_provider[name][b]) / len(durations_by_bucket_provider[name][b]))
            if durations_by_bucket_provider[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_provider.get(name, []):
            recent.append({
                "id": str(r.run_id),
                "pipeline_name": r.pipeline_name,
                "step_name": r.step_name,
                "status": r.status,
                "ago": helpers._format_ago(r.executed_at),
                "duration": helpers._format_seconds(r.duration_ms / 1000) if r.duration_ms is not None else None,
            })

        cost_sum, unpriced_steps = cost_totals.get(name, [0.0, 0])

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": helpers._format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "cost": cost_sum,
            "unpriced_steps": unpriced_steps,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "step_breakdown": breakdown_by_provider.get(name, []),
        }

    # ── Headline stats ────────────────────────────────────────────────────────
    total_step_executions = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_step_executions - total_failed) / total_step_executions * 100) if total_step_executions else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())

    # ── Chart data ────────────────────────────────────────────────────────────
    provider_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        provider_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_provider.get(name),
            "input_tokens": token_totals.get(name, [0, 0])[0],
            "output_tokens": token_totals.get(name, [0, 0])[1],
            "cost": cost_totals.get(name, [0.0, 0])[0],
            "unpriced_steps": cost_totals.get(name, [0.0, 0])[1],
            "models": sorted(models_by_provider.get(name, [])),
        })

    duration_chart_rows = sorted(avg_duration_by_provider.items(), key=lambda t: t[1], reverse=True)
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    steps_ts = helpers._build_ts(
        [(r.executed_at, eff_provider(r.provider, r.model)) for r in all_rows], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = helpers._build_ts(
        [(r.executed_at, eff_provider(r.provider, r.model), (r.input_tokens or 0) + (r.output_tokens or 0))
         for r in all_rows],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_providers.html", {
        "time_range": time_range,
        "range_label": range_label,
        "provider_rows": provider_rows,
        "duration_chart": duration_chart,
        "steps_ts": steps_ts,
        "tokens_ts": tokens_ts,
        "total_provider_count": len(run_counts),
        "total_step_executions": total_step_executions,
        "overall_success_rate": overall_success_rate,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "currency": pricing.get_table().currency if pricing.get_table() else "USD",
        "drilldown_data": drilldown_data,
        "active_page": "insights_providers",
    })



# --- lines 2731-2953 ---
@router.get("/insights/models", response_class=HTMLResponse)
async def ui_insights_models(request: Request, time_range: str = "7d"):
    """Success rate/duration/tokens grouped by model (gateway executor only).

    Grouped by the display-qualified model identity (see helpers._qualified_model) rather than the
    bare model string — deliberately does NOT guess a provider for rows missing one (unlike
    Insights > Providers, whose whole point is provider bucketing). A bare pre-migration
    "claude-sonnet-5" and a qualified "anthropic/claude-sonnet-5" are kept as distinct rows
    here rather than merged on a guess — same reasoning as helpers._qualified_model everywhere else.
    """
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.provider, PipelineStep.model, PipelineStep.status,
                PipelineStep.run_id, PipelineStep.executed_at, PipelineStep.duration_ms,
                PipelineStep.input_tokens, PipelineStep.output_tokens, PipelineStep.cost,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.executor == "gateway", PipelineStep.model.is_not(None))
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_rows = rows.all()

    run_counts: dict[str, int] = defaultdict(int)
    failed_counts: dict[str, int] = defaultdict(int)
    token_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cost_totals: dict[str, list] = defaultdict(lambda: [0.0, 0])
    duration_sum: dict[str, float] = defaultdict(float)
    duration_n: dict[str, int] = defaultdict(int)
    status_counts_by_model: dict[str, dict[str, int]] = defaultdict(dict)
    agents_by_model: dict[str, set[str]] = defaultdict(set)
    breakdown_combo: dict[tuple[str, str, str, str | None], dict] = {}

    for r in all_rows:
        model = helpers._qualified_model(r.provider, r.model)
        run_counts[model] += 1
        if r.status == "failed":
            failed_counts[model] += 1
        token_totals[model][0] += r.input_tokens or 0
        token_totals[model][1] += r.output_tokens or 0
        cost_totals[model][0] += r.cost or 0.0
        cost_totals[model][1] += 1 if r.cost is None else 0
        if r.duration_ms is not None:
            duration_sum[model] += r.duration_ms
            duration_n[model] += 1
        status_counts_by_model[model][r.status] = status_counts_by_model[model].get(r.status, 0) + 1
        if r.agent:
            agents_by_model[model].add(r.agent)

        bkey = (model, r.pipeline_name, r.step_name, r.agent)
        bc = breakdown_combo.setdefault(bkey, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "duration_sum_ms": 0.0, "duration_n": 0,
        })
        bc["total"] += 1
        if r.status == "failed":
            bc["failed"] += 1
        bc["input_tokens"] += r.input_tokens or 0
        bc["output_tokens"] += r.output_tokens or 0
        if r.duration_ms is not None:
            bc["duration_sum_ms"] += r.duration_ms
            bc["duration_n"] += 1

    avg_duration_by_model = {m: duration_sum[m] / duration_n[m] / 1000 for m in duration_n if duration_n[m]}
    overall_avg_duration_secs = (
        sum(duration_sum.values()) / sum(duration_n.values()) / 1000 if sum(duration_n.values()) else None
    )

    # ── Timeseries + recent-list buckets ───────────────────────────────────────
    now = utc_now()
    resolution = helpers._ts_resolution(time_range)
    if all_rows:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_rows)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = helpers._ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_model: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_model: dict[str, list] = defaultdict(list)

    for r in all_rows:
        model = helpers._qualified_model(r.provider, r.model)
        bucket = helpers._ts_bucket(r.executed_at, resolution)
        runs_by_bucket_model[model][bucket] += 1
        if r.duration_ms is not None:
            durations_by_bucket_model[model][bucket].append(r.duration_ms / 1000)
        if len(recent_by_model[model]) < 5:
            recent_by_model[model].append(r)

    # ── Per-model pipeline/step/agent breakdown ────────────────────────────────
    breakdown_by_model: dict[str, list[dict]] = defaultdict(list)
    for (model, pipeline_name, step_name, agent), c in breakdown_combo.items():
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        breakdown_by_model[model].append({
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "agent": agent,
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": helpers._format_seconds(avg_duration_secs),
        })
    for rows_ in breakdown_by_model.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────
    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_model.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_model.get(name)
        inp, out = token_totals.get(name, [0, 0])

        runs_ts_data = [runs_by_bucket_model[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_model[name][b]) / len(durations_by_bucket_model[name][b]))
            if durations_by_bucket_model[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_model.get(name, []):
            recent.append({
                "id": str(r.run_id),
                "pipeline_name": r.pipeline_name,
                "step_name": r.step_name,
                "status": r.status,
                "ago": helpers._format_ago(r.executed_at),
                "duration": helpers._format_seconds(r.duration_ms / 1000) if r.duration_ms is not None else None,
            })

        cost_sum, unpriced_steps = cost_totals.get(name, [0.0, 0])

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": helpers._format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "cost": cost_sum,
            "unpriced_steps": unpriced_steps,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "step_breakdown": breakdown_by_model.get(name, []),
        }

    # ── Headline stats ────────────────────────────────────────────────────────
    total_step_executions = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_step_executions - total_failed) / total_step_executions * 100) if total_step_executions else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())

    # ── Chart data ────────────────────────────────────────────────────────────
    model_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        model_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_model.get(name),
            "input_tokens": token_totals.get(name, [0, 0])[0],
            "output_tokens": token_totals.get(name, [0, 0])[1],
            "cost": cost_totals.get(name, [0.0, 0])[0],
            "unpriced_steps": cost_totals.get(name, [0.0, 0])[1],
            "agents": sorted(agents_by_model.get(name, [])),
        })

    duration_chart_rows = sorted(avg_duration_by_model.items(), key=lambda t: t[1], reverse=True)
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    steps_ts = helpers._build_ts(
        [(r.executed_at, helpers._qualified_model(r.provider, r.model)) for r in all_rows], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = helpers._build_ts(
        [(r.executed_at, helpers._qualified_model(r.provider, r.model), (r.input_tokens or 0) + (r.output_tokens or 0))
         for r in all_rows],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_models.html", {
        "time_range": time_range,
        "range_label": range_label,
        "model_rows": model_rows,
        "duration_chart": duration_chart,
        "steps_ts": steps_ts,
        "tokens_ts": tokens_ts,
        "total_model_count": len(run_counts),
        "total_step_executions": total_step_executions,
        "overall_success_rate": overall_success_rate,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "currency": pricing.get_table().currency if pricing.get_table() else "USD",
        "drilldown_data": drilldown_data,
        "active_page": "insights_models",
    })



