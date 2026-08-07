from fastapi import APIRouter
from .. import pricing
from ..analytics import _pipeline_rollup
from ..analytics import _production_only
from ..analytics import _time_range_cutoff
from ..analytics import get_team_month_to_date_spend as _get_team_month_to_date_spend
from ..db.database import get_session_factory
from ..db.models import PipelineRun
from ..db.models import PipelineStep
from ..db.models import RunFeedback
from ..utils import utc_now
from collections import Counter
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from sqlalchemy import select
import asyncio
import json
import re

from . import helpers
from .helpers import _CHART_PALETTE
from .helpers import templates


router = APIRouter()


# --- lines 589-655 ---
async def _fetch_step_agent_model_combo(
    cutoff: datetime | None,
) -> dict[tuple[str | None, str, str, str | None, str | None, str | None], dict]:
    """Raw (team, pipeline_name, step_name, agent, provider, model) -> counters (total,
    failed, tokens, duration), scoped to production and an optional time cutoff.
    provider/model are kept unqualified (raw DB values) so callers can group by real
    provider or real model directly (Providers/Models Insights) as well as by
    pipeline/step/agent/team (Pipelines/Steps/Agents/Teams Insights) — each computes its
    own display-qualified model label at aggregation time via helpers._qualified_model(provider,
    model) rather than baking it into the cache key.

    Shared by every Insights drilldown that breaks down by (team, pipeline, step, agent,
    model) — each re-aggregates this same data along a different axis rather than
    re-querying it."""
    sf = get_session_factory()
    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.team,
                PipelineRun.pipeline_name,
                PipelineStep.step_name,
                PipelineStep.agent,
                PipelineStep.provider,
                PipelineStep.model,
                PipelineStep.status,
                func.count().label("n"),
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
                func.avg(PipelineStep.duration_ms),
                func.coalesce(func.sum(PipelineStep.cost), 0.0),
                func.count(PipelineStep.id) - func.count(PipelineStep.cost),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(
                PipelineRun.team, PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.agent, PipelineStep.provider, PipelineStep.model, PipelineStep.status,
            )
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        raw = rows.all()

    combo: dict[tuple[str | None, str, str, str | None, str | None, str | None], dict] = {}
    for (team, pipeline_name, step_name, agent, provider, model, status, n, in_tok, out_tok,
         avg_dur_ms, cost_sum, unpriced) in raw:
        key = (team, pipeline_name, step_name, agent, provider, model)
        c = combo.setdefault(key, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "duration_sum_ms": 0.0, "duration_n": 0, "cost": 0.0, "unpriced_steps": 0,
        })
        c["total"] += n
        if status == "failed":
            c["failed"] += n
        c["input_tokens"] += in_tok
        c["output_tokens"] += out_tok
        c["cost"] += cost_sum
        c["unpriced_steps"] += unpriced
        if avg_dur_ms is not None:
            # Postgres/asyncpg returns avg(integer_column) as decimal.Decimal, not
            # float (SQLite returns a plain float) — cast so this doesn't crash
            # mixing Decimal into a float accumulator.
            c["duration_sum_ms"] += float(avg_dur_ms) * n
            c["duration_n"] += n
    return combo



# --- lines 1470-1726 ---
@router.get("/insights", response_class=HTMLResponse)
async def ui_insights_overview(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    # Every query below is a rollup for this insights page — scoped to production.
    async with sf() as session:
        q = _production_only(select(PipelineRun.status, func.count().label("n")).group_by(PipelineRun.status))
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        status_counts: dict[str, int] = dict(rows.all())

        q = _production_only(select(PipelineRun.team, func.count().label("n")).group_by(PipelineRun.team))
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        team_counts = rows.all()

        q = _production_only(
            select(
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        total_input_tokens, total_output_tokens = (await session.execute(q)).one()

        # Fetch model + trace together so we can count both tool calls and LLM
        # call iterations per model in a single pass over the blobs.
        q = _production_only(
            select(PipelineStep.model, PipelineStep.agent_trace)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.executor == "gateway")
            .where(PipelineStep.agent_trace.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        model_traces = rows.all()

        q = _production_only(
            select(PipelineRun.pipeline_name, func.count().label("n"))
            .group_by(PipelineRun.pipeline_name)
            .order_by(func.count().desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        pipeline_run_counts = rows.all()

        q = _production_only(
            select(PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineStep.agent)
            .order_by(func.count().desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        agent_step_counts = rows.all()

        q = _production_only(
            select(
                PipelineStep.model,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None))
            .group_by(PipelineStep.model)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        tokens_by_model = rows.all()

        q = (
            select(RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(PipelineRun.stage == "production")
            .group_by(RunFeedback.outcome)
        )
        if cutoff:
            q = q.where(RunFeedback.submitted_at >= cutoff)
        rows = await session.execute(q)
        feedback_by_outcome: dict[str, int] = dict(rows.all())

        # Raw rows for timeseries bucketing — one consolidated fetch per level.
        q = _production_only(
            select(PipelineRun.triggered_at, PipelineRun.status, PipelineRun.team, PipelineRun.pipeline_name)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        run_ts_rows = rows.all()

        q = _production_only(
            select(
                PipelineRun.triggered_at,
                PipelineStep.agent,
                PipelineStep.model,
                PipelineStep.input_tokens,
                PipelineStep.output_tokens,
                PipelineStep.agent_trace,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        step_ts_rows = rows.all()

    tool_counter: Counter = Counter()
    llm_call_counter: Counter = Counter()
    for model, trace_json in model_traces:
        try:
            events = json.loads(trace_json)
        except (TypeError, ValueError):
            continue
        for event in events:
            t = event.get("type")
            if t == "tool_call":
                tool_counter[event.get("name") or "unknown"] += 1
            elif t == "llm_call":
                llm_call_counter[model or "Unknown model"] += 1
    top_tools = tool_counter.most_common(10)

    llm_calls_sorted = llm_call_counter.most_common()

    total_runs = sum(status_counts.values())
    failed_count = status_counts.get("failed", 0)
    failure_rate = round(failed_count / total_runs * 100) if total_runs else None

    runs_by_team = sorted(
        ((team or "Unattributed", n) for team, n in team_counts),
        key=lambda t: t[1], reverse=True,
    )
    team_count = len(runs_by_team)

    tokens_by_model_sorted = sorted(
        ((model or "Unknown model", i, o) for model, i, o in tokens_by_model),
        key=lambda t: t[1] + t[2], reverse=True,
    )

    def _chart(labels, data, colors, extra_rows=None):
        """Build a chart dict with pre-zipped rows for Jinja2 table rendering."""
        rows = [{"label": l, "value": v, "color": c} for l, v, c in zip(labels, data, colors)]
        if extra_rows:
            for row, extra in zip(rows, extra_rows):
                row.update(extra)
        return {"labels": labels, "data": data, "colors": colors, "rows": rows}

    status_labels = list(status_counts.keys())
    status_data = list(status_counts.values())
    status_chart = _chart(
        status_labels, status_data,
        [helpers._STATUS_HEX.get(s, "#71717a") for s in status_labels],
    )

    team_labels = [t for t, _ in runs_by_team]
    team_data = [n for _, n in runs_by_team]
    team_chart = _chart(
        team_labels, team_data,
        [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(team_labels))],
    )

    pipeline_labels = [name for name, _ in pipeline_run_counts]
    pipeline_data = [n for _, n in pipeline_run_counts]
    pipeline_chart = _chart(
        pipeline_labels, pipeline_data,
        [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(pipeline_labels))],
    )

    agent_labels = [agent or "—" for agent, _ in agent_step_counts]
    agent_data = [n for _, n in agent_step_counts]
    agent_step_chart = _chart(
        agent_labels, agent_data,
        [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(agent_labels))],
    )

    llm_labels = [model for model, _ in llm_calls_sorted]
    llm_data = [count for _, count in llm_calls_sorted]
    llm_calls_chart = _chart(
        llm_labels, llm_data,
        [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(llm_labels))],
    )

    # Tokens by model: donut slices are totals; table rows carry input/output detail.
    model_labels = [m for m, _, _ in tokens_by_model_sorted]
    model_totals = [i + o for _, i, o in tokens_by_model_sorted]
    model_colors = [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(model_labels))]
    model_token_chart = _chart(
        model_labels, model_totals, model_colors,
        extra_rows=[{"input": i, "output": o} for _, i, o in tokens_by_model_sorted],
    )

    tool_chart = {
        "labels": [name for name, _ in top_tools],
        "data": [count for _, count in top_tools],
    }

    now = utc_now()
    ts_kw = {"now": now, "cutoff": cutoff, "time_range": time_range}
    status_ts  = helpers._build_ts(run_ts_rows,  dim_fn=lambda r: r[1],  **ts_kw)   # status
    team_ts    = helpers._build_ts(run_ts_rows,  dim_fn=lambda r: r[2] or "Unattributed", **ts_kw)   # team
    pipeline_ts = helpers._build_ts(run_ts_rows, dim_fn=lambda r: r[3],  **ts_kw)   # pipeline
    agent_ts   = helpers._build_ts(step_ts_rows, dim_fn=lambda r: r[1] or "—", **ts_kw)   # agent
    llm_ts     = helpers._build_ts(
        step_ts_rows,
        dim_fn=lambda r: r[2] or "Unknown model",
        val_fn=lambda r: sum(1 for e in (json.loads(r[5]) if r[5] else []) if e.get("type") == "llm_call"),
        **ts_kw,
    )
    tokens_ts  = helpers._build_ts(
        [r for r in step_ts_rows if r[3] is not None],
        dim_fn=lambda r: r[2] or "Unknown model",
        val_fn=lambda r: (r[3] or 0) + (r[4] or 0),
        **ts_kw,
    )

    feedback_total = sum(feedback_by_outcome.values())
    feedback_correct = feedback_by_outcome.get("correct", 0)

    (gw_agents, _gw_error), (oc_agents, _oc_error) = await asyncio.gather(
        helpers._fetch_vectorstep_gateway_agents(),
        helpers._fetch_openclaw_agents(),
    )
    live_pricing_ctx = await helpers._compute_live_pricing_rows(gw_agents + oc_agents)

    return templates.TemplateResponse(request, "insights_overview.html", {
        "time_range": time_range,
        "range_label": range_label,
        "total_runs": total_runs,
        "failed_count": failed_count,
        "failure_rate": failure_rate,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "team_count": team_count,
        "feedback_total": feedback_total,
        "feedback_correct": feedback_correct,
        "feedback_by_outcome": feedback_by_outcome,
        "status_chart": status_chart, "status_ts": status_ts,
        "team_chart": team_chart, "team_ts": team_ts,
        "pipeline_chart": pipeline_chart, "pipeline_ts": pipeline_ts,
        "agent_step_chart": agent_step_chart, "agent_ts": agent_ts,
        "llm_calls_chart": llm_calls_chart, "llm_ts": llm_ts,
        "model_token_chart": model_token_chart, "tokens_ts": tokens_ts,
        "tool_chart": tool_chart,
        **live_pricing_ctx,
        "active_page": "insights_overview",
    })



# --- lines 1727-1963 ---
@router.get("/insights/pipelines", response_class=HTMLResponse)
async def ui_insights_pipelines(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    # Per-pipeline run/token/duration/team/feedback rollup — shared with the
    # /pipelines/{name}/stats and /stats/pipelines JSON endpoints (analytics.py)
    # so this page and those endpoints can never disagree.
    rollup = await _pipeline_rollup(sf, time_range, "production")
    run_counts = {name: r["runs_total"] for name, r in rollup.items()}
    failed_counts = {name: r["status_counts"].get("failed", 0) for name, r in rollup.items()}
    token_totals = {name: (r["tokens"]["input"], r["tokens"]["output"]) for name, r in rollup.items()}
    cost_totals = {name: (r["cost"]["total"], r["cost"]["unpriced_steps"]) for name, r in rollup.items()}
    teams_by_pipeline: dict[str, list[str]] = {name: list(r["teams"]) for name, r in rollup.items()}
    avg_duration_by_pipeline = {
        name: r["duration_seconds"]["avg"] for name, r in rollup.items()
        if r["duration_seconds"]["avg"] is not None
    }
    status_counts_by_pipeline: dict[str, dict[str, int]] = {name: dict(r["status_counts"]) for name, r in rollup.items()}
    feedback_raw: dict[str, dict[str, int]] = {
        name: {
            "correct": r["accuracy"]["correct"],
            "partial": r["accuracy"]["partial"],
            "incorrect": r["accuracy"]["incorrect"],
        }
        for name, r in rollup.items()
    }

    # Per-run rows — needed for the timeseries/recent-runs drilldown, which the
    # shared rollup (aggregated) doesn't expose at this granularity.
    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.id,
                PipelineRun.pipeline_name,
                PipelineRun.status,
                PipelineRun.triggered_at,
                PipelineRun.completed_at,
            )
            .order_by(PipelineRun.triggered_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_runs_raw = rows.all()

    step_combo = await _fetch_step_agent_model_combo(cutoff)

    # ── Per-pipeline aggregates ───────────────────────────────────────────────
    # avg_duration_by_pipeline comes from the shared rollup above.

    now = utc_now()
    resolution = helpers._ts_resolution(time_range)
    if all_runs_raw:
        oldest = min(r.triggered_at.replace(tzinfo=None) for r in all_runs_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = helpers._ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_pipeline: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_pipeline: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_pipeline: dict[str, list] = defaultdict(list)

    for row in all_runs_raw:
        bucket = helpers._ts_bucket(row.triggered_at, resolution)
        runs_by_bucket_pipeline[row.pipeline_name][bucket] += 1
        if row.completed_at:
            secs = (row.completed_at.replace(tzinfo=None) - row.triggered_at.replace(tzinfo=None)).total_seconds()
            if secs >= 0:
                durations_by_bucket_pipeline[row.pipeline_name][bucket].append(secs)
        if len(recent_by_pipeline[row.pipeline_name]) < 5:
            recent_by_pipeline[row.pipeline_name].append(row)

    # ── Per-pipeline step breakdown (agents/models/tokens involved) ───────────
    # step_combo (fetched above) is keyed by (pipeline_name, step_name, agent, provider,
    # model) — re-aggregate it here indexed by pipeline for this page's drilldown. The
    # Steps/Models/Providers insights pages reuse the same combo, indexed differently.
    step_breakdown_by_pipeline: dict[str, list[dict]] = defaultdict(list)
    for (_team, pipeline_name, step_name, agent, provider, model), c in step_combo.items():
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        step_breakdown_by_pipeline[pipeline_name].append({
            "step_name": step_name,
            "agent": agent,
            "model": helpers._qualified_model(provider, model),
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": helpers._format_seconds(avg_duration_secs),
        })
    for rows_ in step_breakdown_by_pipeline.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────

    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_pipeline.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_pipeline.get(name)
        inp, out = token_totals.get(name, (0, 0))
        cost_sum, unpriced_steps = cost_totals.get(name, (0.0, 0))

        runs_ts_data = [runs_by_bucket_pipeline[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_pipeline[name][b]) / len(durations_by_bucket_pipeline[name][b]))
            if durations_by_bucket_pipeline[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_pipeline.get(name, []):
            dur_str = None
            if r.completed_at:
                secs = (r.completed_at.replace(tzinfo=None) - r.triggered_at.replace(tzinfo=None)).total_seconds()
                dur_str = helpers._format_seconds(secs)
            recent.append({
                "id": str(r.id),
                "status": r.status,
                "ago": helpers._format_ago(r.triggered_at),
                "duration": dur_str,
            })

        fb = feedback_raw.get(name, {})
        fb_correct = fb.get("correct", 0)
        fb_partial = fb.get("partial", 0)
        fb_incorrect = fb.get("incorrect", 0)
        fb_total = fb_correct + fb_partial + fb_incorrect

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
            "step_breakdown": step_breakdown_by_pipeline.get(name, []),
            "feedback_total": fb_total,
            "feedback_correct": fb_correct,
            "feedback_partial": fb_partial,
            "feedback_incorrect": fb_incorrect,
            "accuracy_pct": round(fb_correct / fb_total * 100) if fb_total else None,
        }

    # ── Headline accuracy across all pipelines ────────────────────────────────

    insights_feedback_total = sum(sum(fb.values()) for fb in feedback_raw.values())
    insights_feedback_correct = sum(fb.get("correct", 0) for fb in feedback_raw.values())
    insights_accuracy_pct = round(insights_feedback_correct / insights_feedback_total * 100) if insights_feedback_total else None

    # ── Headline stats ────────────────────────────────────────────────────────

    all_durations_flat = []
    for r in all_runs_raw:
        if r.completed_at:
            secs = (r.completed_at.replace(tzinfo=None) - r.triggered_at.replace(tzinfo=None)).total_seconds()
            if secs >= 0:
                all_durations_flat.append(secs)
    min_duration_secs = min(all_durations_flat) if all_durations_flat else None
    overall_avg_duration_secs = (sum(all_durations_flat) / len(all_durations_flat)) if all_durations_flat else None
    max_duration_secs = max(all_durations_flat) if all_durations_flat else None

    # ── Chart data ────────────────────────────────────────────────────────────

    pipeline_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        fb = feedback_raw.get(name, {})
        fb_total = sum(fb.values())
        fb_correct = fb.get("correct", 0)
        pipeline_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_pipeline.get(name),
            "input_tokens": token_totals.get(name, (0, 0))[0],
            "output_tokens": token_totals.get(name, (0, 0))[1],
            "cost": cost_totals.get(name, (0.0, 0))[0],
            "unpriced_steps": cost_totals.get(name, (0.0, 0))[1],
            "teams": sorted(teams_by_pipeline.get(name, [])),
            "feedback_total": fb_total,
            "accuracy_pct": round(fb_correct / fb_total * 100) if fb_total else None,
        })

    duration_chart_rows = sorted(
        ((name, secs) for name, secs in avg_duration_by_pipeline.items()),
        key=lambda t: t[1], reverse=True,
    )
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    token_chart_rows = sorted(
        pipeline_rows, key=lambda r: r["input_tokens"] + r["output_tokens"], reverse=True,
    )
    token_chart_rows = [r for r in token_chart_rows if r["input_tokens"] or r["output_tokens"]]
    token_chart = {
        "labels": [r["name"] for r in token_chart_rows],
        "input": [r["input_tokens"] for r in token_chart_rows],
        "output": [r["output_tokens"] for r in token_chart_rows],
    }

    total_cost = sum(r["cost"] for r in pipeline_rows) or None
    total_unpriced_steps = sum(r["unpriced_steps"] for r in pipeline_rows)

    return templates.TemplateResponse(request, "insights_pipelines.html", {
        "time_range": time_range,
        "range_label": range_label,
        "pipeline_rows": pipeline_rows,
        "duration_chart": duration_chart,
        "token_chart": token_chart,
        "total_pipeline_count": len(run_counts),
        "min_duration_secs": min_duration_secs,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "max_duration_secs": max_duration_secs,
        "total_cost": total_cost,
        "total_unpriced_steps": total_unpriced_steps,
        "currency": pricing.get_table().currency if pricing.get_table() else "USD",
        "drilldown_data": drilldown_data,
        "insights_feedback_total": insights_feedback_total,
        "insights_accuracy_pct": insights_accuracy_pct,
        "active_page": "insights_pipelines",
    })



# --- lines 2954-3128 ---
@router.get("/insights/mcp", response_class=HTMLResponse)
async def ui_insights_mcp(request: Request, time_range: str = "7d"):
    """MCP tool call usage — extracted from PipelineStep.agent_trace (gateway executor
    only; OpenClaw doesn't expose intermediate events, so agent_trace is always NULL for
    those steps and they're naturally excluded). Tool names are namespaced by the Gateway
    as "{server}__{tool}" (see MCPManager) — split on that to group by server too.

    Unlike every other Insights page, tool calls have no token/duration concept of their
    own (those belong to the step/LLM call as a whole) — "errors" (tool_result.is_error)
    takes the place of "failures" as the reliability signal.
    """
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.run_id, PipelineStep.executed_at, PipelineStep.agent_trace,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent_trace.is_not(None))
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_rows = rows.all()

    # Per-row tool usage: {tool_name: {"calls": n, "errors": n}} extracted from that
    # step's trace. A single step can call several tools, each several times.
    row_tool_usage: list[tuple] = []
    for r in all_rows:
        try:
            events = json.loads(r.agent_trace)
        except (TypeError, ValueError):
            continue
        usage: dict[str, dict] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            name = event.get("name")
            if not name:
                continue
            t = event.get("type")
            if t == "tool_call":
                usage.setdefault(name, {"calls": 0, "errors": 0})["calls"] += 1
            elif t == "tool_result":
                u = usage.setdefault(name, {"calls": 0, "errors": 0})
                if event.get("is_error"):
                    u["errors"] += 1
        if usage:
            row_tool_usage.append((r, usage))

    now = utc_now()
    resolution = helpers._ts_resolution(time_range)
    if all_rows:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_rows)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = helpers._ts_all_buckets(ts_start, now, resolution)

    call_counts: dict[str, int] = defaultdict(int)
    error_counts: dict[str, int] = defaultdict(int)
    pipelines_by_tool: dict[str, set[str]] = defaultdict(set)
    calls_by_bucket_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors_by_bucket_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    recent_by_tool: dict[str, list] = defaultdict(list)
    breakdown_combo: dict[tuple[str, str, str, str | None], dict] = {}

    for r, usage in row_tool_usage:
        bucket = helpers._ts_bucket(r.executed_at, resolution)
        for tool, u in usage.items():
            call_counts[tool] += u["calls"]
            error_counts[tool] += u["errors"]
            pipelines_by_tool[tool].add(r.pipeline_name)
            calls_by_bucket_tool[tool][bucket] += u["calls"]
            errors_by_bucket_tool[tool][bucket] += u["errors"]
            if len(recent_by_tool[tool]) < 5:
                recent_by_tool[tool].append({
                    "id": str(r.run_id),
                    "pipeline_name": r.pipeline_name,
                    "step_name": r.step_name,
                    "calls": u["calls"],
                    "errors": u["errors"],
                    "ago": helpers._format_ago(r.executed_at),
                })
            bkey = (tool, r.pipeline_name, r.step_name, r.agent)
            bc = breakdown_combo.setdefault(bkey, {"calls": 0, "errors": 0})
            bc["calls"] += u["calls"]
            bc["errors"] += u["errors"]

    # ── Per-tool pipeline/step/agent breakdown ─────────────────────────────────
    breakdown_by_tool: dict[str, list[dict]] = defaultdict(list)
    for (tool, pipeline_name, step_name, agent), c in breakdown_combo.items():
        calls = c["calls"]
        breakdown_by_tool[tool].append({
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "agent": agent,
            "calls": calls,
            "errors": c["errors"],
            "success_rate": round((calls - c["errors"]) / calls * 100) if calls else None,
        })
    for rows_ in breakdown_by_tool.values():
        rows_.sort(key=lambda r: r["calls"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────
    drilldown_data: dict[str, dict] = {}
    for tool, calls in call_counts.items():
        errors = error_counts.get(tool, 0)
        success_rate = round((calls - errors) / calls * 100) if calls else None
        drilldown_data[tool] = {
            "calls": calls,
            "errors": errors,
            "success_rate": success_rate,
            "calls_ts": {"labels": bucket_labels, "data": [calls_by_bucket_tool[tool].get(b, 0) for b in bucket_labels]},
            "errors_ts": {"labels": bucket_labels, "data": [errors_by_bucket_tool[tool].get(b, 0) for b in bucket_labels]},
            "recent": recent_by_tool.get(tool, []),
            "breakdown": breakdown_by_tool.get(tool, []),
        }

    # ── Headline stats ────────────────────────────────────────────────────────
    total_calls = sum(call_counts.values())
    total_errors = sum(error_counts.values())
    overall_success_rate = round((total_calls - total_errors) / total_calls * 100) if total_calls else None
    distinct_servers = len({tool.partition("__")[0] for tool in call_counts})

    # ── Chart data ────────────────────────────────────────────────────────────
    tool_rows = []
    for tool, calls in sorted(call_counts.items(), key=lambda t: t[1], reverse=True):
        server, _, bare_tool = tool.partition("__")
        errors = error_counts.get(tool, 0)
        tool_rows.append({
            "name": tool,
            "server": server,
            "tool": bare_tool or tool,
            "calls": calls,
            "errors": errors,
            "success_rate": round((calls - errors) / calls * 100) if calls else None,
            "pipelines": sorted(pipelines_by_tool.get(tool, [])),
        })

    calls_by_server: dict[str, int] = defaultdict(int)
    for tool, calls in call_counts.items():
        calls_by_server[tool.partition("__")[0]] += calls
    server_chart_rows = sorted(calls_by_server.items(), key=lambda t: t[1], reverse=True)
    server_chart = {
        "labels": [name for name, _ in server_chart_rows],
        "data": [n for _, n in server_chart_rows],
    }

    calls_ts_rows = [(r.executed_at, tool, u["calls"]) for r, usage in row_tool_usage for tool, u in usage.items()]
    errors_ts_rows = [(r.executed_at, tool, u["errors"]) for r, usage in row_tool_usage for tool, u in usage.items()]
    calls_ts = helpers._build_ts(calls_ts_rows, now, cutoff, time_range, dim_fn=lambda r: r[1], val_fn=lambda r: r[2])
    errors_ts = helpers._build_ts(errors_ts_rows, now, cutoff, time_range, dim_fn=lambda r: r[1], val_fn=lambda r: r[2])

    return templates.TemplateResponse(request, "insights_mcp.html", {
        "time_range": time_range,
        "range_label": range_label,
        "tool_rows": tool_rows,
        "server_chart": server_chart,
        "calls_ts": calls_ts,
        "errors_ts": errors_ts,
        "total_tool_count": len(call_counts),
        "total_calls": total_calls,
        "total_errors": total_errors,
        "overall_success_rate": overall_success_rate,
        "distinct_servers": distinct_servers,
        "drilldown_data": drilldown_data,
        "active_page": "insights_mcp",
    })



# --- lines 3129-3414 ---
@router.get("/insights/teams", response_class=HTMLResponse)
async def ui_insights_teams(request: Request, time_range: str = "7d"):
    """Per-team rollup — which pipelines/steps/agents/models a team uses, and its token
    spend, so a team can see a complete picture of what they're running and make informed
    cost decisions. `team` is a PipelineRun attribute (resolved from the webhook auth
    token — see README §3b); NULL is bucketed as "Unattributed" rather than dropped, since
    unattributed spend is exactly the kind of thing this page exists to surface.

    Run counts/success-rate/duration come from PipelineRun directly (a "run" is a run,
    not a step); tokens and the "what they're using" breakdown come from the shared
    per-step combo (see _fetch_step_agent_model_combo), which also carries `team`.
    """
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    def norm_team(t: str | None) -> str:
        return t or "Unattributed"

    async with sf() as session:
        q = _production_only(select(PipelineRun.team, func.count().label("n")).group_by(PipelineRun.team))
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        run_counts_raw = rows.all()

        q = _production_only(
            select(PipelineRun.team, func.count().label("n"))
            .where(PipelineRun.status == "failed")
            .group_by(PipelineRun.team)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        failed_counts_raw = rows.all()

        q = _production_only(
            select(PipelineRun.team, PipelineRun.status, func.count().label("n"))
            .group_by(PipelineRun.team, PipelineRun.status)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        status_counts_raw = rows.all()

        q = _production_only(select(PipelineRun.team, PipelineRun.pipeline_name).distinct())
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        pipelines_by_team_raw = rows.all()

        # No duration column on pipeline_runs — compute from timestamps, same approach as
        # ui_insights_pipelines. Only terminal runs (completed_at set) count.
        q = _production_only(
            select(PipelineRun.team, PipelineRun.triggered_at, PipelineRun.completed_at)
            .where(PipelineRun.completed_at.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        duration_rows_raw = rows.all()

        # All runs in range — used for per-team timeseries and recent-runs list
        q = _production_only(
            select(
                PipelineRun.id, PipelineRun.team, PipelineRun.pipeline_name,
                PipelineRun.status, PipelineRun.triggered_at, PipelineRun.completed_at,
            )
            .order_by(PipelineRun.triggered_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_runs_raw = rows.all()

        # Per-step token usage with timestamps — for the tokens-over-time chart (tokens
        # live on PipelineStep, not PipelineRun, so this needs its own query).
        q = _production_only(
            select(PipelineRun.team, PipelineStep.executed_at, PipelineStep.input_tokens, PipelineStep.output_tokens)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        token_ts_raw = rows.all()

    # step_combo (shared with Pipelines/Steps/Agents Insights) now also carries team —
    # re-aggregated here indexed by team for tokens and the "what they're using" breakdown.
    step_combo = await _fetch_step_agent_model_combo(cutoff)

    run_counts: dict[str, int] = defaultdict(int)
    for team, n in run_counts_raw:
        run_counts[norm_team(team)] += n

    failed_counts: dict[str, int] = defaultdict(int)
    for team, n in failed_counts_raw:
        failed_counts[norm_team(team)] += n

    status_counts_by_team: dict[str, dict[str, int]] = defaultdict(dict)
    for team, status, n in status_counts_raw:
        tk = norm_team(team)
        status_counts_by_team[tk][status] = status_counts_by_team[tk].get(status, 0) + n

    pipelines_by_team: dict[str, set[str]] = defaultdict(set)
    for team, pipeline_name in pipelines_by_team_raw:
        pipelines_by_team[norm_team(team)].add(pipeline_name)

    durations_by_team: dict[str, list[float]] = defaultdict(list)
    for team, triggered_at, completed_at in duration_rows_raw:
        secs = (completed_at.replace(tzinfo=None) - triggered_at.replace(tzinfo=None)).total_seconds()
        if secs >= 0:
            durations_by_team[norm_team(team)].append(secs)
    avg_duration_by_team = {t: sum(v) / len(v) for t, v in durations_by_team.items() if v}

    token_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cost_totals: dict[str, list] = defaultdict(lambda: [0.0, 0])
    breakdown_combo_by_team: dict[str, list] = defaultdict(list)
    for (team, pipeline_name, step_name, agent, provider, model), c in step_combo.items():
        tk = norm_team(team)
        token_totals[tk][0] += c["input_tokens"]
        token_totals[tk][1] += c["output_tokens"]
        cost_totals[tk][0] += c["cost"]
        cost_totals[tk][1] += c["unpriced_steps"]
        breakdown_combo_by_team[tk].append((pipeline_name, step_name, agent, provider, model, c))

    breakdown_by_team: dict[str, list[dict]] = defaultdict(list)
    for tk, entries in breakdown_combo_by_team.items():
        for pipeline_name, step_name, agent, provider, model, c in entries:
            total = c["total"]
            avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
            breakdown_by_team[tk].append({
                "pipeline_name": pipeline_name,
                "step_name": step_name,
                "agent": agent,
                "model": helpers._qualified_model(provider, model),
                "total": total,
                "success_rate": round((total - c["failed"]) / total * 100) if total else None,
                "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
                "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
                "avg_duration": helpers._format_seconds(avg_duration_secs),
            })
    for rows_ in breakdown_by_team.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Timeseries + recent-list buckets ───────────────────────────────────────
    now = utc_now()
    resolution = helpers._ts_resolution(time_range)
    if all_runs_raw:
        oldest = min(r.triggered_at.replace(tzinfo=None) for r in all_runs_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = helpers._ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_team: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_team: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_team: dict[str, list] = defaultdict(list)

    for r in all_runs_raw:
        tk = norm_team(r.team)
        bucket = helpers._ts_bucket(r.triggered_at, resolution)
        runs_by_bucket_team[tk][bucket] += 1
        if r.completed_at:
            secs = (r.completed_at.replace(tzinfo=None) - r.triggered_at.replace(tzinfo=None)).total_seconds()
            if secs >= 0:
                durations_by_bucket_team[tk][bucket].append(secs)
        if len(recent_by_team[tk]) < 5:
            dur_str = None
            if r.completed_at:
                secs = (r.completed_at.replace(tzinfo=None) - r.triggered_at.replace(tzinfo=None)).total_seconds()
                dur_str = helpers._format_seconds(secs)
            recent_by_team[tk].append({
                "id": str(r.id), "pipeline_name": r.pipeline_name, "status": r.status,
                "ago": helpers._format_ago(r.triggered_at), "duration": dur_str,
            })

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────
    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_team.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_team.get(name)
        inp, out = token_totals.get(name, [0, 0])
        cost_sum, unpriced_steps = cost_totals.get(name, [0.0, 0])

        runs_ts_data = [runs_by_bucket_team[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_team[name][b]) / len(durations_by_bucket_team[name][b]))
            if durations_by_bucket_team[name].get(b) else None
            for b in bucket_labels
        ]

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
            "recent_runs": recent_by_team.get(name, []),
            "step_breakdown": breakdown_by_team.get(name, []),
            "pipelines": sorted(pipelines_by_team.get(name, [])),
        }

    # ── Headline stats ────────────────────────────────────────────────────────
    total_runs = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_runs - total_failed) / total_runs * 100) if total_runs else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())
    total_pipelines_used = len({p for pset in pipelines_by_team.values() for p in pset})

    # ── Chart data ────────────────────────────────────────────────────────────
    team_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        inp, out = token_totals.get(name, [0, 0])
        cost_sum, unpriced_steps = cost_totals.get(name, [0.0, 0])
        team_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_team.get(name),
            "input_tokens": inp,
            "output_tokens": out,
            "cost": cost_sum,
            "unpriced_steps": unpriced_steps,
            "pipelines": sorted(pipelines_by_team.get(name, [])),
        })

    # Month-to-date spend vs pricing.team_budgets — advisory only, deliberately NOT
    # scoped by the page's time_range selector (a budget is always "this calendar
    # month", not "the last 7/30 days"). See get_team_month_to_date_spend.
    team_budgets_raw = await _get_team_month_to_date_spend(sf)
    team_budgets = {row["team"]: row for row in team_budgets_raw.values() if row["budget"] is not None}
    currency = pricing.get_table().currency if pricing.get_table() else "USD"

    token_chart_rows = sorted(
        (r for r in team_rows if r["input_tokens"] or r["output_tokens"]),
        key=lambda r: r["input_tokens"] + r["output_tokens"], reverse=True,
    )
    token_chart = {
        "labels": [r["name"] for r in token_chart_rows],
        "input": [r["input_tokens"] for r in token_chart_rows],
        "output": [r["output_tokens"] for r in token_chart_rows],
    }

    runs_ts = helpers._build_ts(
        [(r.triggered_at, norm_team(r.team)) for r in all_runs_raw], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = helpers._build_ts(
        [(r.executed_at, norm_team(r.team), (r.input_tokens or 0) + (r.output_tokens or 0)) for r in token_ts_raw],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_teams.html", {
        "time_range": time_range,
        "range_label": range_label,
        "team_rows": team_rows,
        "token_chart": token_chart,
        "runs_ts": runs_ts,
        "tokens_ts": tokens_ts,
        "total_team_count": len(run_counts),
        "total_runs": total_runs,
        "overall_success_rate": overall_success_rate,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_pipelines_used": total_pipelines_used,
        "drilldown_data": drilldown_data,
        "team_budgets": team_budgets,
        "currency": currency,
        "active_page": "insights_teams",
    })



