from fastapi import APIRouter
from ..analytics import _production_only
from ..db.database import get_session_factory
from ..db.models import PipelineRun
from ..db.models import PipelineStep
from ..db.models import RunFeedback
from ..utils import utc_now
from collections import Counter
from collections import defaultdict
from datetime import timedelta
from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from sqlalchemy import select
import asyncio
import json

from . import helpers
from .helpers import templates


# prefix="/ui" here (unlike every other area module) because this module owns the
# bare "" root path — FastAPI bakes self.prefix + path at decoration time, and ""
# + "" is rejected as soon as the router is merged elsewhere. See ui/__init__.py
# for how the aggregate router avoids double-prefixing this module's routes.
router = APIRouter(prefix="/ui")


# --- lines 769-962 ---
@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    sf = get_session_factory()
    cutoff_24h = utc_now() - timedelta(hours=24)
    cutoff_7d  = utc_now() - timedelta(days=7)

    status_panels = await _dashboard_status_panels()

    async with sf() as session:
        rows = await session.execute(_production_only(
            select(PipelineRun.status, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_24h)
            .group_by(PipelineRun.status)
        ))
        counts_24h: dict[str, int] = dict(rows.all())

        rows = await session.execute(_production_only(
            select(PipelineRun.status, func.count().label("n"))
            .group_by(PipelineRun.status)
        ))
        counts_all: dict[str, int] = dict(rows.all())

        rows = await session.execute(_production_only(
            select(PipelineRun.pipeline_name, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_7d)
            .group_by(PipelineRun.pipeline_name)
            .order_by(func.count().desc())
        ))
        pipeline_activity = rows.all()

        rows = await session.execute(_production_only(
            select(PipelineRun.pipeline_name, PipelineRun.team, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_7d)
            .group_by(PipelineRun.pipeline_name, PipelineRun.team)
        ))
        teams_by_pipeline: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for name, team, n in rows.all():
            teams_by_pipeline[name].append((team or "unattributed", n))

        rows = await session.execute(_production_only(
            select(
                PipelineRun.pipeline_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0)
                + func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.input_tokens.is_not(None))
            .group_by(PipelineRun.pipeline_name)
        ))
        tokens_by_pipeline: dict[str, int] = dict(rows.all())

        rows = await session.execute(_production_only(
            select(PipelineRun.source, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_24h)
            .group_by(PipelineRun.source)
        ))
        source_counts = dict(rows.all())

        # Browse surface — shows testing runs too (badged in the template), unlike
        # every aggregate query above/below.
        rows = await session.execute(
            select(PipelineRun)
            .order_by(PipelineRun.triggered_at.desc())
            .limit(10)
        )
        recent_runs = rows.scalars().all()
        feedback_by_run = await helpers._feedback_by_run_id(session, [r.id for r in recent_runs])

        rows = await session.execute(_production_only(
            select(PipelineRun.triggered_at, PipelineRun.status)
            .where(PipelineRun.triggered_at >= cutoff_24h)
        ))
        runs_ts_raw = rows.all()

        rows = await session.execute(
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(PipelineRun.stage == "production")
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        )
        feedback_agg: dict[str, dict[str, int]] = {}
        for name, outcome, n in rows.all():
            d = feedback_agg.setdefault(name, {"correct": 0, "partial": 0, "incorrect": 0, "total": 0})
            d[outcome] = n
            d["total"] += n

        rows = await session.execute(_production_only(
            select(PipelineStep.executor, PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.agent.is_not(None))
            .group_by(PipelineStep.executor, PipelineStep.agent)
            .order_by(func.count().desc())
            .limit(5)
        ))
        top_agents = rows.all()

        rows = await session.execute(_production_only(
            select(PipelineStep.agent_trace)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.agent_trace.is_not(None))
        ))
        tool_call_counts: Counter = Counter()
        for (trace_json,) in rows.all():
            try:
                events = json.loads(trace_json)
            except (TypeError, ValueError):
                continue
            for event in events or []:
                if isinstance(event, dict) and event.get("type") == "tool_call" and event.get("name"):
                    tool_call_counts[event["name"]] += 1
        top_tools = tool_call_counts.most_common(5)

    accuracy_list = [
        {"pipeline": name, "pct": round(d["correct"] / d["total"] * 100), "total": d["total"]}
        for name, d in feedback_agg.items() if d["total"] > 0
    ]
    most_accurate = sorted(accuracy_list, key=lambda x: (-x["pct"], -x["total"]))[:5]
    least_accurate = sorted(accuracy_list, key=lambda x: (x["pct"], -x["total"]))[:5]

    # Enrich pipeline_activity (7d run counts) with per-pipeline team breakdown (7d),
    # accuracy (all-time, same data as most/least accurate above), and tokens (7d).
    pipeline_activity = [
        {
            "name": name,
            "count": count,
            "teams": sorted(teams_by_pipeline.get(name, []), key=lambda t: -t[1]),
            "accuracy": (
                {"pct": round(feedback_agg[name]["correct"] / feedback_agg[name]["total"] * 100),
                 "total": feedback_agg[name]["total"]}
                if feedback_agg.get(name, {}).get("total") else None
            ),
            "tokens": tokens_by_pipeline.get(name, 0),
        }
        for name, count in pipeline_activity
    ]

    total_24h = sum(counts_24h.values())
    non_terminal_24h = counts_24h.get("running", 0) + counts_24h.get("interrupted", 0)
    terminal_24h = total_24h - non_terminal_24h
    success_rate = (
        round((terminal_24h - counts_24h.get("failed", 0)) / terminal_24h * 100)
        if terminal_24h > 0 else None
    )
    pipelines = getattr(request.app.state, "pipelines", [])

    # Status donut (24h)
    status_donut = {
        "labels": list(counts_24h.keys()),
        "data": list(counts_24h.values()),
        "colors": [helpers._STATUS_HEX.get(s, "#71717a") for s in counts_24h.keys()],
    }

    # 24h runs timeseries bucketed by hour and status
    now = utc_now()
    bucket_labels = helpers._ts_all_buckets(cutoff_24h, now, "hour")
    status_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for triggered_at, status in runs_ts_raw:
        status_buckets[status][helpers._ts_bucket(triggered_at, "hour")] += 1
    runs_ts = {
        "labels": bucket_labels,
        "datasets": [
            {
                "label": status,
                "data": [status_buckets[status].get(b, 0) for b in bucket_labels],
                "color": helpers._STATUS_HEX.get(status, "#71717a"),
            }
            for status in sorted(status_buckets, key=lambda s: sum(status_buckets[s].values()), reverse=True)
        ],
    }

    return templates.TemplateResponse(request, "dashboard.html", {
        "counts_24h": counts_24h,
        "counts_all": counts_all,
        "total_24h": total_24h,
        "success_rate": success_rate,
        "pipeline_activity": pipeline_activity,
        "source_counts": source_counts,
        "recent_runs": recent_runs,
        "feedback_by_run": feedback_by_run,
        "pipeline_count": len(pipelines),
        "scheduled_count": sum(1 for p in pipelines if p.schedule),
        "team_count": helpers._team_count,
        "status_donut": status_donut,
        "runs_ts": runs_ts,
        "most_accurate": most_accurate,
        "least_accurate": least_accurate,
        "top_agents": top_agents,
        "top_tools": top_tools,
        "active_page": "dashboard",
        **status_panels,
    })



# --- lines 4119-4186 ---
async def _dashboard_status_panels() -> dict:
    """Backend/MCP/model status for the dashboard's status panels.

    Deliberately reuses the same reachability checks as /ui/agents and /ui/mcp rather
    than a fabricated per-provider (Anthropic/OpenRouter/...) health dot — VectorStep never
    calls providers directly, so "online" here means "the Gateway that talks to them
    responded," which is the only thing this service can honestly claim to know.
    """
    (gw_agents, gw_error), (oc_agents, oc_error), (mcp_tools, mcp_servers, mcp_error) = await asyncio.gather(
        helpers._fetch_vectorstep_gateway_agents(),
        helpers._fetch_openclaw_agents(),
        helpers._fetch_vectorstep_gateway_mcp(),
    )

    backends = [{
        "name": "VectorStep Gateway",
        "online": gw_error is None,
        "agent_count": len(gw_agents),
        "error": gw_error,
    }]
    if helpers._openclaw_enabled:
        backends.append({
            "name": "OpenClaw",
            "online": oc_error is None,
            "agent_count": len(oc_agents),
            "error": oc_error,
        })

    mcp_server_rows = [
        {
            "name": name,
            "running": (mcp_servers.get(name) or {}).get("running"),
            "restart_count": (mcp_servers.get(name) or {}).get("restart_count", 0),
        }
        for name in sorted(set(mcp_tools) | set(mcp_servers))
    ]
    mcp_running_count = sum(1 for s in mcp_server_rows if s["running"])

    # "Configured" = each agent's live primary model, not its failover fallbacks —
    # fallbacks only ever run when the primary fails, so folding them in here would
    # overstate what's actually driving day-to-day traffic.
    models_by_provider: dict[str, Counter] = defaultdict(Counter)
    for a in gw_agents + oc_agents:
        model = a.get("model")
        if not model:
            continue
        provider, sep, bare = model.partition("/")
        if not sep:
            provider, bare = "unspecified", model
        models_by_provider[provider][bare] += 1

    models_configured = [
        {"provider": provider, "models": [{"name": m, "agent_count": n} for m, n in counter.most_common()]}
        for provider, counter in sorted(models_by_provider.items())
    ]
    total_model_count = sum(len(counter) for counter in models_by_provider.values())

    return {
        "backends": backends,
        "mcp_server_rows": mcp_server_rows,
        "mcp_running_count": mcp_running_count,
        "mcp_error": mcp_error,
        "models_configured": models_configured,
        "total_model_count": total_model_count,
        **await helpers._compute_live_pricing_rows(gw_agents + oc_agents),
    }



