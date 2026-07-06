import asyncio
import glob
import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from .db.database import get_session_factory
from .db.models import PipelineRun, PipelineStep, RunFeedback
from .executors.human import pending_count as _pending_approval_count
from .gateway import gateway_call_safe
from .models.pipeline import FanOutGroupConfig, ParallelGroupConfig, PipelineConfig
from .utils import utc_now
from . import run_events

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui")

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "..", "templates")
)


# ── Template helpers ──────────────────────────────────────────────────────────

def _status_classes(status: str) -> str:
    return {
        "completed":   "bg-green-950 text-green-400 ring-green-800",
        "running":     "bg-blue-950 text-blue-400 ring-blue-800",
        "escalated":   "bg-amber-950 text-amber-400 ring-amber-800",
        "aborted":     "bg-orange-950 text-orange-400 ring-orange-800",
        "failed":      "bg-red-950 text-red-400 ring-red-800",
        "stopped":     "bg-purple-950 text-purple-400 ring-purple-800",
        "interrupted": "bg-zinc-700 text-zinc-300 ring-zinc-500",
    }.get(status or "", "bg-zinc-800 text-zinc-400 ring-zinc-600")


_CHART_PALETTE = ["#6366f1", "#22d3ee", "#fbbf24", "#34d399", "#fb7185", "#a78bfa", "#71717a"]

_STATUS_HEX = {
    "completed":   "#4ade80",
    "running":     "#60a5fa",
    "escalated":   "#fbbf24",
    "aborted":     "#fb923c",
    "failed":      "#f87171",
    "stopped":     "#c084fc",
    "interrupted": "#a1a1aa",
}


def _format_seconds(secs: float | None) -> str:
    """Like _format_duration, but takes a raw seconds value (e.g. an average) rather than two datetimes."""
    if secs is None:
        return "—"
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _confidence_bar_color(c: float | None) -> str:
    if c is None:
        return "bg-gray-300"
    if c >= 0.75:
        return "bg-green-500"
    if c >= 0.5:
        return "bg-amber-400"
    return "bg-red-400"


def _format_duration(start: datetime, end: datetime | None) -> str:
    if end is None:
        return "in progress"
    secs = int((end.replace(tzinfo=None) - start.replace(tzinfo=None)).total_seconds())
    if secs < 0:
        return "—"
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _format_ago(dt: datetime) -> str:
    secs = int((utc_now() - dt.replace(tzinfo=None)).total_seconds())
    if secs < 5:
        return "just now"
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


_TELEGRAM_HTML_TAG_RESTORE = [
    (re.compile(r"&lt;b&gt;(.*?)&lt;/b&gt;", re.IGNORECASE | re.DOTALL), r"<b>\1</b>"),
    (re.compile(r"&lt;strong&gt;(.*?)&lt;/strong&gt;", re.IGNORECASE | re.DOTALL), r"<strong>\1</strong>"),
    (re.compile(r"&lt;i&gt;(.*?)&lt;/i&gt;", re.IGNORECASE | re.DOTALL), r"<i>\1</i>"),
    (re.compile(r"&lt;em&gt;(.*?)&lt;/em&gt;", re.IGNORECASE | re.DOTALL), r"<em>\1</em>"),
    (re.compile(r"&lt;code&gt;(.*?)&lt;/code&gt;", re.IGNORECASE | re.DOTALL), r"<code>\1</code>"),
]
# Only http(s) links are restored — never javascript:/data: schemes — and only from the
# escaped form, so a raw unescaped "<a href=...>" smuggled in via {{summary}} or an
# agent's own output never matches (it's not literal &lt;a href=&#34;...&gt; text).
_TELEGRAM_HTML_LINK_RE = re.compile(
    r'&lt;a href=&#34;(https?://[^&"]*)&#34;&gt;(.*?)&lt;/a&gt;', re.IGNORECASE | re.DOTALL
)


def _telegram_html_to_safe_html(text: str):
    """Render the small Telegram HTML parse-mode subset used in `human` step
    prompt_templates (<b>, <i>, <code>, <a href>) as real HTML on the /ui/approvals
    pages, without trusting arbitrary content.

    The message text is a fully-rendered Jinja2 prompt_template — it can embed
    {{summary}}, {{steps.x.field}}, etc., which ultimately trace back to webhook
    payloads or LLM output, neither of which is trusted input. So this escapes the
    *entire* string first (exactly like Jinja2's default autoescape would), then
    re-opens only the small whitelist of tags above from their escaped form. Anything
    else — including an attempt to smuggle in a real <script> tag — stays inert
    escaped text rather than becoming live markup.
    """
    from markupsafe import Markup, escape

    escaped = str(escape(text))
    for pattern, replacement in _TELEGRAM_HTML_TAG_RESTORE:
        escaped = pattern.sub(replacement, escaped)
    escaped = _TELEGRAM_HTML_LINK_RE.sub(
        lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener noreferrer">{m.group(2)}</a>',
        escaped,
    )
    return Markup(escaped)


def _source_label(source: str) -> str:
    return {"alertmanager": "Alertmanager", "scheduler": "Scheduler", "generic": "Generic"}.get(
        source, source
    )


_TIME_RANGES = {
    "24h": (timedelta(hours=24), "24 hours"),
    "7d": (timedelta(days=7), "7 days"),
    "30d": (timedelta(days=30), "30 days"),
}


def _time_range_cutoff(time_range: str) -> tuple[datetime | None, str]:
    """Map 24h|7d|30d|all to (cutoff_datetime_or_None, label) for the Insights pages."""
    delta_label = _TIME_RANGES.get(time_range)
    if delta_label is None:
        return None, "all time"
    delta, label = delta_label
    return utc_now() - delta, label


def _ts_resolution(time_range: str) -> str:
    return "hour" if time_range == "24h" else "week" if time_range == "all" else "day"


def _ts_bucket(dt: datetime, resolution: str) -> str:
    d = dt.replace(tzinfo=None)
    if resolution == "hour":
        return d.strftime("%d %H:00")
    if resolution == "week":
        monday = d - timedelta(days=d.weekday())
        return monday.strftime("%b %d")
    return d.strftime("%b %d")


def _ts_all_buckets(start: datetime, end: datetime, resolution: str) -> list[str]:
    """All bucket keys from start to end, so empty buckets render as zero."""
    buckets: list[str] = []
    seen: set[str] = set()
    cur = start.replace(tzinfo=None, second=0, microsecond=0)
    if resolution == "hour":
        cur = cur.replace(minute=0)
        step = timedelta(hours=1)
    elif resolution == "week":
        cur = (cur - timedelta(days=cur.weekday())).replace(hour=0, minute=0)
        step = timedelta(weeks=1)
    else:
        cur = cur.replace(hour=0, minute=0)
        step = timedelta(days=1)
    end_clean = end.replace(tzinfo=None)
    while cur <= end_clean + step:
        key = _ts_bucket(cur, resolution)
        if key not in seen:
            seen.add(key)
            buckets.append(key)
        cur += step
    return buckets


def _build_ts(
    rows: list,
    now: datetime,
    cutoff: datetime | None,
    time_range: str,
    dim_fn,
    val_fn=None,
    top_n: int = 7,
) -> dict:
    """Build a Chart.js multi-series line-chart dict from raw (timestamp, ...) rows.

    dim_fn(row) -> series name (string)
    val_fn(row) -> numeric value (default: 1 per row, i.e. counts)
    """
    if val_fn is None:
        val_fn = lambda _r: 1

    resolution = _ts_resolution(time_range)
    start = (cutoff or (min((r[0] for r in rows), default=now) if rows else now - timedelta(days=7)))
    bucket_labels = _ts_all_buckets(start, now, resolution)

    accumulator: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = _ts_bucket(row[0], resolution)
        dim = str(dim_fn(row)) if dim_fn(row) is not None else "—"
        accumulator[dim][key] += val_fn(row)

    # Keep only the top_n most active series
    dims = sorted(accumulator, key=lambda d: sum(accumulator[d].values()), reverse=True)[:top_n]

    datasets = [
        {
            "label": dim,
            "data": [accumulator[dim].get(b, 0) for b in bucket_labels],
            "color": _CHART_PALETTE[i % len(_CHART_PALETTE)],
        }
        for i, dim in enumerate(dims)
    ]
    return {"labels": bucket_labels, "datasets": datasets}


class _LiteralBlockDumper(yaml.Dumper):
    pass


def _literal_str(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralBlockDumper.add_representer(str, _literal_str)


def _to_yaml(obj) -> str:
    return yaml.dump(
        obj, Dumper=_LiteralBlockDumper,
        default_flow_style=False, allow_unicode=True, sort_keys=False,
    )


def _to_json(obj, indent=None) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False)


def _config_fingerprint(steps: list) -> str:
    """12-char SHA-256 of the ordered (step_name, agent, model) sequence for a run."""
    key = tuple(
        (s.step_name, s.agent or "", s.model or "")
        for s in sorted(steps, key=lambda x: x.step_index)
    )
    return hashlib.sha256(str(key).encode()).hexdigest()[:12]


def _config_description(steps: list) -> dict:
    """Human-readable summary of the config for display in the feedback breakdown."""
    sorted_steps = sorted(steps, key=lambda s: s.step_index)
    step_names = [s.step_name for s in sorted_steps if "/" not in s.step_name]
    models = sorted({s.model for s in sorted_steps if s.model})
    return {"steps": step_names, "models": models}


_OUTCOME_CLASSES = {
    "correct":   "bg-green-950 text-green-400 ring-green-800",
    "partial":   "bg-amber-950 text-amber-400 ring-amber-800",
    "incorrect": "bg-red-950 text-red-400 ring-red-800",
}


async def _feedback_by_run_id(session, run_ids: list[str]) -> dict[str, str]:
    """outcome for each run_id that has feedback — absent key means unmarked.

    Used by every template that renders a runs table (dashboard, /runs, pipeline
    detail) to show the feedback_badge macro without a per-row query.
    """
    if not run_ids:
        return {}
    rows = await session.execute(
        select(RunFeedback.run_id, RunFeedback.outcome).where(RunFeedback.run_id.in_(run_ids))
    )
    return dict(rows.all())

templates.env.filters["to_yaml"] = _to_yaml
templates.env.filters["tojson"] = _to_json
templates.env.filters["format_number"] = lambda n: f"{int(n):,}"
templates.env.filters["telegram_html"] = _telegram_html_to_safe_html
templates.env.globals.update({
    "status_classes": _status_classes,
    "confidence_bar_color": _confidence_bar_color,
    "format_duration": _format_duration,
    "format_seconds": _format_seconds,
    "format_ago": _format_ago,
    "pending_approval_count": _pending_approval_count,
    "source_label": _source_label,
    "outcome_classes": lambda o: _OUTCOME_CLASSES.get(o or "", "bg-zinc-800 text-zinc-400 ring-zinc-600"),
})


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    sf = get_session_factory()
    cutoff_24h = utc_now() - timedelta(hours=24)
    cutoff_7d  = utc_now() - timedelta(days=7)

    async with sf() as session:
        rows = await session.execute(
            select(PipelineRun.status, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_24h)
            .group_by(PipelineRun.status)
        )
        counts_24h: dict[str, int] = dict(rows.all())

        rows = await session.execute(
            select(PipelineRun.status, func.count().label("n"))
            .group_by(PipelineRun.status)
        )
        counts_all: dict[str, int] = dict(rows.all())

        rows = await session.execute(
            select(PipelineRun.pipeline_name, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_7d)
            .group_by(PipelineRun.pipeline_name)
            .order_by(func.count().desc())
        )
        pipeline_activity = rows.all()

        rows = await session.execute(
            select(PipelineRun.pipeline_name, PipelineRun.team, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_7d)
            .group_by(PipelineRun.pipeline_name, PipelineRun.team)
        )
        teams_by_pipeline: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for name, team, n in rows.all():
            teams_by_pipeline[name].append((team or "unattributed", n))

        rows = await session.execute(
            select(
                PipelineRun.pipeline_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0)
                + func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.input_tokens.is_not(None))
            .group_by(PipelineRun.pipeline_name)
        )
        tokens_by_pipeline: dict[str, int] = dict(rows.all())

        rows = await session.execute(
            select(PipelineRun.source, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_24h)
            .group_by(PipelineRun.source)
        )
        source_counts = dict(rows.all())

        rows = await session.execute(
            select(PipelineRun)
            .order_by(PipelineRun.triggered_at.desc())
            .limit(10)
        )
        recent_runs = rows.scalars().all()
        feedback_by_run = await _feedback_by_run_id(session, [r.id for r in recent_runs])

        rows = await session.execute(
            select(PipelineRun.triggered_at, PipelineRun.status)
            .where(PipelineRun.triggered_at >= cutoff_24h)
        )
        runs_ts_raw = rows.all()

        rows = await session.execute(
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count().label("n"))
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        )
        feedback_agg: dict[str, dict[str, int]] = {}
        for name, outcome, n in rows.all():
            d = feedback_agg.setdefault(name, {"correct": 0, "partial": 0, "incorrect": 0, "total": 0})
            d[outcome] = n
            d["total"] += n

        rows = await session.execute(
            select(PipelineStep.executor, PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.agent.is_not(None))
            .group_by(PipelineStep.executor, PipelineStep.agent)
            .order_by(func.count().desc())
            .limit(5)
        )
        top_agents = rows.all()

        rows = await session.execute(
            select(PipelineStep.agent_trace)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.agent_trace.is_not(None))
        )
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
        "colors": [_STATUS_HEX.get(s, "#71717a") for s in counts_24h.keys()],
    }

    # 24h runs timeseries bucketed by hour and status
    now = utc_now()
    bucket_labels = _ts_all_buckets(cutoff_24h, now, "hour")
    status_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for triggered_at, status in runs_ts_raw:
        status_buckets[status][_ts_bucket(triggered_at, "hour")] += 1
    runs_ts = {
        "labels": bucket_labels,
        "datasets": [
            {
                "label": status,
                "data": [status_buckets[status].get(b, 0) for b in bucket_labels],
                "color": _STATUS_HEX.get(status, "#71717a"),
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
        "team_count": _team_count,
        "status_donut": status_donut,
        "runs_ts": runs_ts,
        "most_accurate": most_accurate,
        "least_accurate": least_accurate,
        "top_agents": top_agents,
        "top_tools": top_tools,
        "active_page": "dashboard",
    })


@router.get("/runs", response_class=HTMLResponse)
async def ui_runs(
    request: Request,
    status: str | None = None,
    pipeline: str | None = None,
    team: str | None = None,
    time_range: str = "all",
    limit: int = 50,
    offset: int = 0,
):
    cutoff, range_label = _time_range_cutoff(time_range)

    def _filtered(q):
        if status:
            q = q.where(PipelineRun.status == status)
        if pipeline:
            q = q.where(PipelineRun.pipeline_name == pipeline)
        if team:
            q = q.where(PipelineRun.team == team)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        return q

    sf = get_session_factory()
    async with sf() as session:
        q = _filtered(select(PipelineRun).order_by(PipelineRun.triggered_at.desc()))
        rows = await session.execute(q.limit(limit).offset(offset))
        runs = rows.scalars().all()
        feedback_by_run = await _feedback_by_run_id(session, [r.id for r in runs])

        rows = await session.execute(
            select(PipelineRun.pipeline_name).distinct().order_by(PipelineRun.pipeline_name)
        )
        pipeline_names = [r[0] for r in rows.all()]

        rows = await session.execute(
            select(PipelineRun.team).distinct().where(PipelineRun.team.is_not(None)).order_by(PipelineRun.team)
        )
        team_names = [r[0] for r in rows.all()]

        # Stat cards — aggregated over every run matching the active filters, not just the current page.
        q = _filtered(select(PipelineRun.status, func.count()).group_by(PipelineRun.status))
        rows = await session.execute(q)
        status_counts: dict[str, int] = dict(rows.all())

        total_matching = sum(status_counts.values())
        non_terminal = status_counts.get("running", 0) + status_counts.get("interrupted", 0)
        terminal = total_matching - non_terminal
        success_rate = (
            round((terminal - status_counts.get("failed", 0)) / terminal * 100)
            if terminal > 0 else None
        )
        escalated_failed = (
            status_counts.get("escalated", 0)
            + status_counts.get("failed", 0)
            + status_counts.get("aborted", 0)
        )

        q = _filtered(
            select(RunFeedback.outcome, func.count())
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .group_by(RunFeedback.outcome)
        )
        rows = await session.execute(q)
        feedback_counts: dict[str, int] = dict(rows.all())
        marked_total = sum(feedback_counts.values())
        accuracy_pct = (
            round(feedback_counts.get("correct", 0) / marked_total * 100)
            if marked_total > 0 else None
        )

        q = _filtered(
            select(func.coalesce(func.sum(PipelineStep.input_tokens), 0) + func.coalesce(func.sum(PipelineStep.output_tokens), 0))
            .select_from(PipelineRun)
            .join(PipelineStep, PipelineStep.run_id == PipelineRun.id)
        )
        token_total = (await session.execute(q)).scalar() or 0

    return templates.TemplateResponse(request, "runs.html", {
        "runs": runs,
        "feedback_by_run": feedback_by_run,
        "pipeline_names": pipeline_names,
        "team_names": team_names,
        "selected_status": status or "",
        "selected_pipeline": pipeline or "",
        "selected_team": team or "",
        "time_range": time_range,
        "range_label": range_label,
        "limit": limit,
        "offset": offset,
        "statuses": ["completed", "running", "escalated", "aborted", "failed", "stopped", "interrupted"],
        "active_page": "runs",
        "total_matching": total_matching,
        "success_rate": success_rate,
        "escalated_failed": escalated_failed,
        "marked_total": marked_total,
        "accuracy_pct": accuracy_pct,
        "token_total": token_total,
    })


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def ui_run_detail(request: Request, run_id: str):
    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.id == run_id)
            .options(selectinload(PipelineRun.steps))
        )
        run = rows.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    sorted_steps = sorted(run.steps, key=lambda s: s.step_index)

    display_items: list[dict] = []
    seen_groups: set[str] = set()

    for step in sorted_steps:
        parsed = json.loads(step.parsed_output) if step.parsed_output else {}
        pretty = json.dumps(parsed, indent=2) if parsed else ""

        verifier_parsed = json.loads(step.verifier_output) if step.verifier_output else {}
        verifier_pretty = json.dumps(verifier_parsed, indent=2) if verifier_parsed else ""
        verifier_label = "Challenger" if step.verifier_mode == "challenger" else "Verifier"

        trace = json.loads(step.agent_trace) if step.agent_trace else []

        if "/" in step.step_name:
            group_name, branch_name = step.step_name.split("/", 1)
            if group_name not in seen_groups:
                seen_groups.add(group_name)
                display_items.append({"type": "group_header", "name": group_name})
            display_items.append({
                "type": "branch",
                "group": group_name,
                "name": branch_name,
                "step": step,
                "parsed": parsed,
                "pretty": pretty,
                "verifier_pretty": verifier_pretty,
                "verifier_label": verifier_label,
                "trace": trace,
            })
        else:
            display_items.append({
                "type": "step",
                "name": step.step_name,
                "step": step,
                "parsed": parsed,
                "pretty": pretty,
                "verifier_pretty": verifier_pretty,
                "verifier_label": verifier_label,
                "trace": trace,
            })

    normalised = json.loads(run.normalised_context) if run.normalised_context else {}
    run_log = json.loads(run.logs) if run.logs else []

    total_input_tokens = sum(s.input_tokens or 0 for s in run.steps)
    total_output_tokens = sum(s.output_tokens or 0 for s in run.steps)

    # Re-run support: load prior steps from the original run so the UI can show
    # the full picture (replayed steps + new steps together).
    original_run = None
    rerun_prior_items: list[dict] = []
    rerun_from_step: str | None = None

    if normalised.get("source") == "rerun":
        rerun_meta = normalised.get("metadata", {})
        original_run_id = rerun_meta.get("original_run_id")
        rerun_from_step = rerun_meta.get("from_step")
        if original_run_id:
            async with sf() as session:
                orig_result = await session.execute(
                    select(PipelineRun)
                    .where(PipelineRun.id == original_run_id)
                    .options(selectinload(PipelineRun.steps))
                )
                original_run = orig_result.scalar_one_or_none()

        if original_run and rerun_from_step:
            orig_sorted = sorted(original_run.steps, key=lambda s: s.step_index)
            prior_seen_groups: set[str] = set()
            for step in orig_sorted:
                if step.step_name == rerun_from_step:
                    break
                parsed = json.loads(step.parsed_output) if step.parsed_output else {}
                pretty = json.dumps(parsed, indent=2) if parsed else ""
                verifier_parsed = json.loads(step.verifier_output) if step.verifier_output else {}
                verifier_pretty = json.dumps(verifier_parsed, indent=2) if verifier_parsed else ""
                verifier_label = "Challenger" if step.verifier_mode == "challenger" else "Verifier"
                trace = json.loads(step.agent_trace) if step.agent_trace else []
                if "/" in step.step_name:
                    group_name, branch_name = step.step_name.split("/", 1)
                    if group_name not in prior_seen_groups:
                        prior_seen_groups.add(group_name)
                        rerun_prior_items.append({"type": "group_header", "name": group_name})
                    rerun_prior_items.append({
                        "type": "branch", "group": group_name, "name": branch_name,
                        "step": step, "parsed": parsed, "pretty": pretty,
                        "verifier_pretty": verifier_pretty, "verifier_label": verifier_label,
                        "trace": trace,
                    })
                else:
                    rerun_prior_items.append({
                        "type": "step", "name": step.step_name,
                        "step": step, "parsed": parsed, "pretty": pretty,
                        "verifier_pretty": verifier_pretty, "verifier_label": verifier_label,
                        "trace": trace,
                    })

    feedback = None
    async with sf() as session:
        result = await session.execute(select(RunFeedback).where(RunFeedback.run_id == run_id))
        feedback = result.scalar_one_or_none()

    from .executors.human import get_pending_for_run
    pending_approvals = get_pending_for_run(run_id) if run.status == "running" else []

    return templates.TemplateResponse(request, "run_detail.html", {
        "run": run,
        "display_items": display_items,
        "rerun_prior_items": rerun_prior_items,
        "rerun_from_step": rerun_from_step,
        "original_run": original_run,
        "normalised": normalised,
        "run_log": run_log,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "feedback": feedback,
        "pending_approvals": pending_approvals,
        "active_page": "runs",
    })


@router.get("/runs/{run_id}/stream")
async def ui_run_stream(request: Request, run_id: str):
    """Server-Sent Events endpoint for live run tailing.

    - Run still in progress: streams events as they happen, closes on completion.
    - Run already finished: replays the stored log and closes immediately.
    """
    sf = get_session_factory()
    async with sf() as session:
        run = await session.get(PipelineRun, run_id)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    async def _generate():
        # Flush connection immediately so the browser fires onopen.
        yield ": connected\n\n"

        # Subscribe and snapshot history atomically (no await between these two
        # calls, so no events can slip through the gap).
        q, history = run_events.subscribe(run_id)
        try:
            # Re-read run status now that we're subscribed.
            async with sf() as session:
                fresh_run = await session.get(PipelineRun, run_id)

            if not fresh_run or fresh_run.status != "running":
                # Run already finished — replay from the DB log then close.
                status = fresh_run.status if fresh_run else "unknown"
                if fresh_run and fresh_run.logs:
                    for event in json.loads(fresh_run.logs):
                        yield f"data: {json.dumps(event)}\n\n"
                yield f"data: {json.dumps({'type': 'run_complete', 'status': status})}\n\n"
                return

            # Run still active — replay everything that happened before we
            # connected, then stream new events from the queue.
            for event in history:
                yield f"data: {json.dumps(event)}\n\n"

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "run_complete":
                    break
        finally:
            run_events.unsubscribe(run_id, q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _agents_in_pipeline(p: PipelineConfig) -> list[str]:
    """Distinct `executor:agent` identifiers referenced by a pipeline's steps, read straight from config.

    Prefixed with the executor (openclaw/gateway/etc.) for the same reason the Agent
    Library page namespaces agents that way — the same agent name can exist on
    multiple backends and they are not the same agent (see README §Agent Library).

    Reliable and available even for pipelines that have never run — unlike the model
    actually used, which depends on the agent's own backend config and is only known
    from run history (see ui_pipelines for why models are deliberately left off).
    """
    agents: set[str] = set()
    for step in p.steps:
        if isinstance(step, ParallelGroupConfig):
            for s in step.parallel.steps:
                a = s.executor_config.get("agent")
                if a:
                    agents.add(f"{s.executor}:{a}")
        elif isinstance(step, FanOutGroupConfig):
            a = step.fan_out.executor_config.get("agent")
            if a:
                agents.add(f"{step.fan_out.executor}:{a}")
        else:
            a = step.executor_config.get("agent")
            if a:
                agents.add(f"{step.executor}:{a}")
    return sorted(agents)


@router.get("/pipelines", response_class=HTMLResponse)
async def ui_pipelines(request: Request, tag: str | None = None, agent: str | None = None):
    all_pipelines = getattr(request.app.state, "pipelines", [])
    all_tags = sorted({t for p in all_pipelines for t in p.tags})
    agents_by_pipeline = {p.name: _agents_in_pipeline(p) for p in all_pipelines}
    all_agents = sorted({a for agents in agents_by_pipeline.values() for a in agents})

    pipelines = all_pipelines
    if tag:
        pipelines = [p for p in pipelines if tag in p.tags]
    if agent:
        pipelines = [p for p in pipelines if agent in agents_by_pipeline.get(p.name, [])]

    sf = get_session_factory()

    async with sf() as session:
        rows = await session.execute(
            select(PipelineRun)
            .order_by(PipelineRun.triggered_at.desc())
        )
        all_runs = rows.scalars().all()

        fb_rows = await session.execute(
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count().label("n"))
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        )
        feedback_by_pipeline: dict[str, dict[str, int]] = {}
        for name, outcome, n in fb_rows.all():
            if name not in feedback_by_pipeline:
                feedback_by_pipeline[name] = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
            feedback_by_pipeline[name][outcome] = n
            feedback_by_pipeline[name]["total"] += n

        token_rows = await session.execute(
            select(
                PipelineRun.pipeline_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineStep, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineRun.pipeline_name)
        )
        tokens_by_pipeline: dict[str, tuple[int, int]] = {
            name: (inp, out) for name, inp, out in token_rows.all()
        }

    last_run: dict[str, datetime] = {}
    last_status: dict[str, str] = {}
    run_counts: dict[str, int] = {}
    failed_counts: dict[str, int] = {}
    for run in all_runs:
        run_counts[run.pipeline_name] = run_counts.get(run.pipeline_name, 0) + 1
        if run.status == "failed":
            failed_counts[run.pipeline_name] = failed_counts.get(run.pipeline_name, 0) + 1
        if run.pipeline_name not in last_run:
            last_run[run.pipeline_name] = run.triggered_at
            last_status[run.pipeline_name] = run.status

    success_rate_by_pipeline: dict[str, int] = {}
    avg_tokens_by_pipeline: dict[str, tuple[int, int]] = {}
    for name, n in run_counts.items():
        success_rate_by_pipeline[name] = round((n - failed_counts.get(name, 0)) / n * 100)
        inp, out = tokens_by_pipeline.get(name, (0, 0))
        if inp or out:
            avg_tokens_by_pipeline[name] = (round(inp / n), round(out / n))

    # ── Header stat cards — scoped to the filtered pipeline set ─────────────
    filtered_names = [p.name for p in pipelines]
    scheduled_count = sum(1 for p in pipelines if p.schedule)

    total_runs = sum(run_counts.get(name, 0) for name in filtered_names)
    total_failed = sum(failed_counts.get(name, 0) for name in filtered_names)
    overall_success_rate = (
        round((total_runs - total_failed) / total_runs * 100) if total_runs else None
    )

    fb_correct = sum(feedback_by_pipeline.get(name, {}).get("correct", 0) for name in filtered_names)
    fb_total = sum(feedback_by_pipeline.get(name, {}).get("total", 0) for name in filtered_names)
    overall_accuracy_pct = round(fb_correct / fb_total * 100) if fb_total else None

    agents_in_view_count = len({a for name in filtered_names for a in agents_by_pipeline.get(name, [])})

    return templates.TemplateResponse(request, "pipelines.html", {
        "pipelines": pipelines,
        "last_run": last_run,
        "last_status": last_status,
        "run_counts": run_counts,
        "feedback_by_pipeline": feedback_by_pipeline,
        "agents_by_pipeline": agents_by_pipeline,
        "success_rate_by_pipeline": success_rate_by_pipeline,
        "avg_tokens_by_pipeline": avg_tokens_by_pipeline,
        "all_tags": all_tags,
        "all_agents": all_agents,
        "selected_tag": tag or "",
        "selected_agent": agent or "",
        "total_pipeline_count": len(pipelines),
        "scheduled_count": scheduled_count,
        "overall_success_rate": overall_success_rate,
        "overall_accuracy_pct": overall_accuracy_pct,
        "agents_in_view_count": agents_in_view_count,
        "active_page": "pipelines",
    })


@router.get("/insights", response_class=HTMLResponse)
async def ui_insights_overview(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = select(PipelineRun.status, func.count().label("n")).group_by(PipelineRun.status)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        status_counts: dict[str, int] = dict(rows.all())

        q = select(PipelineRun.team, func.count().label("n")).group_by(PipelineRun.team)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        team_counts = rows.all()

        q = (
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
        q = (
            select(PipelineStep.model, PipelineStep.agent_trace)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.executor == "gateway")
            .where(PipelineStep.agent_trace.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        model_traces = rows.all()

        q = (
            select(PipelineRun.pipeline_name, func.count().label("n"))
            .group_by(PipelineRun.pipeline_name)
            .order_by(func.count().desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        pipeline_run_counts = rows.all()

        q = (
            select(PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineStep.agent)
            .order_by(func.count().desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        agent_step_counts = rows.all()

        q = (
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

        q = select(RunFeedback.outcome, func.count().label("n")).group_by(RunFeedback.outcome)
        if cutoff:
            q = q.where(RunFeedback.submitted_at >= cutoff)
        rows = await session.execute(q)
        feedback_by_outcome: dict[str, int] = dict(rows.all())

        # Raw rows for timeseries bucketing — one consolidated fetch per level.
        q = select(PipelineRun.triggered_at, PipelineRun.status, PipelineRun.team, PipelineRun.pipeline_name)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        run_ts_rows = rows.all()

        q = (
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
        [_STATUS_HEX.get(s, "#71717a") for s in status_labels],
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
    status_ts  = _build_ts(run_ts_rows,  dim_fn=lambda r: r[1],  **ts_kw)   # status
    team_ts    = _build_ts(run_ts_rows,  dim_fn=lambda r: r[2] or "Unattributed", **ts_kw)   # team
    pipeline_ts = _build_ts(run_ts_rows, dim_fn=lambda r: r[3],  **ts_kw)   # pipeline
    agent_ts   = _build_ts(step_ts_rows, dim_fn=lambda r: r[1] or "—", **ts_kw)   # agent
    llm_ts     = _build_ts(
        step_ts_rows,
        dim_fn=lambda r: r[2] or "Unknown model",
        val_fn=lambda r: sum(1 for e in (json.loads(r[5]) if r[5] else []) if e.get("type") == "llm_call"),
        **ts_kw,
    )
    tokens_ts  = _build_ts(
        [r for r in step_ts_rows if r[3] is not None],
        dim_fn=lambda r: r[2] or "Unknown model",
        val_fn=lambda r: (r[3] or 0) + (r[4] or 0),
        **ts_kw,
    )

    feedback_total = sum(feedback_by_outcome.values())
    feedback_correct = feedback_by_outcome.get("correct", 0)

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
        "active_page": "insights_overview",
    })


@router.get("/insights/pipelines", response_class=HTMLResponse)
async def ui_insights_pipelines(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = select(PipelineRun.pipeline_name, func.count().label("n")).group_by(PipelineRun.pipeline_name)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        run_counts = dict(rows.all())

        q = (
            select(PipelineRun.pipeline_name, func.count().label("n"))
            .where(PipelineRun.status == "failed")
            .group_by(PipelineRun.pipeline_name)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        failed_counts = dict(rows.all())

        q = (
            select(
                PipelineRun.pipeline_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None))
            .group_by(PipelineRun.pipeline_name)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        token_totals = {name: (i, o) for name, i, o in rows.all()}

        q = (
            select(PipelineRun.pipeline_name, PipelineRun.team)
            .distinct()
            .group_by(PipelineRun.pipeline_name, PipelineRun.team)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        teams_by_pipeline: dict[str, list[str]] = defaultdict(list)
        for name, team in rows.all():
            teams_by_pipeline[name].append(team or "Unattributed")

        # No duration column on pipeline_runs — compute from timestamps in Python
        # (same approach _format_duration already uses) rather than a cross-dialect
        # SQL timestamp-diff. Only terminal runs (completed_at set) count.
        q = (
            select(PipelineRun.pipeline_name, PipelineRun.triggered_at, PipelineRun.completed_at)
            .where(PipelineRun.completed_at.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        durations_by_pipeline: dict[str, list[float]] = defaultdict(list)
        for name, triggered_at, completed_at in rows.all():
            secs = (completed_at.replace(tzinfo=None) - triggered_at.replace(tzinfo=None)).total_seconds()
            if secs >= 0:
                durations_by_pipeline[name].append(secs)

        # Status breakdown per pipeline (for drilldown status bar)
        q = (
            select(PipelineRun.pipeline_name, PipelineRun.status, func.count().label("n"))
            .group_by(PipelineRun.pipeline_name, PipelineRun.status)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        status_counts_by_pipeline: dict[str, dict[str, int]] = defaultdict(dict)
        for name, status, n in rows.all():
            status_counts_by_pipeline[name][status] = n

        # All runs in range — used for per-pipeline timeseries and recent-runs list
        q = (
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

        # Feedback per pipeline scoped to the same time range (via run date)
        fb_q = (
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        )
        if cutoff:
            fb_q = fb_q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(fb_q)
        feedback_raw: dict[str, dict[str, int]] = {}
        for name, outcome, n in rows.all():
            if name not in feedback_raw:
                feedback_raw[name] = {"correct": 0, "partial": 0, "incorrect": 0}
            feedback_raw[name][outcome] = n

    # ── Per-pipeline aggregates ───────────────────────────────────────────────

    avg_duration_by_pipeline = {
        name: sum(secs) / len(secs) for name, secs in durations_by_pipeline.items() if secs
    }

    now = utc_now()
    resolution = _ts_resolution(time_range)
    if all_runs_raw:
        oldest = min(r.triggered_at.replace(tzinfo=None) for r in all_runs_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = _ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_pipeline: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_pipeline: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_pipeline: dict[str, list] = defaultdict(list)

    for row in all_runs_raw:
        bucket = _ts_bucket(row.triggered_at, resolution)
        runs_by_bucket_pipeline[row.pipeline_name][bucket] += 1
        if row.completed_at:
            secs = (row.completed_at.replace(tzinfo=None) - row.triggered_at.replace(tzinfo=None)).total_seconds()
            if secs >= 0:
                durations_by_bucket_pipeline[row.pipeline_name][bucket].append(secs)
        if len(recent_by_pipeline[row.pipeline_name]) < 5:
            recent_by_pipeline[row.pipeline_name].append(row)

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
                dur_str = _format_seconds(secs)
            recent.append({
                "id": str(r.id),
                "status": r.status,
                "ago": _format_ago(r.triggered_at),
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
            "avg_duration": _format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
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

    all_durations_flat = [s for sl in durations_by_pipeline.values() for s in sl]
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
        "drilldown_data": drilldown_data,
        "insights_feedback_total": insights_feedback_total,
        "insights_accuracy_pct": insights_accuracy_pct,
        "active_page": "insights_pipelines",
    })


@router.get("/insights/agents", response_class=HTMLResponse)
async def ui_insights_agents(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = (
            select(
                PipelineStep.executor, PipelineStep.agent, func.count().label("n"),
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
                func.avg(PipelineStep.duration_ms),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineStep.executor, PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        step_totals = rows.all()

        q = (
            select(PipelineStep.executor, PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.status == "failed")
            .group_by(PipelineStep.executor, PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        failed_counts = {(executor, agent): n for executor, agent, n in rows.all()}

        q = (
            select(PipelineStep.executor, PipelineStep.agent, PipelineStep.model)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.model.is_not(None))
            .distinct()
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        models_by_agent: dict[tuple[str, str], list[str]] = defaultdict(list)
        for executor, agent, model in rows.all():
            models_by_agent[(executor, agent)].append(model)

    agent_rows = []
    for executor, agent, step_count, input_tokens, output_tokens, avg_duration_ms in step_totals:
        failed = failed_counts.get((executor, agent), 0)
        success_rate = round((step_count - failed) / step_count * 100) if step_count else None
        agent_rows.append({
            "executor": executor,
            "agent": agent or "—",
            "step_count": step_count,
            "failed_count": failed,
            "success_rate": success_rate,
            "avg_duration_secs": (avg_duration_ms / 1000) if avg_duration_ms is not None else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "models": sorted(models_by_agent.get((executor, agent), [])),
        })
    agent_rows.sort(key=lambda r: r["step_count"], reverse=True)

    duration_chart_rows = sorted(
        (r for r in agent_rows if r["avg_duration_secs"] is not None),
        key=lambda r: r["avg_duration_secs"], reverse=True,
    )
    duration_chart = {
        "labels": [r["agent"] for r in duration_chart_rows],
        "data": [round(r["avg_duration_secs"]) for r in duration_chart_rows],
    }

    token_chart_rows = sorted(
        (r for r in agent_rows if r["input_tokens"] or r["output_tokens"]),
        key=lambda r: r["input_tokens"] + r["output_tokens"], reverse=True,
    )
    token_chart = {
        "labels": [r["agent"] for r in token_chart_rows],
        "input": [r["input_tokens"] for r in token_chart_rows],
        "output": [r["output_tokens"] for r in token_chart_rows],
    }

    return templates.TemplateResponse(request, "insights_agents.html", {
        "time_range": time_range,
        "range_label": range_label,
        "agent_rows": agent_rows,
        "duration_chart": duration_chart,
        "token_chart": token_chart,
        "active_page": "insights_agents",
    })


@router.get("/pipelines/{name}", response_class=HTMLResponse)
async def ui_pipeline_detail(request: Request, name: str):
    pipelines = getattr(request.app.state, "pipelines", [])
    pipeline = next((p for p in pipelines if p.name == name), None)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pipeline_dir = getattr(request.app.state, "pipeline_dir", "./pipelines")
    raw_yaml = _read_pipeline_yaml(pipeline_dir, name) or ""

    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.pipeline_name == name)
            .order_by(PipelineRun.triggered_at.desc())
            .limit(10)
        )
        recent_runs = rows.scalars().all()
        feedback_by_run = await _feedback_by_run_id(session, [r.id for r in recent_runs])

        rows = await session.execute(
            select(PipelineRun.status, func.count().label("n"))
            .where(PipelineRun.pipeline_name == name)
            .group_by(PipelineRun.status)
        )
        status_counts = dict(rows.all())

        rows = await session.execute(
            select(RunFeedback.outcome, func.count().label("n"))
            .where(RunFeedback.pipeline_name == name)
            .group_by(RunFeedback.outcome)
        )
        feedback_counts = dict(rows.all())

    feedback_total = sum(feedback_counts.values())

    return templates.TemplateResponse(request, "pipeline_detail.html", {
        "pipeline": pipeline,
        "raw_yaml": raw_yaml,
        "recent_runs": recent_runs,
        "feedback_by_run": feedback_by_run,
        "status_counts": status_counts,
        "total_runs": sum(status_counts.values()),
        "feedback_counts": feedback_counts,
        "feedback_total": feedback_total,
        "active_page": "pipelines",
    })


@router.get("/pipelines/{name}/feedback", response_class=HTMLResponse)
async def ui_pipeline_feedback(request: Request, name: str):
    pipelines = getattr(request.app.state, "pipelines", [])
    pipeline = next((p for p in pipelines if p.name == name), None)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            select(RunFeedback, PipelineRun.triggered_at, PipelineRun.status)
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(RunFeedback.pipeline_name == name)
            .order_by(RunFeedback.submitted_at.desc())
        )
        feedback_rows = rows.all()

        run_ids = [fb.run_id for fb, _, _ in feedback_rows]
        steps_by_run: dict[str, list] = defaultdict(list)
        if run_ids:
            step_rows = await session.execute(
                select(PipelineStep)
                .where(PipelineStep.run_id.in_(run_ids))
                .order_by(PipelineStep.step_index)
            )
            for step in step_rows.scalars().all():
                steps_by_run[step.run_id].append(step)

    # Build per-run records annotated with config fingerprint
    marked_runs = []
    for fb, triggered_at, run_status in feedback_rows:
        steps = steps_by_run.get(fb.run_id, [])
        fp = _config_fingerprint(steps) if steps else "unknown"
        desc = _config_description(steps) if steps else {"steps": [], "models": []}
        marked_runs.append({
            "run_id": fb.run_id,
            "outcome": fb.outcome,
            "notes": fb.notes,
            "submitted_at": fb.submitted_at,
            "triggered_at": triggered_at,
            "run_status": run_status,
            "fingerprint": fp,
            "config": desc,
        })

    # Group by config fingerprint
    config_groups: dict[str, dict] = {}
    for r in marked_runs:
        fp = r["fingerprint"]
        if fp not in config_groups:
            config_groups[fp] = {
                "fingerprint": fp,
                "config": r["config"],
                "runs": [],
                "correct": 0, "partial": 0, "incorrect": 0,
                "first_seen": r["triggered_at"],
                "last_seen": r["triggered_at"],
            }
        g = config_groups[fp]
        g["runs"].append(r)
        g[r["outcome"]] = g.get(r["outcome"], 0) + 1
        if r["triggered_at"] and (g["first_seen"] is None or r["triggered_at"] < g["first_seen"]):
            g["first_seen"] = r["triggered_at"]
        if r["triggered_at"] and (g["last_seen"] is None or r["triggered_at"] > g["last_seen"]):
            g["last_seen"] = r["triggered_at"]

    # Sort groups: most recently active first
    sorted_groups = sorted(
        config_groups.values(),
        key=lambda g: g["last_seen"] or datetime.min,
        reverse=True,
    )
    for g in sorted_groups:
        total = g["correct"] + g["partial"] + g["incorrect"]
        g["total"] = total
        g["correct_pct"] = round(g["correct"] / total * 100) if total else 0

    total_marked = len(marked_runs)
    total_correct = sum(1 for r in marked_runs if r["outcome"] == "correct")
    total_partial = sum(1 for r in marked_runs if r["outcome"] == "partial")
    total_incorrect = sum(1 for r in marked_runs if r["outcome"] == "incorrect")

    return templates.TemplateResponse(request, "pipeline_feedback.html", {
        "pipeline": pipeline,
        "marked_runs": marked_runs,
        "config_groups": sorted_groups,
        "total_marked": total_marked,
        "total_correct": total_correct,
        "total_partial": total_partial,
        "total_incorrect": total_incorrect,
        "active_page": "pipelines",
    })


@router.get("/schedules", response_class=HTMLResponse)
async def ui_schedules(request: Request):
    scheduler = getattr(request.app.state, "scheduler", None)
    pipelines = getattr(request.app.state, "pipelines", [])

    jobs = []
    if scheduler:
        for job in scheduler.get_jobs():
            if not job.id.startswith("pipeline:"):
                continue
            p = next((p for p in pipelines if p.name == job.name), None)
            jobs.append({
                "name": job.name,
                "cron": p.schedule.cron if p and p.schedule else "—",
                "summary": p.schedule.summary if p and p.schedule else "",
                "severity": p.schedule.severity if p and p.schedule else "",
                "next_run": job.next_run_time,
            })

    return templates.TemplateResponse(request, "schedules.html", {
        "jobs": jobs,
        "active_page": "schedules",
    })



# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_pipeline_yaml(pipeline_dir: str, pipeline_name: str) -> str | None:
    for path in glob.glob(os.path.join(pipeline_dir, "*.yaml")):
        try:
            with open(path) as f:
                content = f.read()
            data = yaml.safe_load(content)
            if isinstance(data, dict) and data.get("name") == pipeline_name:
                return content
        except Exception:
            continue
    return None


# ── Step library ─────────────────────────────────────────────────────────────

def _iter_all_raw_steps(steps: list):
    """Yield every step dict from a raw YAML steps list, including parallel inner steps."""
    for step in steps:
        if isinstance(step, dict) and "parallel" in step:
            yield from _iter_all_raw_steps(step["parallel"].get("steps", []))
        else:
            yield step


def _compute_step_usage(pipeline_dir: str, library: dict) -> dict[str, list[str]]:
    """Scan pipeline YAMLs to find which pipelines reference each library step."""
    usage: dict[str, list[str]] = {name: [] for name in library}
    for path in glob.glob(os.path.join(pipeline_dir, "*.yaml")):
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
            pipeline_name = raw.get("name", os.path.basename(path))
            for step in _iter_all_raw_steps(raw.get("steps", [])):
                use = step.get("use") if isinstance(step, dict) else None
                if use and use in usage:
                    usage[use].append(pipeline_name)
        except Exception:
            pass
    return usage


@router.get("/steps", response_class=HTMLResponse)
async def ui_steps(request: Request, tag: str | None = None):
    step_library: dict = getattr(request.app.state, "step_library", {})
    pipeline_dir: str = getattr(request.app.state, "pipeline_dir", "./pipelines")

    step_usage = _compute_step_usage(pipeline_dir, step_library)
    all_steps = sorted(step_library.values(), key=lambda s: s.get("name", ""))
    all_tags = sorted({t for s in all_steps for t in s.get("tags") or []})
    steps = [s for s in all_steps if tag in (s.get("tags") or [])] if tag else all_steps

    return templates.TemplateResponse(request, "steps.html", {
        "steps": steps,
        "step_usage": step_usage,
        "all_tags": all_tags,
        "selected_tag": tag or "",
        "active_page": "steps",
    })


# ── Agent pages ────────────────────────────────────────────────────────────

# Per-executor URLs — overridden at startup by configure() called from main.py lifespan.
# Defaults allow the service to start without config and fall back gracefully.
_pork_gateway_base: str = os.environ.get("PORK_GATEWAY_URL", "http://localhost:18780")
_openclaw_ws_url: str = "ws://127.0.0.1:18789/rpc"
_openclaw_enabled: bool = True  # set False when executors.openclaw is absent from config
_team_count: int = 0  # number of teams configured under auth.teams — see README §3b


def configure(openclaw_ws_url: str = "", pork_gateway_base: str = "", team_count: int = 0) -> None:
    """Set agent source URLs and team count from config.yaml values. Call from main.py lifespan."""
    global _openclaw_ws_url, _pork_gateway_base, _openclaw_enabled, _team_count
    _openclaw_enabled = bool(openclaw_ws_url)
    if openclaw_ws_url:
        _openclaw_ws_url = openclaw_ws_url
    if pork_gateway_base:
        _pork_gateway_base = pork_gateway_base
    _team_count = team_count


def _first_description_line(content: str | None) -> str | None:
    """Return the first non-heading paragraph from soul/description content."""
    if not content:
        return None
    paragraph: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if paragraph:
                break
            continue
        if stripped:
            paragraph.append(stripped)
        elif paragraph:
            break
    return " ".join(paragraph) or None


# ── Per-executor agent discovery ──────────────────────────────────────────────

async def _fetch_openclaw_agents() -> tuple[list[dict], str | None]:
    """Fetch agents list from the OpenClaw Gateway WS API (agents.list RPC)."""
    if not _openclaw_enabled:
        return [], None
    result = await gateway_call_safe("agents.list", {}, gateway_url=_openclaw_ws_url)
    if result is None:
        return [], f"Could not reach OpenClaw Gateway at {_openclaw_ws_url} — is it running?"
    return result.get("agents") or [], None


async def _fetch_pork_gateway_agents() -> tuple[list[dict], str | None]:
    """Fetch agents list from the P-Ork Gateway REST /agents endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{_pork_gateway_base}/agents")
            resp.raise_for_status()
            return resp.json().get("agents", []), None
    except Exception as exc:
        logger.debug("P-Ork Gateway /agents failed: %s", exc)
        return [], f"Could not reach P-Ork Gateway at {_pork_gateway_base} — is it running?"


async def _fetch_pork_gateway_mcp() -> tuple[dict, dict, str | None]:
    """Fetch the MCP tool registry + server status from the P-Ork Gateway REST API.

    GET /mcp/tools returns {server_name: [{name, registeredName, description, inputSchema}, ...]}
    GET /mcp/servers returns {server_name: {running, pid, restart_count}}
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            tools_resp, servers_resp = await asyncio.gather(
                client.get(f"{_pork_gateway_base}/mcp/tools"),
                client.get(f"{_pork_gateway_base}/mcp/servers"),
            )
            tools_resp.raise_for_status()
            servers_resp.raise_for_status()
            return tools_resp.json(), servers_resp.json(), None
    except Exception as exc:
        logger.debug("P-Ork Gateway MCP endpoints failed: %s", exc)
        return {}, {}, f"Could not reach P-Ork Gateway at {_pork_gateway_base} — is it running?"


async def _fetch_openclaw_agent_files(agent_id: str) -> dict[str, str | None]:
    """Fetch SOUL.md / TOOLS.md / IDENTITY.md from the OpenClaw Gateway WS API."""
    def _content(payload: dict | None) -> str | None:
        return ((payload or {}).get("file") or {}).get("content") or None

    soul_r, tools_r, identity_r = await asyncio.gather(
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "SOUL.md"}, gateway_url=_openclaw_ws_url),
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "TOOLS.md"}, gateway_url=_openclaw_ws_url),
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "IDENTITY.md"}, gateway_url=_openclaw_ws_url),
    )
    return {
        "soul": _content(soul_r),
        "tools": _content(tools_r),
        "identity": _content(identity_r),
    }


async def _fetch_pork_gateway_agent_files(agent_id: str) -> dict[str, str | None]:
    """Fetch soul and agent.yaml from the P-Ork Gateway REST API."""
    result: dict[str, str | None] = {"soul": None, "tools": None, "identity": None, "agent_file": None}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            soul_resp, agent_resp = await asyncio.gather(
                client.get(f"{_pork_gateway_base}/agents/{agent_id}/soul"),
                client.get(f"{_pork_gateway_base}/agents/{agent_id}/agent"),
            )
            if soul_resp.status_code == 200:
                result["soul"] = soul_resp.json().get("content") or soul_resp.text
            if agent_resp.status_code == 200:
                result["agent_file"] = agent_resp.json().get("content") or agent_resp.text
    except Exception:
        pass
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/agents", response_class=HTMLResponse)
async def ui_agents(request: Request):
    # Fetch from both backends concurrently
    (oc_agents, oc_error), (gw_agents, gw_error) = await asyncio.gather(
        _fetch_openclaw_agents(),
        _fetch_pork_gateway_agents(),
    )

    # Tag every entry with its executor source so the template and URL routing
    # can distinguish agents that share names across backends.
    all_agents: list[dict] = []
    for a in oc_agents:
        all_agents.append({**a, "executor": "openclaw"})
    for a in gw_agents:
        all_agents.append({**a, "executor": "gateway"})

    # Batch-fetch the first SOUL.md line for openclaw agents as a list-page preview.
    if oc_agents and _openclaw_enabled:
        oc_ids = [a.get("id") or a.get("name") for a in oc_agents]
        soul_results = await asyncio.gather(*[
            gateway_call_safe("agents.files.get", {"agentId": aid, "name": "SOUL.md"}, gateway_url=_openclaw_ws_url)
            for aid in oc_ids
        ], return_exceptions=True)
        for agent, soul_result in zip(all_agents[:len(oc_agents)], soul_results):
            if isinstance(soul_result, Exception) or soul_result is None:
                continue
            content = ((soul_result or {}).get("file") or {}).get("content")
            desc = _first_description_line(content)
            if desc:
                agent["description"] = desc

    # Batch-fetch soul descriptions for gateway agents.
    if gw_agents:
        gw_ids = [a.get("id") or a.get("name") for a in gw_agents]

        async def _fetch_gw_soul(agent_id: str) -> str | None:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{_pork_gateway_base}/agents/{agent_id}/soul")
                    if resp.status_code == 200:
                        return resp.json().get("content") or resp.text
            except Exception:
                pass
            return None

        gw_soul_results = await asyncio.gather(*[_fetch_gw_soul(aid) for aid in gw_ids], return_exceptions=True)
        gw_agent_entries = all_agents[len(oc_agents):]
        for agent, soul_content in zip(gw_agent_entries, gw_soul_results):
            if isinstance(soul_content, Exception) or not soul_content:
                continue
            desc = _first_description_line(soul_content)
            if desc:
                agent["description"] = desc

    # Collect non-None error messages
    gateway_errors = [e for e in [oc_error, gw_error] if e]

    # DB stats — agent column now stores "executor:name" prefixed values.
    # Group by the full prefixed key so openclaw:X and gateway:X are counted separately.
    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            select(
                PipelineStep.agent,
                PipelineStep.status,
                func.count().label("n"),
                func.max(PipelineStep.executed_at).label("last_run"),
            )
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent, PipelineStep.status)
        )
        step_rows = rows.all()

    agent_stats: dict[str, dict] = {}
    for row in step_rows:
        s = agent_stats.setdefault(
            row.agent,
            {"succeeded": 0, "failed": 0, "total": 0, "last_run": None, "success_rate": None},
        )
        s["total"] += row.n
        if row.status == "failed":
            s["failed"] += row.n
        else:
            s["succeeded"] += row.n
        if row.last_run and (s["last_run"] is None or row.last_run > s["last_run"]):
            s["last_run"] = row.last_run

    for s in agent_stats.values():
        if s["total"] > 0:
            s["success_rate"] = round(s["succeeded"] / s["total"] * 100, 1)

    # If both backends are unreachable, synthesise stub entries from DB history
    # so the page still renders something useful.  Only synthesise for rows that
    # carry a recognisable "executor:agent" prefix — bare legacy entries are ignored.
    if not all_agents:
        seen: set[str] = set()
        for key in sorted(agent_stats.keys()):
            if ":" in key:
                executor, _, aid = key.partition(":")
                stub_key = f"{executor}:{aid}"
                if stub_key not in seen:
                    seen.add(stub_key)
                    all_agents.append({"id": aid, "name": aid, "executor": executor})

    return templates.TemplateResponse(request, "agents.html", {
        "agents": all_agents,
        "agent_stats": agent_stats,
        "gateway_errors": gateway_errors,
        "active_page": "agents",
    })


@router.get("/agents/{executor}/{agent_id}", response_class=HTMLResponse)
async def ui_agent_detail(request: Request, executor: str, agent_id: str):
    prefixed_key = f"{executor}:{agent_id}"

    # Fetch live config + file contents from the appropriate backend
    if executor == "openclaw":
        agents_raw, _ = await _fetch_openclaw_agents()
        agent_files = await _fetch_openclaw_agent_files(agent_id)
    else:  # gateway (or any future executor with REST discovery)
        agents_raw, _ = await _fetch_pork_gateway_agents()
        agent_files = await _fetch_pork_gateway_agent_files(agent_id)

    agent_config = next(
        (a for a in agents_raw if a.get("name") == agent_id or a.get("id") == agent_id),
        None,
    )

    # DB run stats scoped to this executor:agent pair
    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            select(
                PipelineStep.model,
                PipelineStep.status,
                func.count().label("n"),
                func.max(PipelineStep.executed_at).label("last_run"),
            )
            .where(PipelineStep.agent == prefixed_key)
            .group_by(PipelineStep.model, PipelineStep.status)
        )
        step_rows = rows.all()

    model_stats: dict[str, dict] = {}
    for row in step_rows:
        m = row.model or "unknown"
        s = model_stats.setdefault(
            m,
            {"succeeded": 0, "failed": 0, "total": 0, "last_run": None, "success_rate": None},
        )
        s["total"] += row.n
        if row.status == "failed":
            s["failed"] += row.n
        else:
            s["succeeded"] += row.n
        if row.last_run and (s["last_run"] is None or row.last_run > s["last_run"]):
            s["last_run"] = row.last_run

    for s in model_stats.values():
        if s["total"] > 0:
            s["success_rate"] = round(s["succeeded"] / s["total"] * 100, 1)

    return templates.TemplateResponse(request, "agent_detail.html", {
        "agent_id": agent_id,
        "executor": executor,
        "agent_config": agent_config,
        "soul": agent_files["soul"],
        "tools": agent_files["tools"],
        "identity": agent_files["identity"],
        "agent_file": agent_files.get("agent_file"),
        "model_stats": model_stats,
        "active_page": "agents",
    })


# ── MCP tools ─────────────────────────────────────────────────────────────────

@router.get("/mcp", response_class=HTMLResponse)
async def ui_mcp(request: Request):
    """Browse the MCP tool registry exposed by the P-Ork Gateway.

    OpenClaw isn't included here — it has no REST endpoint for tool
    introspection (the gateway executor is the only one that exposes
    GET /mcp/tools and GET /mcp/servers).
    """
    tools_by_server, servers_status, error = await _fetch_pork_gateway_mcp()

    servers = []
    for name in sorted(set(tools_by_server) | set(servers_status)):
        status = servers_status.get(name, {})
        servers.append({
            "name": name,
            "tools": sorted(tools_by_server.get(name, []), key=lambda t: t.get("name", "")),
            "running": status.get("running"),
            "pid": status.get("pid"),
            "restart_count": status.get("restart_count", 0),
        })

    return templates.TemplateResponse(request, "mcp.html", {
        "servers": servers,
        "gateway_error": error,
        "server_count": len(servers),
        "total_tools": sum(len(s["tools"]) for s in servers),
        "active_page": "mcp",
    })


# ── Human-in-the-loop approvals ────────────────────────────────────────────────
#
# /approvals is a normal dashboard page (sidebar chrome) listing every pending
# approval regardless of channel — a universal fallback so a team isn't stuck if
# their primary chat channel (Slack/Telegram) is unreachable. /approvals/{token} is
# a standalone (no sidebar) page reached via a direct token link — used by the
# Teams approval channel, which posts this link instead of an in-chat button
# since Teams interactive cards need a public Bot Framework callback endpoint
# this deployment doesn't expose. See executors/human.py TeamsApprovalChannel.

@router.get("/approvals", response_class=HTMLResponse)
async def ui_approvals_list(request: Request):
    from .executors.human import list_pending

    return templates.TemplateResponse(request, "approvals_list.html", {
        "pending": list_pending(),
        "active_page": "approvals",
    })


@router.get("/approvals/{token}", response_class=HTMLResponse)
async def ui_approval(request: Request, token: str):
    from .executors.human import get_pending_meta

    meta = get_pending_meta(token)
    if meta is None:
        return templates.TemplateResponse(request, "approval.html", {
            "state": "not_found",
            "token": token,
        })

    return templates.TemplateResponse(request, "approval.html", {
        "state": "pending",
        "token": token,
        "meta": meta,
    })


@router.post("/approvals/{token}/approve", response_class=HTMLResponse)
async def ui_approval_approve(request: Request, token: str):
    return _decide(request, token, approved=True)


@router.post("/approvals/{token}/reject", response_class=HTMLResponse)
async def ui_approval_reject(request: Request, token: str):
    return _decide(request, token, approved=False)


def _decide(request: Request, token: str, approved: bool):
    from .executors.human import resolve_approval

    if not resolve_approval(token, approved):
        return templates.TemplateResponse(request, "approval.html", {
            "state": "not_found",
            "token": token,
        })

    return templates.TemplateResponse(request, "approval.html", {
        "state": "decided",
        "token": token,
        "decision": "approve" if approved else "reject",
    })
