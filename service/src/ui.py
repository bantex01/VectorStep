import asyncio
import glob
import json
import logging
import os
from datetime import datetime, timedelta

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from .db.database import get_session_factory
from .db.models import PipelineRun, PipelineStep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui")

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "..", "templates")
)


# ── Template helpers ──────────────────────────────────────────────────────────

def _status_classes(status: str) -> str:
    return {
        "completed": "bg-green-950 text-green-400 ring-green-800",
        "running":   "bg-blue-950 text-blue-400 ring-blue-800",
        "escalated": "bg-amber-950 text-amber-400 ring-amber-800",
        "aborted":   "bg-orange-950 text-orange-400 ring-orange-800",
        "failed":    "bg-red-950 text-red-400 ring-red-800",
        "stopped":   "bg-purple-950 text-purple-400 ring-purple-800",
    }.get(status or "", "bg-zinc-800 text-zinc-400 ring-zinc-600")


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
    secs = int((datetime.utcnow() - dt.replace(tzinfo=None)).total_seconds())
    if secs < 5:
        return "just now"
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _source_label(source: str) -> str:
    return {"alertmanager": "Alertmanager", "scheduler": "Scheduler", "generic": "Generic"}.get(
        source, source
    )


templates.env.globals.update({
    "status_classes": _status_classes,
    "confidence_bar_color": _confidence_bar_color,
    "format_duration": _format_duration,
    "format_ago": _format_ago,
    "source_label": _source_label,
})


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    sf = get_session_factory()
    cutoff_24h = datetime.utcnow() - timedelta(hours=24)
    cutoff_7d  = datetime.utcnow() - timedelta(days=7)

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
            select(PipelineRun.source, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_24h)
            .group_by(PipelineRun.source)
        )
        source_counts = dict(rows.all())

        rows = await session.execute(
            select(PipelineRun)
            .order_by(PipelineRun.triggered_at.desc())
            .limit(15)
        )
        recent_runs = rows.scalars().all()

    total_24h = sum(counts_24h.values())
    terminal_24h = total_24h - counts_24h.get("running", 0)
    success_rate = (
        round((terminal_24h - counts_24h.get("failed", 0)) / terminal_24h * 100)
        if terminal_24h > 0 else None
    )
    pipelines = getattr(request.app.state, "pipelines", [])

    return templates.TemplateResponse(request, "dashboard.html", {
        "counts_24h": counts_24h,
        "counts_all": counts_all,
        "total_24h": total_24h,
        "success_rate": success_rate,
        "pipeline_activity": pipeline_activity,
        "source_counts": source_counts,
        "recent_runs": recent_runs,
        "pipeline_count": len(pipelines),
        "scheduled_count": sum(1 for p in pipelines if p.schedule),
        "active_page": "dashboard",
    })


@router.get("/runs", response_class=HTMLResponse)
async def ui_runs(
    request: Request,
    status: str | None = None,
    pipeline: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    sf = get_session_factory()
    async with sf() as session:
        q = select(PipelineRun).order_by(PipelineRun.triggered_at.desc())
        if status:
            q = q.where(PipelineRun.status == status)
        if pipeline:
            q = q.where(PipelineRun.pipeline_name == pipeline)
        rows = await session.execute(q.limit(limit).offset(offset))
        runs = rows.scalars().all()

        rows = await session.execute(
            select(PipelineRun.pipeline_name).distinct().order_by(PipelineRun.pipeline_name)
        )
        pipeline_names = [r[0] for r in rows.all()]

    return templates.TemplateResponse(request, "runs.html", {
        "runs": runs,
        "pipeline_names": pipeline_names,
        "selected_status": status or "",
        "selected_pipeline": pipeline or "",
        "limit": limit,
        "offset": offset,
        "statuses": ["completed", "running", "escalated", "aborted", "failed", "stopped"],
        "active_page": "runs",
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
            })

    normalised = json.loads(run.normalised_context) if run.normalised_context else {}

    return templates.TemplateResponse(request, "run_detail.html", {
        "run": run,
        "display_items": display_items,
        "normalised": normalised,
        "active_page": "runs",
    })


@router.get("/pipelines", response_class=HTMLResponse)
async def ui_pipelines(request: Request):
    pipelines = getattr(request.app.state, "pipelines", [])
    sf = get_session_factory()

    async with sf() as session:
        rows = await session.execute(
            select(PipelineRun)
            .order_by(PipelineRun.triggered_at.desc())
        )
        all_runs = rows.scalars().all()

    last_run: dict[str, datetime] = {}
    last_status: dict[str, str] = {}
    run_counts: dict[str, int] = {}
    for run in all_runs:
        run_counts[run.pipeline_name] = run_counts.get(run.pipeline_name, 0) + 1
        if run.pipeline_name not in last_run:
            last_run[run.pipeline_name] = run.triggered_at
            last_status[run.pipeline_name] = run.status

    return templates.TemplateResponse(request, "pipelines.html", {
        "pipelines": pipelines,
        "last_run": last_run,
        "last_status": last_status,
        "run_counts": run_counts,
        "active_page": "pipelines",
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

        rows = await session.execute(
            select(PipelineRun.status, func.count().label("n"))
            .where(PipelineRun.pipeline_name == name)
            .group_by(PipelineRun.status)
        )
        status_counts = dict(rows.all())

    return templates.TemplateResponse(request, "pipeline_detail.html", {
        "pipeline": pipeline,
        "raw_yaml": raw_yaml,
        "recent_runs": recent_runs,
        "status_counts": status_counts,
        "total_runs": sum(status_counts.values()),
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


# ── Agent pages ────────────────────────────────────────────────────────────

# P-Ork Gateway REST URL (config-driven, with fallback)
_GATEWAY_BASE = os.environ.get("PORK_GATEWAY_URL", "http://localhost:18780")


async def _gateway_agents() -> tuple[list[dict], str | None]:
    """Fetch agents list from the P-Ork Gateway REST /agents endpoint.
    Returns (agents_list, error_message_or_None).
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{_GATEWAY_BASE}/agents")
            resp.raise_for_status()
            return resp.json().get("agents", []), None
    except Exception as exc:
        logger.debug("gateway /agents failed: %s", exc)
        return [], f"Could not reach P-Ork Gateway at {_GATEWAY_BASE} — is it running?"


async def _gateway_agent_files(agent_id: str) -> dict[str, str | None]:
    """Fetch agent soul.md from the P-Ork Gateway REST /agents/{id}/soul endpoint."""
    result = {"soul": None, "tools": None, "identity": None}
    # Gateway only exposes /agents and /agents/{name}/soul currently
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{_GATEWAY_BASE}/agents/{agent_id}/soul")
            if resp.status_code == 200:
                result["soul"] = resp.json().get("content") or resp.text
    except Exception:
        pass
    return result


@router.get("/agents", response_class=HTMLResponse)
async def ui_agents(request: Request):
    agents_raw, gateway_error = await _gateway_agents()

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

    # If gateway is down, synthesise stub entries from DB history so page still works
    if not agents_raw and agent_stats:
        agents_raw = [{"id": aid, "name": aid} for aid in sorted(agent_stats.keys())]

    return templates.TemplateResponse(request, "agents.html", {
        "agents": agents_raw,
        "agent_stats": agent_stats,
        "gateway_error": gateway_error,
        "active_page": "agents",
    })


@router.get("/agents/{agent_id}", response_class=HTMLResponse)
async def ui_agent_detail(request: Request, agent_id: str):
    # Fetch agents list + soul from gateway
    agents_raw, _ = await _gateway_agents()
    agent_files = await _gateway_agent_files(agent_id)

    agent_config = next(
        (a for a in agents_raw if a.get("name") == agent_id or a.get("id") == agent_id),
        None,
    )

    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            select(
                PipelineStep.model,
                PipelineStep.status,
                func.count().label("n"),
                func.max(PipelineStep.executed_at).label("last_run"),
            )
            .where(PipelineStep.agent == agent_id)
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
        "agent_config": agent_config,
        "soul": agent_files["soul"],
        "tools": agent_files["tools"],
        "identity": agent_files["identity"],
        "model_stats": model_stats,
        "active_page": "agents",
    })
