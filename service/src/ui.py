import asyncio
import glob
import json
import logging
import os
from datetime import datetime, timedelta

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from .db.database import get_session_factory
from .db.models import PipelineRun, PipelineStep
from .gateway import gateway_call_safe
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


def _source_label(source: str) -> str:
    return {"alertmanager": "Alertmanager", "scheduler": "Scheduler", "generic": "Generic"}.get(
        source, source
    )


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


templates.env.filters["to_yaml"] = _to_yaml
templates.env.filters["tojson"] = _to_json
templates.env.filters["format_number"] = lambda n: f"{int(n):,}"
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
    non_terminal_24h = counts_24h.get("running", 0) + counts_24h.get("interrupted", 0)
    terminal_24h = total_24h - non_terminal_24h
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
    team: str | None = None,
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
        if team:
            q = q.where(PipelineRun.team == team)
        rows = await session.execute(q.limit(limit).offset(offset))
        runs = rows.scalars().all()

        rows = await session.execute(
            select(PipelineRun.pipeline_name).distinct().order_by(PipelineRun.pipeline_name)
        )
        pipeline_names = [r[0] for r in rows.all()]

        rows = await session.execute(
            select(PipelineRun.team).distinct().where(PipelineRun.team.is_not(None)).order_by(PipelineRun.team)
        )
        team_names = [r[0] for r in rows.all()]

    return templates.TemplateResponse(request, "runs.html", {
        "runs": runs,
        "pipeline_names": pipeline_names,
        "team_names": team_names,
        "selected_status": status or "",
        "selected_pipeline": pipeline or "",
        "selected_team": team or "",
        "limit": limit,
        "offset": offset,
        "statuses": ["completed", "running", "escalated", "aborted", "failed", "stopped", "interrupted"],
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

    return templates.TemplateResponse(request, "run_detail.html", {
        "run": run,
        "display_items": display_items,
        "normalised": normalised,
        "run_log": run_log,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
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
async def ui_steps(request: Request):
    step_library: dict = getattr(request.app.state, "step_library", {})
    pipeline_dir: str = getattr(request.app.state, "pipeline_dir", "./pipelines")

    step_usage = _compute_step_usage(pipeline_dir, step_library)
    steps = sorted(step_library.values(), key=lambda s: s.get("name", ""))

    return templates.TemplateResponse(request, "steps.html", {
        "steps": steps,
        "step_usage": step_usage,
        "active_page": "steps",
    })


# ── Agent pages ────────────────────────────────────────────────────────────

# Per-executor URLs — overridden at startup by configure() called from main.py lifespan.
# Defaults allow the service to start without config and fall back gracefully.
_pork_gateway_base: str = os.environ.get("PORK_GATEWAY_URL", "http://localhost:18780")
_openclaw_ws_url: str = "ws://127.0.0.1:18789/rpc"


def configure(openclaw_ws_url: str = "", pork_gateway_base: str = "") -> None:
    """Set agent source URLs from config.yaml values. Call from main.py lifespan."""
    global _openclaw_ws_url, _pork_gateway_base
    if openclaw_ws_url:
        _openclaw_ws_url = openclaw_ws_url
    if pork_gateway_base:
        _pork_gateway_base = pork_gateway_base


# ── Per-executor agent discovery ──────────────────────────────────────────────

async def _fetch_openclaw_agents() -> tuple[list[dict], str | None]:
    """Fetch agents list from the OpenClaw Gateway WS API (agents.list RPC)."""
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
