from fastapi import APIRouter
from ..analytics import _production_only
from ..analytics import get_agent_versions as _get_agent_versions
from ..db.database import get_session_factory
from ..db.models import PipelineRun
from ..db.models import PipelineStep
from ..gateway import gateway_call_safe
from ..utils import utc_now
from collections import Counter
from collections import defaultdict
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from sqlalchemy import func
from sqlalchemy import select
import asyncio
import httpx
import json
import yaml

from . import helpers
from .helpers import templates


router = APIRouter()


# --- lines 3996-4015 ---
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


# --- lines 4187-4203 ---
async def _fetch_openclaw_agent_files(agent_id: str) -> dict[str, str | None]:
    """Fetch SOUL.md / TOOLS.md / IDENTITY.md from the OpenClaw Gateway WS API."""
    def _content(payload: dict | None) -> str | None:
        return ((payload or {}).get("file") or {}).get("content") or None

    soul_r, tools_r, identity_r = await asyncio.gather(
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "SOUL.md"}, gateway_url=helpers._openclaw_ws_url),
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "TOOLS.md"}, gateway_url=helpers._openclaw_ws_url),
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "IDENTITY.md"}, gateway_url=helpers._openclaw_ws_url),
    )
    return {
        "soul": _content(soul_r),
        "tools": _content(tools_r),
        "identity": _content(identity_r),
    }



# --- lines 4204-4223 ---
async def _fetch_vectorstep_gateway_agent_files(agent_id: str) -> dict[str, str | None]:
    """Fetch soul and agent.yaml from the VectorStep Gateway REST API."""
    result: dict[str, str | None] = {"soul": None, "tools": None, "identity": None, "agent_file": None}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            soul_resp, agent_resp = await asyncio.gather(
                client.get(f"{helpers._vectorstep_gateway_base}/agents/{agent_id}/soul"),
                client.get(f"{helpers._vectorstep_gateway_base}/agents/{agent_id}/agent"),
            )
            if soul_resp.status_code == 200:
                result["soul"] = soul_resp.json().get("content") or soul_resp.text
            if agent_resp.status_code == 200:
                result["agent_file"] = agent_resp.json().get("content") or agent_resp.text
    except Exception:
        pass
    return result


# ── Routes ────────────────────────────────────────────────────────────────────


# --- lines 4224-4409 ---
@router.get("/agents", response_class=HTMLResponse)
async def ui_agents(request: Request, executor: str | None = None, model: str | None = None):
    # Fetch from both backends concurrently
    (oc_agents, oc_error), (gw_agents, gw_error) = await asyncio.gather(
        helpers._fetch_openclaw_agents(),
        helpers._fetch_vectorstep_gateway_agents(),
    )

    # Tag every entry with its executor source so the template and URL routing
    # can distinguish agents that share names across backends.
    all_agents: list[dict] = []
    for a in oc_agents:
        all_agents.append({**a, "executor": "openclaw"})
    for a in gw_agents:
        all_agents.append({**a, "executor": "gateway"})

    # Batch-fetch the first SOUL.md line for openclaw agents as a list-page preview.
    if oc_agents and helpers._openclaw_enabled:
        oc_ids = [a.get("id") or a.get("name") for a in oc_agents]
        soul_results = await asyncio.gather(*[
            gateway_call_safe("agents.files.get", {"agentId": aid, "name": "SOUL.md"}, gateway_url=helpers._openclaw_ws_url)
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
                    resp = await client.get(f"{helpers._vectorstep_gateway_base}/agents/{agent_id}/soul")
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

        token_rows = await session.execute(
            select(
                PipelineStep.agent,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent)
        )
        tokens_by_agent: dict[str, tuple[int, int]] = {
            agent: (inp, out) for agent, inp, out in token_rows.all()
        }

        duration_rows = await session.execute(
            select(PipelineStep.agent, func.avg(PipelineStep.duration_ms))
            .where(PipelineStep.agent.isnot(None), PipelineStep.duration_ms.isnot(None))
            .group_by(PipelineStep.agent)
        )
        avg_duration_by_agent: dict[str, float] = dict(duration_rows.all())

    agent_stats: dict[str, dict] = {}
    for row in step_rows:
        s = agent_stats.setdefault(
            row.agent,
            {"succeeded": 0, "failed": 0, "total": 0, "last_run": None, "success_rate": None,
             "avg_input_tokens": None, "avg_output_tokens": None, "avg_duration_secs": None},
        )
        s["total"] += row.n
        if row.status == "failed":
            s["failed"] += row.n
        else:
            s["succeeded"] += row.n
        if row.last_run and (s["last_run"] is None or row.last_run > s["last_run"]):
            s["last_run"] = row.last_run

    for key, s in agent_stats.items():
        if s["total"] > 0:
            s["success_rate"] = round(s["succeeded"] / s["total"] * 100, 1)
            inp, out = tokens_by_agent.get(key, (0, 0))
            if inp or out:
                s["avg_input_tokens"] = round(inp / s["total"])
                s["avg_output_tokens"] = round(out / s["total"])
            avg_duration_ms = avg_duration_by_agent.get(key)
            if avg_duration_ms is not None:
                s["avg_duration_secs"] = avg_duration_ms / 1000

    # If both backends are unreachable, synthesise stub entries from DB history
    # so the page still renders something useful.  Only synthesise for rows that
    # carry a recognisable "executor:agent" prefix — bare legacy entries are ignored.
    if not all_agents:
        seen: set[str] = set()
        for key in sorted(agent_stats.keys()):
            if ":" in key:
                stub_executor, _, aid = key.partition(":")
                stub_key = f"{stub_executor}:{aid}"
                if stub_key not in seen:
                    seen.add(stub_key)
                    all_agents.append({"id": aid, "name": aid, "executor": stub_executor})

    # Reverse lookup — which pipelines reference each "executor:agent", read from config
    # (same source as the Pipelines page's agent badges).
    all_pipelines = getattr(request.app.state, "pipelines", [])
    pipelines_by_agent: dict[str, list[str]] = defaultdict(list)
    for p in all_pipelines:
        for key in helpers._agents_in_pipeline(p):
            pipelines_by_agent[key].append(p.name)

    # ── Filter option universes (unfiltered) ──────────────────────────────────
    all_executors = sorted({a.get("executor", "") for a in all_agents if a.get("executor")})
    all_models = sorted({
        m for a in all_agents
        for m in ([a["model"]] if a.get("model") else []) + (a.get("model_fallbacks") or [])
    })

    if executor:
        all_agents = [a for a in all_agents if a.get("executor") == executor]
    if model:
        all_agents = [
            a for a in all_agents
            if a.get("model") == model or model in (a.get("model_fallbacks") or [])
        ]

    # ── Header stat cards — scoped to the filtered agent set ─────────────────
    filtered_keys = [
        f"{a.get('executor', '')}:{a.get('id') or a.get('agentId') or a.get('name') or ''}"
        for a in all_agents
    ]
    executor_counts = Counter(a.get("executor", "") for a in all_agents)
    total_steps = sum(agent_stats.get(k, {}).get("total", 0) for k in filtered_keys)
    total_succeeded = sum(agent_stats.get(k, {}).get("succeeded", 0) for k in filtered_keys)
    overall_success_rate = round(total_succeeded / total_steps * 100) if total_steps else None
    total_input_tokens = sum(tokens_by_agent.get(k, (0, 0))[0] for k in filtered_keys)
    total_output_tokens = sum(tokens_by_agent.get(k, (0, 0))[1] for k in filtered_keys)

    return templates.TemplateResponse(request, "agents.html", {
        "agents": all_agents,
        "agent_stats": agent_stats,
        "pipelines_by_agent": pipelines_by_agent,
        "gateway_errors": gateway_errors,
        "all_executors": all_executors,
        "all_models": all_models,
        "selected_executor": executor or "",
        "selected_model": model or "",
        "total_agent_count": len(all_agents),
        "executor_counts": dict(executor_counts),
        "overall_success_rate": overall_success_rate,
        "total_steps": total_steps,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "active_page": "agents",
    })



# --- lines 4410-4416 ---
@router.get("/providers", response_class=RedirectResponse)
async def ui_providers(time_range: str = "7d"):
    """Folded into Insights — /ui/providers now redirects there. Kept as a route (rather
    than removed outright) so old bookmarks/links keep working."""
    return RedirectResponse(url=f"/ui/insights/providers?time_range={time_range}", status_code=307)



# --- lines 4417-4619 ---
@router.get("/agents/{executor}/{agent_id}", response_class=HTMLResponse)
async def ui_agent_detail(request: Request, executor: str, agent_id: str):
    prefixed_key = f"{executor}:{agent_id}"

    # Fetch live config + file contents from the appropriate backend
    if executor == "openclaw":
        agents_raw, _ = await helpers._fetch_openclaw_agents()
        agent_files = await _fetch_openclaw_agent_files(agent_id)
    else:  # gateway (or any future executor with REST discovery)
        agents_raw, _ = await helpers._fetch_vectorstep_gateway_agents()
        agent_files = await _fetch_vectorstep_gateway_agent_files(agent_id)

    agent_config = next(
        (a for a in agents_raw if a.get("name") == agent_id or a.get("id") == agent_id),
        None,
    )

    # DB run stats scoped to this executor:agent pair, production runs only (see _production_only)
    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.model,
                    PipelineStep.provider,
                    PipelineStep.status,
                    func.count().label("n"),
                    func.coalesce(func.sum(PipelineStep.input_tokens), 0).label("input_tokens"),
                    func.coalesce(func.sum(PipelineStep.output_tokens), 0).label("output_tokens"),
                    func.avg(PipelineStep.duration_ms).label("avg_duration_ms"),
                    func.max(PipelineStep.executed_at).label("last_run"),
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.agent == prefixed_key)
                .group_by(PipelineStep.model, PipelineStep.provider, PipelineStep.status)
            )
        )
        model_status_rows = rows.all()

        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.step_name,
                    PipelineRun.pipeline_name,
                    PipelineStep.model,
                    PipelineStep.provider,
                    PipelineStep.status,
                    func.count().label("n"),
                    func.coalesce(func.sum(PipelineStep.input_tokens), 0).label("input_tokens"),
                    func.coalesce(func.sum(PipelineStep.output_tokens), 0).label("output_tokens"),
                    func.max(PipelineStep.executed_at).label("last_run"),
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.agent == prefixed_key)
                .group_by(
                    PipelineStep.step_name, PipelineRun.pipeline_name,
                    PipelineStep.model, PipelineStep.provider, PipelineStep.status,
                )
            )
        )
        step_status_rows = rows.all()

        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.executed_at, PipelineStep.model, PipelineStep.provider,
                    PipelineStep.input_tokens, PipelineStep.output_tokens,
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.agent == prefixed_key)
            )
        )
        ts_rows = rows.all()

        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.run_id, PipelineStep.step_name, PipelineStep.model,
                    PipelineStep.provider, PipelineStep.status, PipelineStep.effective_confidence,
                    PipelineStep.duration_ms, PipelineStep.input_tokens,
                    PipelineStep.output_tokens, PipelineStep.executed_at,
                    PipelineRun.pipeline_name,
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.agent == prefixed_key)
                .order_by(PipelineStep.executed_at.desc())
                .limit(15)
            )
        )
        recent_rows = rows.all()

    # ── By-model breakdown — runs, success rate, avg duration, avg tokens ────
    model_stats: dict[str, dict] = {}
    for row in model_status_rows:
        m = helpers._qualified_model(row.provider, row.model)
        s = model_stats.setdefault(m, {
            "succeeded": 0, "failed": 0, "total": 0, "last_run": None,
            "input_tokens": 0, "output_tokens": 0, "duration_sum_ms": 0.0, "duration_n": 0,
        })
        s["total"] += row.n
        if row.status == "failed":
            s["failed"] += row.n
        else:
            s["succeeded"] += row.n
        s["input_tokens"] += row.input_tokens
        s["output_tokens"] += row.output_tokens
        if row.avg_duration_ms is not None:
            s["duration_sum_ms"] += row.avg_duration_ms * row.n
            s["duration_n"] += row.n
        if row.last_run and (s["last_run"] is None or row.last_run > s["last_run"]):
            s["last_run"] = row.last_run

    for s in model_stats.values():
        total = s["total"]
        s["success_rate"] = round(s["succeeded"] / total * 100, 1) if total else None
        s["avg_input_tokens"] = round(s["input_tokens"] / total) if total and s["input_tokens"] else None
        s["avg_output_tokens"] = round(s["output_tokens"] / total) if total and s["output_tokens"] else None
        s["avg_duration_secs"] = (s["duration_sum_ms"] / s["duration_n"] / 1000) if s["duration_n"] else None

    # ── By-step breakdown — which pipeline steps this agent runs, per pipeline/model ──
    # Keyed by (step_name, pipeline_name, model) — the same step name can be wired to a
    # different model in different pipelines, so folding pipelines together would hide that.
    step_combo: dict[tuple[str, str, str], dict] = {}
    for row in step_status_rows:
        key = (row.step_name, row.pipeline_name, helpers._qualified_model(row.provider, row.model))
        c = step_combo.setdefault(key, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "last_run": None,
        })
        c["total"] += row.n
        if row.status == "failed":
            c["failed"] += row.n
        c["input_tokens"] += row.input_tokens
        c["output_tokens"] += row.output_tokens
        if row.last_run and (c["last_run"] is None or row.last_run > c["last_run"]):
            c["last_run"] = row.last_run

    step_stats = []
    for (step_name, pipeline_name, model), c in step_combo.items():
        total = c["total"]
        step_stats.append({
            "step_name": step_name,
            "pipeline_name": pipeline_name,
            "model": model,
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "last_run": c["last_run"],
        })
    step_stats.sort(key=lambda r: r["total"], reverse=True)

    # ── Usage over time — run count and token volume, split by model ────────
    now = utc_now()
    runs_ts = helpers._build_ts(
        [(r.executed_at, helpers._qualified_model(r.provider, r.model)) for r in ts_rows], now, None, "all",
        dim_fn=lambda r: r[1],
    )
    tokens_ts = helpers._build_ts(
        [(r.executed_at, helpers._qualified_model(r.provider, r.model),
          (r.input_tokens or 0) + (r.output_tokens or 0)) for r in ts_rows],
        now, None, "all",
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    # ── Recent activity — last 15 steps this agent ran, across any pipeline ──
    recent_activity = [{
        "run_id": r.run_id,
        "pipeline_name": r.pipeline_name,
        "step_name": r.step_name,
        "model": helpers._qualified_model(r.provider, r.model),
        "status": r.status,
        "confidence": r.effective_confidence,
        "duration_secs": (r.duration_ms / 1000) if r.duration_ms is not None else None,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "ago": helpers._format_ago(r.executed_at),
    } for r in recent_rows]

    # Version history is only meaningful for gateway agents — VectorStep has no
    # equivalent content-hash mechanism for openclaw (SPEC-prompt-versioning.md §6d).
    agent_versions = await _get_agent_versions(sf, agent_id) if executor == "gateway" else []

    return templates.TemplateResponse(request, "agent_detail.html", {
        "agent_id": agent_id,
        "executor": executor,
        "agent_config": agent_config,
        "soul": agent_files["soul"],
        "tools": agent_files["tools"],
        "identity": agent_files["identity"],
        "agent_file": agent_files.get("agent_file"),
        "model_stats": model_stats,
        "step_stats": step_stats,
        "runs_ts": runs_ts,
        "tokens_ts": tokens_ts,
        "recent_activity": recent_activity,
        "agent_versions": agent_versions,
        "active_page": "agents",
    })


# ── MCP tools ─────────────────────────────────────────────────────────────────


# --- lines 4620-4659 ---
@router.get("/mcp", response_class=HTMLResponse)
async def ui_mcp(request: Request):
    """Browse the MCP tool registry exposed by the VectorStep Gateway.

    OpenClaw isn't included here — it has no REST endpoint for tool
    introspection (the gateway executor is the only one that exposes
    GET /mcp/tools and GET /mcp/servers).
    """
    tools_by_server, servers_status, error = await helpers._fetch_vectorstep_gateway_mcp()

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


