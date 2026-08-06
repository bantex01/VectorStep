from fastapi import APIRouter
from ..analytics import _production_only
from ..db.database import get_session_factory
from ..db.models import PipelineRun
from ..db.models import PipelineStep
from ..db.models import RunFeedback
from ..models.pipeline import PipelineConfig
from ..readiness import READINESS_KNOB_HELP
from ..readiness import builder_seed as _builder_seed
from ..readiness import evaluate_readiness as _evaluate_readiness
from ..readiness import gather_readiness_evidence as _gather_readiness_evidence
from collections import defaultdict
from datetime import datetime
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from sqlalchemy import select
import asyncio
import glob
import hashlib
import os
import yaml

from . import helpers
from .helpers import templates


router = APIRouter()


# --- lines 680-688 ---
def _config_fingerprint(steps: list) -> str:
    """12-char SHA-256 of the ordered (step_name, agent, model) sequence for a run."""
    key = tuple(
        (s.step_name, s.agent or "", s.model or "")
        for s in sorted(steps, key=lambda x: x.step_index)
    )
    return hashlib.sha256(str(key).encode()).hexdigest()[:12]



# --- lines 689-696 ---
def _config_description(steps: list) -> dict:
    """Human-readable summary of the config for display in the feedback breakdown."""
    sorted_steps = sorted(steps, key=lambda s: s.step_index)
    step_names = [s.step_name for s in sorted_steps if "/" not in s.step_name]
    models = sorted({s.model for s in sorted_steps if s.model})
    return {"steps": step_names, "models": models}



# --- lines 1363-1469 ---
@router.get("/pipelines", response_class=HTMLResponse)
async def ui_pipelines(request: Request, tag: str | None = None, agent: str | None = None):
    all_pipelines = getattr(request.app.state, "pipelines", [])
    all_tags = sorted({t for p in all_pipelines for t in p.tags})
    agents_by_pipeline = {p.name: helpers._agents_in_pipeline(p) for p in all_pipelines}
    all_agents = sorted({a for agents in agents_by_pipeline.values() for a in agents})

    pipelines = all_pipelines
    if tag:
        pipelines = [p for p in pipelines if tag in p.tags]
    if agent:
        pipelines = [p for p in pipelines if agent in agents_by_pipeline.get(p.name, [])]

    sf = get_session_factory()

    async with sf() as session:
        # All aggregates on this page (last-run status, run counts, success rate, avg
        # tokens, accuracy) are rollups, not a browse list — scoped to production.
        rows = await session.execute(_production_only(
            select(PipelineRun)
            .order_by(PipelineRun.triggered_at.desc())
        ))
        all_runs = rows.scalars().all()

        fb_rows = await session.execute(_production_only(
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        ))
        feedback_by_pipeline: dict[str, dict[str, int]] = {}
        for name, outcome, n in fb_rows.all():
            if name not in feedback_by_pipeline:
                feedback_by_pipeline[name] = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
            feedback_by_pipeline[name][outcome] = n
            feedback_by_pipeline[name]["total"] += n

        token_rows = await session.execute(_production_only(
            select(
                PipelineRun.pipeline_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineStep, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineRun.pipeline_name)
        ))
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



# --- lines 3415-3516 ---
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
        # Browse surface — shows testing runs too (badged in the template).
        rows = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.pipeline_name == name)
            .order_by(PipelineRun.triggered_at.desc())
            .limit(10)
        )
        recent_runs = rows.scalars().all()
        feedback_by_run = await helpers._feedback_by_run_id(session, [r.id for r in recent_runs])

        # Success-rate / accuracy bars below are rollups — scoped to production.
        rows = await session.execute(_production_only(
            select(PipelineRun.status, func.count().label("n"))
            .where(PipelineRun.pipeline_name == name)
            .group_by(PipelineRun.status)
        ))
        status_counts = dict(rows.all())

        rows = await session.execute(_production_only(
            select(RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(RunFeedback.pipeline_name == name)
            .group_by(RunFeedback.outcome)
        ))
        feedback_counts = dict(rows.all())

    feedback_total = sum(feedback_counts.values())

    promotion_readiness = None
    readiness_builder_seed = None
    if pipeline.stage == "testing":
        readiness_evidence = await _gather_readiness_evidence(sf, pipeline)
        promotion_readiness = _evaluate_readiness(readiness_evidence, pipeline)
        readiness_builder_seed = _builder_seed(pipeline)

    # Group this pipeline's agent usage by executor:agent, then enrich with each
    # agent's live-configured model + fallbacks (only known from the backend's
    # own config, unlike everything else here which comes straight from the
    # pipeline YAML — see helpers._agent_usage_in_pipeline).
    usage = helpers._agent_usage_in_pipeline(pipeline)
    agents_grouped: dict[str, dict] = {}
    for u in usage:
        g = agents_grouped.setdefault(u["key"], {
            "executor": u["executor"], "agent": u["agent"], "roles": set(), "steps": [],
        })
        g["roles"].add(u["role"])
        if u["step"] not in g["steps"]:
            g["steps"].append(u["step"])

    live_by_key: dict[str, dict] = {}
    if agents_grouped:
        (oc_agents, _), (gw_agents, _) = await asyncio.gather(
            helpers._fetch_openclaw_agents(), helpers._fetch_vectorstep_gateway_agents(),
        )
        for a in oc_agents:
            live_by_key[f"openclaw:{a.get('id') or a.get('name')}"] = a
        for a in gw_agents:
            live_by_key[f"gateway:{a.get('id') or a.get('name')}"] = a

    pipeline_agents = []
    for key, g in sorted(agents_grouped.items()):
        live = live_by_key.get(key)
        pipeline_agents.append({
            "key": key,
            "executor": g["executor"],
            "agent": g["agent"],
            "roles": sorted(g["roles"]),
            "steps": g["steps"],
            "found": live is not None,
            "model": live.get("model") if live else None,
            "model_fallbacks": (live.get("model_fallbacks") or []) if live else [],
        })

    return templates.TemplateResponse(request, "pipeline_detail.html", {
        "pipeline": pipeline,
        "raw_yaml": raw_yaml,
        "recent_runs": recent_runs,
        "feedback_by_run": feedback_by_run,
        "status_counts": status_counts,
        "total_runs": sum(status_counts.values()),
        "feedback_counts": feedback_counts,
        "feedback_total": feedback_total,
        "pipeline_agents": pipeline_agents,
        "promotion_readiness": promotion_readiness,
        "readiness_builder_seed": readiness_builder_seed,
        "readiness_knob_help": READINESS_KNOB_HELP,
        "active_page": "pipelines",
    })



# --- lines 3517-3617 ---
@router.get("/pipelines/{name}/feedback", response_class=HTMLResponse)
async def ui_pipeline_feedback(request: Request, name: str):
    pipelines = getattr(request.app.state, "pipelines", [])
    pipeline = next((p for p in pipelines if p.name == name), None)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    sf = get_session_factory()
    async with sf() as session:
        # Chronological browse list — includes testing runs, badged (see stage below).
        # The config-fingerprint comparison further down is production-only, since
        # mixing testing and production runs would corrupt "compare accuracy before
        # vs after a pipeline change."
        rows = await session.execute(
            select(RunFeedback, PipelineRun.triggered_at, PipelineRun.status, PipelineRun.stage)
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(RunFeedback.pipeline_name == name)
            .order_by(RunFeedback.submitted_at.desc())
        )
        feedback_rows = rows.all()

        run_ids = [fb.run_id for fb, _, _, _ in feedback_rows]
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
    for fb, triggered_at, run_status, stage in feedback_rows:
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
            "stage": stage,
            "fingerprint": fp,
            "config": desc,
        })

    # Group by config fingerprint — summary cards and this comparison are rollups,
    # scoped to production only (unlike the chronological marked_runs list above).
    production_marked_runs = [r for r in marked_runs if r["stage"] == "production"]
    config_groups: dict[str, dict] = {}
    for r in production_marked_runs:
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

    total_marked = len(production_marked_runs)
    total_correct = sum(1 for r in production_marked_runs if r["outcome"] == "correct")
    total_partial = sum(1 for r in production_marked_runs if r["outcome"] == "partial")
    total_incorrect = sum(1 for r in production_marked_runs if r["outcome"] == "incorrect")

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



# --- lines 3747-3774 ---
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


# --- lines 3775-3787 ---
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



