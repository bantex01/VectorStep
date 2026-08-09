from fastapi import APIRouter
from ..analytics import ALL_STEP_STATUSES
from ..analytics import _production_only
from ..db.database import get_session_factory
from ..db.models import PipelineRun
from ..db.models import PipelineStep
from ..db.models import RunFeedback
from ..db.models import StepFeedback
from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from sqlalchemy import select
import glob
import os
import yaml

from . import helpers
from .helpers import templates


router = APIRouter()


# --- lines 3618-3746 ---
@router.get("/marking-queue", response_class=HTMLResponse)
async def ui_marking_queue(
    request: Request,
    pipeline: str | None = None,
    team: str | None = None,
    stage: str = "testing",
):
    """Cross-pipeline review queue: every step with no HUMAN accuracy feedback
    (StepFeedback) — the exact gap `readiness.accuracy.min_human_marked` checks
    for. A step that already has an automatic label (a failed deterministic
    check, or an inherited run-level rating) still shows up here, tagged with
    where that label came from, since neither counts as `human_marked` — see
    CONFIDENCE-EXPLAINED.md §12.2/§9.

    Filters are additive (pipeline AND team AND stage, all optional). stage
    defaults to "testing" — the pre-promotion review case — but production
    pipelines using calibration independently of the testing->production gate
    need this too, so stage is a real filter, not a hard scope."""
    def _filtered(q):
        if pipeline:
            q = q.where(PipelineRun.pipeline_name == pipeline)
        if team:
            q = q.where(PipelineRun.team == team)
        if stage:
            q = q.where(PipelineRun.stage == stage)
        return q

    sf = get_session_factory()
    async with sf() as session:
        q = _filtered(
            select(
                PipelineStep.run_id, PipelineStep.step_name, PipelineStep.executed_at,
                PipelineStep.deterministic_passed,
                PipelineRun.pipeline_name, PipelineRun.team, PipelineRun.stage,
                PipelineRun.replay_of,
            )
            .select_from(PipelineStep)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .outerjoin(StepFeedback, StepFeedback.step_id == PipelineStep.id)
            .where(StepFeedback.id.is_(None))
            .where(PipelineStep.status.in_(ALL_STEP_STATUSES))
            .order_by(PipelineStep.executed_at.asc())
        )
        unmarked_rows = (await session.execute(q)).all()

        run_feedback_ids: set[str] = set()
        run_ids = {r.run_id for r in unmarked_rows}
        if run_ids:
            rf_rows = await session.execute(
                select(RunFeedback.run_id).where(RunFeedback.run_id.in_(run_ids))
            )
            run_feedback_ids = {r[0] for r in rf_rows.all()}

        q = _filtered(
            select(func.count())
            .select_from(PipelineStep)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.status.in_(ALL_STEP_STATUSES))
        )
        total_terminal_steps = (await session.execute(q)).scalar() or 0

        # Unfiltered so switching one filter doesn't collapse the others' options.
        rows = await session.execute(select(PipelineRun.pipeline_name).distinct().order_by(PipelineRun.pipeline_name))
        pipeline_names = [r[0] for r in rows.all()]
        rows = await session.execute(
            select(PipelineRun.team).distinct().where(PipelineRun.team.is_not(None)).order_by(PipelineRun.team)
        )
        team_names = [r[0] for r in rows.all()]

    # Group: pipeline -> step-group (fan-out/parallel branches collapse to their
    # group name, same as readiness._bucket_name) -> individual unmarked items,
    # oldest first (rows are already ordered that way from the query above).
    pipelines_by_name: dict[str, dict] = {}
    for r in unmarked_rows:
        is_replay = r.replay_of is not None
        if is_replay:
            # A replay batch's own D-check failure still means something ("failed
            # check") but the REPLAY tag takes priority — that's the fact an
            # operator marking this queue needs to see first (SPEC-replay-shadow-eval.md).
            provenance = "REPLAY"
        elif r.deterministic_passed is False:
            provenance = "failed check"
        elif r.run_id in run_feedback_ids:
            provenance = "run feedback"
        else:
            provenance = None

        p = pipelines_by_name.setdefault(r.pipeline_name, {
            "name": r.pipeline_name, "stage": r.stage, "steps": {}, "run_ids": set(),
        })
        p["run_ids"].add(r.run_id)
        step_group = r.step_name.split("/", 1)[0]
        s = p["steps"].setdefault(step_group, {"name": step_group, "items": []})
        s["items"].append({
            "run_id": r.run_id, "executed_at": r.executed_at, "provenance": provenance,
            "is_replay": is_replay,
        })

    pipeline_list = []
    total_unmarked_steps = 0
    for p in pipelines_by_name.values():
        step_list = []
        for s in p["steps"].values():
            s["count"] = len(s["items"])
            s["oldest"] = s["items"][0]["executed_at"]
            s["preview"] = s["items"][:5]
            s["more"] = max(0, s["count"] - 5)
            step_list.append(s)
        step_list.sort(key=lambda s: s["count"], reverse=True)
        step_count = sum(s["count"] for s in step_list)
        total_unmarked_steps += step_count
        pipeline_list.append({
            "name": p["name"], "stage": p["stage"], "steps": step_list,
            "step_count": step_count, "run_count": len(p["run_ids"]),
        })
    pipeline_list.sort(key=lambda p: p["step_count"], reverse=True)

    marked_pct = (
        round((total_terminal_steps - total_unmarked_steps) / total_terminal_steps * 100)
        if total_terminal_steps else None
    )

    return templates.TemplateResponse(request, "marking_queue.html", {
        "pipelines": pipeline_list,
        "pipeline_names": pipeline_names,
        "team_names": team_names,
        "selected_pipeline": pipeline or "",
        "selected_team": team or "",
        "selected_stage": stage or "",
        "stage_values": ["testing", "production"],
        "total_pipelines": len(pipeline_list),
        "total_runs": len({r.run_id for r in unmarked_rows}),
        "total_unmarked_steps": total_unmarked_steps,
        "total_terminal_steps": total_terminal_steps,
        "marked_pct": marked_pct,
        "active_page": "marking_queue",
    })



# --- lines 3788-3803 ---
def _read_step_yaml(steps_dir: str, step_name: str) -> str | None:
    """Same lookup as _read_pipeline_yaml, for the step library directory."""
    for path in glob.glob(os.path.join(steps_dir, "*.yaml")):
        try:
            with open(path) as f:
                content = f.read()
            data = yaml.safe_load(content)
            if isinstance(data, dict) and data.get("name") == step_name:
                return content
        except Exception:
            continue
    return None


# ── Step library ─────────────────────────────────────────────────────────────


# --- lines 3804-3812 ---
def _iter_all_raw_steps(steps: list):
    """Yield every step dict from a raw YAML steps list, including parallel inner steps."""
    for step in steps:
        if isinstance(step, dict) and "parallel" in step:
            yield from _iter_all_raw_steps(step["parallel"].get("steps", []))
        else:
            yield step



# --- lines 3813-3829 ---
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



# --- lines 3830-3854 ---
def _compute_step_runtime_names(pipeline_dir: str, library: dict) -> dict[str, set[str]]:
    """Map each library step name to the runtime `step_name`(s) it executes under.

    A pipeline step that does `use: <library-name>` inherits the library step's
    `name` field unless it overrides `name:` locally (see
    `pipeline.loader._resolve_step_references`) — so the DB's `pipeline_steps.step_name`
    usually matches the library name directly, but can differ if a pipeline renames it.
    Always includes the library name itself so unused/never-renamed steps still match.
    """
    runtime_names: dict[str, set[str]] = {name: {name} for name in library}
    for path in glob.glob(os.path.join(pipeline_dir, "*.yaml")):
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
            for step in _iter_all_raw_steps(raw.get("steps", [])):
                if not isinstance(step, dict):
                    continue
                use = step.get("use")
                if use and use in runtime_names:
                    runtime_names[use].add(step.get("name") or use)
        except Exception:
            pass
    return runtime_names



# --- lines 3855-3952 ---
async def _fetch_step_model_stats(
    runtime_names: dict[str, set[str]],
) -> dict[str, list[dict]]:
    """Per-library-step breakdown of run history by (pipeline, agent, model): success rate
    and average token usage, aggregated across every runtime step_name the library step is
    known to execute under (see _compute_step_runtime_names). Scoped to production runs
    only, matching the rest of the app's rollup surfaces (see _production_only).

    Grouped by pipeline as well as agent/model — a step used by several pipelines can be
    wired to a different agent/model in each, and folding them together would hide that."""
    all_names = {n for names in runtime_names.values() for n in names}
    if not all_names:
        return {}

    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.step_name,
                    PipelineRun.pipeline_name,
                    PipelineStep.agent,
                    PipelineStep.model,
                    PipelineStep.provider,
                    PipelineStep.status,
                    func.count().label("n"),
                    func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                    func.coalesce(func.sum(PipelineStep.output_tokens), 0),
                    func.max(PipelineStep.executed_at).label("last_run"),
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.step_name.in_(all_names))
                .group_by(
                    PipelineStep.step_name, PipelineRun.pipeline_name, PipelineStep.agent,
                    PipelineStep.model, PipelineStep.provider, PipelineStep.status,
                )
            )
        )
        db_rows = rows.all()

    # (step_name, pipeline_name, agent, qualified_model) -> aggregated counters. Qualifying
    # the model with its provider (see helpers._qualified_model) up front means two providers that
    # happen to report the same bare model string aren't silently merged together.
    combo_stats: dict[tuple[str, str, str | None, str], dict] = {}
    for step_name, pipeline_name, agent, model, provider, status, n, in_tok, out_tok, last_run in db_rows:
        key = (step_name, pipeline_name, agent, helpers._qualified_model(provider, model))
        c = combo_stats.setdefault(key, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "last_run": None,
        })
        c["total"] += n
        if status == "failed":
            c["failed"] += n
        c["input_tokens"] += in_tok
        c["output_tokens"] += out_tok
        if last_run and (c["last_run"] is None or last_run > c["last_run"]):
            c["last_run"] = last_run

    result: dict[str, list[dict]] = {}
    for lib_name, names in runtime_names.items():
        # Re-aggregate by (pipeline, agent, model) across every runtime name for this
        # library step.
        by_pipeline_agent_model: dict[tuple[str, str | None, str], dict] = {}
        for (step_name, pipeline_name, agent, model), c in combo_stats.items():
            if step_name not in names:
                continue
            row = by_pipeline_agent_model.setdefault(
                (pipeline_name, agent, model),
                {"total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "last_run": None},
            )
            row["total"] += c["total"]
            row["failed"] += c["failed"]
            row["input_tokens"] += c["input_tokens"]
            row["output_tokens"] += c["output_tokens"]
            if c["last_run"] and (row["last_run"] is None or c["last_run"] > row["last_run"]):
                row["last_run"] = c["last_run"]

        if not by_pipeline_agent_model:
            continue

        rows_out = []
        for (pipeline_name, agent, model), row in by_pipeline_agent_model.items():
            total = row["total"]
            rows_out.append({
                "pipeline_name": pipeline_name,
                "agent": agent,
                "model": model,
                "total": total,
                "success_rate": round((total - row["failed"]) / total * 100) if total else None,
                "avg_input_tokens": round(row["input_tokens"] / total) if total else None,
                "avg_output_tokens": round(row["output_tokens"] / total) if total else None,
                "last_run": row["last_run"],
            })
        rows_out.sort(key=lambda r: r["total"], reverse=True)
        result[lib_name] = rows_out

    return result



# --- lines 3953-3974 ---
@router.get("/steps", response_class=HTMLResponse)
async def ui_steps(request: Request, tag: str | None = None):
    step_library: dict = getattr(request.app.state, "step_library", {})
    pipeline_dir: str = getattr(request.app.state, "pipeline_dir", "./pipelines")

    step_usage = _compute_step_usage(pipeline_dir, step_library)
    runtime_names = _compute_step_runtime_names(pipeline_dir, step_library)
    step_model_stats = await _fetch_step_model_stats(runtime_names)
    all_steps = sorted(step_library.values(), key=lambda s: s.get("name", ""))
    all_tags = sorted({t for s in all_steps for t in s.get("tags") or []})
    steps = [s for s in all_steps if tag in (s.get("tags") or [])] if tag else all_steps

    return templates.TemplateResponse(request, "steps.html", {
        "steps": steps,
        "step_usage": step_usage,
        "step_model_stats": step_model_stats,
        "all_tags": all_tags,
        "selected_tag": tag or "",
        "active_page": "steps",
    })



