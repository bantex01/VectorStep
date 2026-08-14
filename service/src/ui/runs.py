from fastapi import APIRouter
from .. import run_events
from ..analytics import _production_only
from ..analytics import _time_range_cutoff
from ..db.database import get_session_factory
from ..db.models import AgentVersionSnapshot
from ..db.models import PipelineRun
from ..db.models import PipelineStep
from ..db.models import RunFeedback
from ..db.models import StepFeedback
from datetime import datetime
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import asyncio
import json
import re
import yaml

from . import helpers
from .helpers import templates


router = APIRouter()


# --- lines 283-307 ---
def _bucket_reset_narrative(bucket_reset: dict, n: int, n_min: int) -> str:
    """Plain-language line for the bucket_reset key on a calibration report
    (SPEC-prompt-versioning.md §4h/§6c) — the single most valuable line in this
    whole feature: it converts "this step suddenly started escalating everything
    and I don't know why" into "this step's calibration reset when you edited its
    prompt on 3 Jul; it needs 20 marked results at the new version."."""
    reason_text = {
        "prompt_changed": "this step's prompt changed",
        "agent_changed": "the agent's configuration changed",
        "both_changed": "this step's prompt and the agent's configuration both changed",
    }.get(bucket_reset["reason"], "this step's configuration changed")
    when = ""
    last_seen = bucket_reset.get("previous_version_last_seen")
    if last_seen:
        try:
            when = f" on {datetime.fromisoformat(last_seen).strftime('%b %d')}"
        except ValueError:
            when = ""
    previous_n = bucket_reset["previous_validated_n"]
    return (
        f"Calibration was reset when {reason_text}{when}. The previous version had "
        f"{previous_n} marked results; this one has {n} of the {n_min} needed."
    )



# --- lines 308-449 ---
def _confidence_narrative(trust: dict, status: str) -> list[str]:
    """Plain-language, numbers-first walkthrough of how this ONE run's trust score was
    derived — S -> verifier combine -> calibration -> grounding -> deterministic checks
    -> final gate decision. No config keys or jargon; every figure is real data from
    this run, not a description of how the mechanism works in general."""
    sig = trust["signals"]
    lines: list[str] = []

    s = sig["S"]
    lines.append(f"This step's agent reported {s:.0%} confidence in its own answer.")

    seed = s
    if sig.get("V") is not None:
        v = sig["V"]
        s_after_v = sig["S_after_V"]
        blind = sig.get("V_mode") == "independent"
        checker = "A second, blind check (no sight of the primary's answer)" if blind else "A second check that reviewed the primary's own reasoning"
        veto_floor = sig.get("V_veto_floor")
        if s_after_v < s:
            lines.append(
                f"{checker} scored it at {v:.0%}. A verifier can only lower or hold "
                f"confidence, never raise it, so this pulled the working score down to {s_after_v:.0%}."
            )
        elif v < s:
            # s_after_v == s despite v < s is only reachable when the veto floor wasn't
            # tripped — under a plain minimum-of-the-two strategy this branch can't
            # happen at all, since that always takes the lower value unconditionally.
            if veto_floor is not None:
                lines.append(
                    f"{checker} scored it at {v:.0%}. This step uses a 'veto' rule: the "
                    f"verifier only overrides the primary's score when it scores below "
                    f"{veto_floor:.0%} — since {v:.0%} cleared that bar, the {s:.0%} "
                    f"confidence stood unchanged."
                )
            else:
                lines.append(
                    f"{checker} scored it at {v:.0%} — not low enough to change anything "
                    f"here, so the {s:.0%} confidence stood."
                )
        else:
            lines.append(
                f"{checker} scored it at {v:.0%} — at or above the {s:.0%} self-report, "
                f"so there was nothing to pull down."
            )
        seed = s_after_v

    calibration = trust.get("calibration")
    if calibration:
        n, n_min = calibration["n"], calibration["n_min"]
        if calibration["validated"]:
            calibrated = calibration["calibrated"]
            lines.append(
                f"This exact combination of step, agent, and model has a track record: "
                f"based on {n} past marked results, it's actually correct about "
                f"{calibrated:.0%} of the time when it reports around this confidence "
                f"level. That measured figure replaced the {seed:.0%} score above."
            )
            seed = calibrated
        elif calibration["on_uncalibrated"] == "escalate":
            lines.append(
                f"This combination doesn't have enough history yet to trust "
                f"({n}/{n_min} marked results) — since it's configured to play it safe "
                f"until proven, this run's confidence was treated as 0%."
            )
            seed = 0.0
        else:
            lines.append(
                f"This combination doesn't have enough history yet to calibrate "
                f"({n}/{n_min} marked results), so the {seed:.0%} score above was used as-is."
            )
        bucket_reset = calibration.get("bucket_reset")
        if bucket_reset:
            lines.append(_bucket_reset_narrative(bucket_reset, n, n_min))

    grounding = trust.get("grounding")
    if grounding and grounding.get("computed"):
        g = grounding["score"]
        enforce = grounding.get("enforce")
        if enforce is None:
            # Historical row from before the grounding report recorded its own enforce
            # flag — computed and reported either way, so its presence alone can't tell
            # us whether it actually gated this specific run. Say so rather than guess.
            lines.append(
                f"A separate check found {g:.0%} of this run's load-bearing claims "
                f"supported by its own evidence (whether this affected the final trust "
                f"isn't recorded for this older run)."
            )
        elif enforce:
            if g < seed:
                lines.append(
                    f"A separate check looked at whether this run's own evidence backs "
                    f"up what it claimed, and found only {g:.0%} of the load-bearing "
                    f"claims supported — this capped the confidence at {g:.0%}."
                )
                seed = g
            else:
                lines.append(
                    f"A separate check found {g:.0%} of this run's load-bearing claims "
                    f"supported by its own evidence — at or above the confidence already "
                    f"reached, so it didn't change anything."
                )
        else:
            lines.append(
                f"A separate check found {g:.0%} of this run's load-bearing claims "
                f"supported by its own evidence — recorded for visibility only, it "
                f"wasn't configured to affect this run's outcome."
            )
    elif grounding and not grounding.get("computed"):
        lines.append("A grounding check was configured but couldn't complete this run, so it had no effect.")

    det = trust.get("deterministic_checks")
    if det:
        failed = [c["name"] for c in det if not c["passed"]]
        if failed:
            lines.append(
                f"A hard, automated check failed ({', '.join(failed)}) — this overrides "
                f"everything above and forces the confidence straight to 0%."
            )
            seed = 0.0
        else:
            lines.append("All of this run's automated checks passed — no additional effect on the confidence above.")

    combined = trust["combined_trust"]
    outcome = {
        "completed": "completed normally",
        "escalated": "was escalated for a human to review",
        "aborted": "was aborted",
        "stopped": "stopped the pipeline here",
        "failed": "failed",
    }.get(status, status)
    threshold = trust.get("gate", {}).get("confidence_threshold")
    if threshold is not None:
        cleared = "cleared" if combined >= threshold else "fell short of"
        lines.append(
            f"Put together, this run's final confidence came out at {combined:.0%}, "
            f"which {cleared} this step's {threshold:.0%} threshold — so it {outcome}."
        )
    else:
        lines.append(f"Put together, this run's final confidence came out at {combined:.0%}, and the step {outcome}.")
    return lines



# --- lines 450-509 ---
def _step_config_summary(trust: dict) -> list[str]:
    """Plain-language summary of how this step is set up to be judged — the gate
    threshold plus whichever of verifier/grounding/deterministic/calibration are
    configured. Complements _confidence_narrative (what happened this run) with what's
    configured to happen on every run, so a reviewer has the full picture in one place
    without going to find the pipeline's YAML."""
    lines: list[str] = []
    gate = trust.get("gate") or {}
    threshold = gate.get("confidence_threshold")
    if threshold is not None:
        lines.append(f"Confidence threshold: {threshold:.0%} (on low confidence: {gate.get('on_low_confidence')}).")

    sig = trust["signals"]
    if sig.get("V") is not None:
        mode = sig.get("V_mode") or "configured"
        strategy = sig.get("V_combination_strategy")
        floor = sig.get("V_veto_floor")
        if strategy == "veto" and floor is not None:
            detail = f" (veto rule, floor {floor:.0%})"
        elif strategy:
            detail = f" ({strategy} combination)"
        else:
            detail = ""
        lines.append(f"Verifier: {mode}{detail}.")

    grounding = trust.get("grounding")
    if grounding:
        agent = grounding.get("agent")
        enforce = grounding.get("enforce")
        if enforce is True:
            lines.append(f"Grounding: enforced (agent: {agent}).")
        elif enforce is False:
            lines.append(f"Grounding: shadow only, recorded but not enforced (agent: {agent}).")
        else:
            lines.append(f"Grounding: configured (agent: {agent}) — enforcement not recorded for this older run.")

    det = trust.get("deterministic_checks")
    if det:
        names = ", ".join(f"{c['name']} ({c['type']})" for c in det)
        lines.append(f"Deterministic checks: {names}.")

    calibration = trust.get("calibration")
    if calibration:
        line = (
            f"Calibration: enforced (needs {calibration['n_min']} marked results; "
            f"on_uncalibrated: {calibration['on_uncalibrated']})."
        )
        bucket_reset = calibration.get("bucket_reset")
        if bucket_reset:
            reason_label = {
                "prompt_changed": "prompt changed",
                "agent_changed": "agent changed",
                "both_changed": "prompt and agent changed",
            }.get(bucket_reset["reason"], "configuration changed")
            line += f" Reset: {reason_label}, previous version had {bucket_reset['previous_validated_n']} marked results."
        lines.append(line)

    return lines



# --- lines 963-1072 ---
@router.get("/runs", response_class=HTMLResponse)
async def ui_runs(
    request: Request,
    status: str | None = None,
    pipeline: str | None = None,
    team: str | None = None,
    stage: str | None = None,
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
        if stage:
            q = q.where(PipelineRun.stage == stage)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        return q

    sf = get_session_factory()
    async with sf() as session:
        # Browse surface — shows testing runs too (badged in the template) unless the
        # user explicitly filters by stage; only the stat cards below are always
        # scoped to production, regardless of the stage filter.
        q = _filtered(select(PipelineRun).order_by(PipelineRun.triggered_at.desc()))
        rows = await session.execute(q.limit(limit).offset(offset))
        runs = rows.scalars().all()
        feedback_by_run = await helpers._feedback_by_run_id(session, [r.id for r in runs])

        rows = await session.execute(
            select(PipelineRun.pipeline_name).distinct().order_by(PipelineRun.pipeline_name)
        )
        pipeline_names = [r[0] for r in rows.all()]

        rows = await session.execute(
            select(PipelineRun.team).distinct().where(PipelineRun.team.is_not(None)).order_by(PipelineRun.team)
        )
        team_names = [r[0] for r in rows.all()]

        # Stat cards — aggregated over every run matching the active filters, not just the
        # current page, and always scoped to production (see _production_only).
        q = _production_only(_filtered(select(PipelineRun.status, func.count()).group_by(PipelineRun.status)))
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

        q = _production_only(_filtered(
            select(RunFeedback.outcome, func.count())
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .group_by(RunFeedback.outcome)
        ))
        rows = await session.execute(q)
        feedback_counts: dict[str, int] = dict(rows.all())
        marked_total = sum(feedback_counts.values())
        accuracy_pct = (
            round(feedback_counts.get("correct", 0) / marked_total * 100)
            if marked_total > 0 else None
        )

        q = _production_only(_filtered(
            select(func.coalesce(func.sum(PipelineStep.input_tokens), 0) + func.coalesce(func.sum(PipelineStep.output_tokens), 0))
            .select_from(PipelineRun)
            .join(PipelineStep, PipelineStep.run_id == PipelineRun.id)
        ))
        token_total = (await session.execute(q)).scalar() or 0

    return templates.TemplateResponse(request, "runs.html", {
        "runs": runs,
        "feedback_by_run": feedback_by_run,
        "pipeline_names": pipeline_names,
        "team_names": team_names,
        "selected_status": status or "",
        "selected_pipeline": pipeline or "",
        "selected_team": team or "",
        "selected_stage": stage or "",
        "stage_values": ["testing", "production"],
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



# --- lines 1073-1252 ---
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
        verifier_label = "Independent" if step.verifier_mode in ("challenger", "independent") else "Critic"

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
                "verifier_parsed": verifier_parsed,
                "verifier_pretty": verifier_pretty,
                "verifier_label": verifier_label,
                "trace": trace,
            })
        else:
            trust = json.loads(step.trust_report) if step.trust_report else None
            approx_cost, approx_is_native = helpers._approx_cost_for_step(step)
            display_items.append({
                "type": "step",
                "name": step.step_name,
                "step": step,
                "parsed": parsed,
                "pretty": pretty,
                "verifier_parsed": verifier_parsed,
                "verifier_pretty": verifier_pretty,
                "verifier_label": verifier_label,
                "trace": trace,
                "trust": trust,
                "confidence_narrative": _confidence_narrative(trust, step.status) if trust else None,
                "step_config_summary": _step_config_summary(trust) if trust else None,
                "approx_cost": approx_cost,
                "approx_is_native": approx_is_native,
            })

    normalised = json.loads(run.normalised_context) if run.normalised_context else {}
    run_log = json.loads(run.logs) if run.logs else []

    total_input_tokens = sum(s.input_tokens or 0 for s in run.steps)
    total_output_tokens = sum(s.output_tokens or 0 for s in run.steps)
    priced_steps = [s for s in run.steps if s.cost is not None]
    total_cost = sum(s.cost for s in priced_steps) if priced_steps else None
    unpriced_steps = sum(1 for s in run.steps if s.cost is None)

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
                verifier_label = "Independent" if step.verifier_mode in ("challenger", "independent") else "Critic"
                trace = json.loads(step.agent_trace) if step.agent_trace else []
                if "/" in step.step_name:
                    group_name, branch_name = step.step_name.split("/", 1)
                    if group_name not in prior_seen_groups:
                        prior_seen_groups.add(group_name)
                        rerun_prior_items.append({"type": "group_header", "name": group_name})
                    rerun_prior_items.append({
                        "type": "branch", "group": group_name, "name": branch_name,
                        "step": step, "parsed": parsed, "pretty": pretty,
                        "verifier_parsed": verifier_parsed,
                        "verifier_pretty": verifier_pretty, "verifier_label": verifier_label,
                        "trace": trace,
                    })
                else:
                    rerun_prior_items.append({
                        "type": "step", "name": step.step_name,
                        "step": step, "parsed": parsed, "pretty": pretty,
                        "verifier_parsed": verifier_parsed,
                        "verifier_pretty": verifier_pretty, "verifier_label": verifier_label,
                        "trace": trace,
                    })

    # Attach each step's agent-version snapshot (soul.md / agent.yaml as they existed
    # at run time) so the run detail page can show *why* the agent behaved as it did
    # without navigating away to the agent's version history.
    agent_versions_needed = {
        item["step"].agent_version
        for item in (*display_items, *rerun_prior_items)
        if item.get("step") is not None and item["step"].agent_version
    }
    agent_snapshot_by_version: dict[str, AgentVersionSnapshot] = {}
    if agent_versions_needed:
        async with sf() as session:
            snap_rows = await session.execute(
                select(AgentVersionSnapshot)
                .where(AgentVersionSnapshot.agent_version.in_(agent_versions_needed))
            )
            agent_snapshot_by_version = {s.agent_version: s for s in snap_rows.scalars().all()}
    for item in (*display_items, *rerun_prior_items):
        step = item.get("step")
        if step is not None and step.agent_version:
            item["agent_snapshot"] = agent_snapshot_by_version.get(step.agent_version)

    feedback = None
    async with sf() as session:
        result = await session.execute(select(RunFeedback).where(RunFeedback.run_id == run_id))
        feedback = result.scalar_one_or_none()

    step_feedback_by_name: dict[str, dict] = {}
    async with sf() as session:
        result = await session.execute(
            select(StepFeedback.step_name, StepFeedback.outcome, StepFeedback.notes)
            .where(StepFeedback.run_id == run_id)
        )
        for name, outcome, notes in result.all():
            step_feedback_by_name[name] = {"outcome": outcome, "notes": notes or ""}

    from ..executors.human import get_pending_for_run
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
        "total_cost": total_cost,
        "unpriced_steps": unpriced_steps,
        "feedback": feedback,
        "step_feedback_by_name": step_feedback_by_name,
        "pending_approvals": pending_approvals,
        "active_page": "runs",
    })



# --- lines 1253-1311 ---
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



