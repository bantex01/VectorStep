from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import replay
from ..db.database import get_session_factory
from ..replay import AgentNotAllowlisted, CandidateSpec, ReplayNotConfigured, ReplayRequest

from .helpers import templates

router = APIRouter()


@router.get("/replays/new", response_class=HTMLResponse)
async def ui_replay_new(request: Request, step: str = ""):
    """The "Replay against candidate…" form linked from the step insights page
    (SPEC-replay-shadow-eval.md §2 "API-first, minimal UI") — plain HTML form,
    no client-side framework."""
    return templates.TemplateResponse(request, "replay_new.html", {
        "step_name": step,
        "active_page": "replays",
    })


@router.post("/replays", response_class=HTMLResponse)
async def ui_replay_create(request: Request):
    """Synchronous, same as POST /steps/{step_name}/replay — the form's submit
    button blocks until the whole batch has been attempted, then lands on the
    report page.

    Parses the body with Request.form() rather than FastAPI's Form(...)
    parameter dependency — the plain HTML form below posts
    application/x-www-form-urlencoded (no file upload), which Starlette parses
    without the python-multipart package Form(...) unconditionally requires at
    import time. No other route in this codebase takes multipart/form-data;
    this form doesn't need it either.
    """
    body = await request.form()
    step_name = str(body.get("step_name", "")).strip()
    mode = str(body.get("mode", "")).strip()
    model = str(body.get("model", "")).strip()
    agent = str(body.get("agent", "")).strip()
    prompt_template = str(body.get("prompt_template", "")).strip()
    try:
        k = int(body.get("k", 20) or 20)
    except ValueError:
        k = 20

    replay_request = ReplayRequest(
        candidate=CandidateSpec(
            model=model or None,
            agent=agent or None,
            prompt_template=prompt_template or None,
        ),
        mode=mode,
        k=k,
    )

    session_factory = get_session_factory()
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="runner not initialised")

    try:
        run_id = await replay.run_replay_batch(
            session_factory, runner, step_name, replay_request,
            gateway_rest_url=runner._gateway_rest_url,
        )
    except ReplayNotConfigured:
        return templates.TemplateResponse(request, "replay_new.html", {
            "step_name": step_name,
            "active_page": "replays",
            "error": "replay is not configured — set replay.safe_agents in config.yaml",
        }, status_code=403)
    except AgentNotAllowlisted as exc:
        return templates.TemplateResponse(request, "replay_new.html", {
            "step_name": step_name,
            "active_page": "replays",
            "error": f"agent '{exc.agent_key}' is not in replay.safe_agents",
        }, status_code=403)
    except ValueError as exc:
        return templates.TemplateResponse(request, "replay_new.html", {
            "step_name": step_name,
            "active_page": "replays",
            "error": str(exc),
        }, status_code=422)

    return RedirectResponse(url=f"/ui/replays/{run_id}", status_code=303)


@router.get("/replays/{run_id}", response_class=HTMLResponse)
async def ui_replay_report(request: Request, run_id: str):
    """Minimal report page — renders exactly what GET /replays/{run_id}/report
    returns, no separate view logic."""
    session_factory = get_session_factory()
    try:
        report = await replay.get_report(session_factory, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return templates.TemplateResponse(request, "replay_report.html", {
        "run_id": run_id,
        "report": report,
        "active_page": "replays",
    })
