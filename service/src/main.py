import asyncio
import json
import logging
import os
import signal
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .db.database import create_tables, get_session_factory, init_db
from .db.models import PipelineRun, PipelineStep
import functools

from .executors import EXECUTORS
from .executors.gateway import GatewayExecutor
from .executors.openclaw_ws import OpenClawWSExecutor, validate_identity
from .models.context import NormalisedContext
from .normaliser import PARSERS
from .normaliser.alertmanager import AlertmanagerStrategy
from .notifications.telegram import TelegramNotifier
from .pipeline import PipelineRunner, load_pipelines, load_step_library, resolve_pipeline
from .ui import configure as configure_ui
from .ui import router as ui_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _setup_logging(config: dict) -> None:
    """Reconfigure logging from config.yaml after startup.

    - service.log  — all application logs (stdout + rotating file if logging.dir is set)
    - access.log   — uvicorn HTTP access logs (file only; excluded from stdout)
    """
    from logging.handlers import RotatingFileHandler

    log_cfg = config.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_dir = log_cfg.get("dir", "")

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    if log_dir:
        abs_log_dir = os.path.abspath(log_dir)
        os.makedirs(abs_log_dir, exist_ok=True)
        service_handler = RotatingFileHandler(
            os.path.join(abs_log_dir, "service.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        service_handler.setFormatter(formatter)
        root.addHandler(service_handler)
        logging.getLogger(__name__).info("File logging enabled: %s", abs_log_dir)
    else:
        logging.getLogger(__name__).info("File logging disabled (logging.dir not set)")

    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Detach uvicorn.access from the root logger so HTTP request lines don't
    # appear in stdout or service.log. Route them to access.log if a log dir is set.
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.propagate = False
    access_logger.handlers.clear()
    if log_dir:
        access_handler = RotatingFileHandler(
            os.path.join(abs_log_dir, "access.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        access_handler.setFormatter(formatter)
        access_logger.addHandler(access_handler)
    else:
        access_logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# App state — populated at startup
# ---------------------------------------------------------------------------

_runner: PipelineRunner | None = None
_pipelines = []
_pipeline_dir: str = "./pipelines"
_step_library: dict = {}
_step_library_dir: str = "./steps"
_scheduler: AsyncIOScheduler | None = None
_app_ref: "FastAPI | None" = None
_poller_task: asyncio.Task | None = None
_artifact_store = None
_artifact_retention_days: int = 7


def _load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Resolve ${ENV_VAR} placeholders
    def _resolve(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_key = value[2:-1]
            return os.environ.get(env_key, "")
        return value

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(i) for i in obj]
        return _resolve(obj)

    return _walk(raw)


def _register_schedules(pipelines: list) -> None:
    """Remove existing pipeline jobs and re-add from current configs."""
    if _scheduler is None:
        return
    for job in _scheduler.get_jobs():
        if job.id.startswith("pipeline:"):
            job.remove()

    for pipeline in pipelines:
        if pipeline.schedule is None:
            continue
        try:
            _scheduler.add_job(
                _run_scheduled_pipeline,
                CronTrigger.from_crontab(pipeline.schedule.cron),
                id=f"pipeline:{pipeline.name}",
                name=pipeline.name,
                args=[pipeline.name],
                replace_existing=True,
            )
            logger.info(
                "Scheduled pipeline '%s' — cron: %s",
                pipeline.name, pipeline.schedule.cron,
            )
        except Exception as exc:
            logger.error("Failed to schedule pipeline '%s': %s", pipeline.name, exc)


def _do_reload() -> int:
    global _pipelines, _step_library
    _step_library = load_step_library(_step_library_dir)
    _pipelines = load_pipelines(_pipeline_dir, step_library=_step_library)
    _register_schedules(_pipelines)
    if _app_ref:
        _app_ref.state.step_library = _step_library
        _app_ref.state.pipelines = _pipelines
    logger.info(
        "Reloaded — %d pipeline(s), %d library step(s)",
        len(_pipelines), len(_step_library),
    )
    return len(_pipelines)


async def _run_scheduled_pipeline(pipeline_name: str) -> None:
    """Synthesise a NormalisedContext and fire a scheduled pipeline run."""
    pipeline = next((p for p in _pipelines if p.name == pipeline_name), None)
    if pipeline is None:
        logger.error("Scheduled pipeline '%s' not found — skipping", pipeline_name)
        return

    schedule = pipeline.schedule
    labels = dict(schedule.labels) if schedule else {}

    normalised = NormalisedContext(
        source="scheduler",
        pipeline=pipeline_name,
        severity=schedule.severity if schedule else "info",
        labels=labels,
        summary=schedule.summary if schedule else f"Scheduled run of {pipeline_name}",
        raw={},
        metadata={},
        received_at=datetime.now(timezone.utc),
    )

    logger.info("Scheduled trigger firing for pipeline '%s'", pipeline_name)
    await _run_pipeline(pipeline, normalised)


async def _cleanup_artifacts() -> None:
    from datetime import timedelta, timezone
    if _artifact_store is None:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=_artifact_retention_days)
    run_ids = await _artifact_store.runs_older_than(cutoff)
    if not run_ids:
        return
    for run_id in run_ids:
        await _artifact_store.delete_run(run_id)
        logger.info("Artifact cleanup: removed run %s", run_id)
    logger.info("Artifact cleanup complete: removed %d run(s)", len(run_ids))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runner, _pipelines, _pipeline_dir, _step_library, _step_library_dir, _scheduler, _app_ref, _poller_task, _artifact_store, _artifact_retention_days
    _app_ref = app

    config = _load_config()
    _setup_logging(config)

    # Database
    db_url = config.get("database", {}).get("url", "sqlite+aiosqlite:///./runs.db")
    init_db(db_url)
    await create_tables()
    logger.info("Database initialised: %s", db_url)

    # Step library (load before pipelines so references can be resolved)
    _step_library_dir = config.get("step_library_dir", "./steps")
    _step_library = load_step_library(_step_library_dir)

    # Pipeline configs
    _pipeline_dir = config.get("pipeline_config_dir", "./pipelines")
    _pipelines = load_pipelines(_pipeline_dir, step_library=_step_library)

    # Notifiers
    from .notifications.webhook import WebhookNotifier
    notifiers = {"webhook": WebhookNotifier()}

    telegram_cfg = config.get("notifications", {}).get("telegram", {})
    bot_token = telegram_cfg.get("bot_token", "")
    chat_id = telegram_cfg.get("chat_id", "")
    if bot_token and chat_id:
        notifiers["telegram"] = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        logger.info("Telegram notifier configured")

        from .executors.human import configure as configure_human_executor
        configure_human_executor(bot_token=bot_token, chat_id=chat_id)

        from .notifications.telegram_poller import poll_telegram_updates
        port = config.get("server", {}).get("port", 8000)
        _poller_task = asyncio.create_task(
            poll_telegram_updates(
                bot_token,
                webhook_base_url=f"http://localhost:{port}",
                allowed_chat_id=chat_id,
            )
        )
    else:
        logger.warning("Telegram notifier not configured — notifications will be skipped")

    # Build executor registry — both gateway executors accept URL + credentials from config
    executors_cfg = config.get("executors", {})
    gateway_cfg = executors_cfg.get("gateway", {})
    openclaw_cfg = executors_cfg.get("openclaw", {})

    executors = dict(EXECUTORS)

    if gateway_cfg.get("url"):
        executors["gateway"] = functools.partial(
            GatewayExecutor,
            url=gateway_cfg["url"],
            token=gateway_cfg.get("token", ""),
        )
        logger.info("P-Ork Gateway executor configured: %s", gateway_cfg["url"])
    else:
        logger.warning("P-Ork Gateway executor not configured — executors.gateway.url missing")

    openclaw_url = openclaw_cfg.get("url", "")
    openclaw_identity_dir = openclaw_cfg.get("identity_dir", "")
    if openclaw_url:
        try:
            validate_identity(openclaw_identity_dir)
            logger.info("OpenClaw executor configured: %s", openclaw_url)
        except FileNotFoundError as exc:
            logger.warning(
                "OpenClaw executor configured but identity files are missing — "
                "steps using executor: openclaw will fail until this is resolved. %s", exc
            )
        executors["openclaw"] = functools.partial(
            OpenClawWSExecutor,
            gateway_url=openclaw_url,
            identity_dir=openclaw_identity_dir,
        )

    # Configure agent source URLs for the UI layer
    pork_gateway_rest = gateway_cfg.get("rest_url") or os.environ.get("PORK_GATEWAY_URL", "http://localhost:18780")
    configure_ui(
        openclaw_ws_url=openclaw_url,
        pork_gateway_base=pork_gateway_rest,
    )
    logger.info(
        "UI agent sources — openclaw: %s, pork-gateway: %s",
        openclaw_url or "(default ws://127.0.0.1:18789/rpc)",
        pork_gateway_rest,
    )

    # Artifact store
    artifacts_cfg = config.get("artifacts", {})
    _artifact_retention_days = int(artifacts_cfg.get("retention_days", 7))
    if "dir" in artifacts_cfg:
        from .artifacts import LocalArtifactStore
        _artifact_store = LocalArtifactStore(base_dir=artifacts_cfg["dir"])
        logger.info(
            "Artifact store configured: %s (retention: %d days)",
            artifacts_cfg["dir"], _artifact_retention_days,
        )
    else:
        _artifact_store = None

    # Runner
    _runner = PipelineRunner(
        executors=executors,
        session_factory=get_session_factory(),
        notifiers=notifiers,
        artifact_store=_artifact_store,
    )

    # Scheduler
    app.state.pipelines = _pipelines
    app.state.pipeline_dir = _pipeline_dir
    app.state.step_library = _step_library
    app.state.step_library_dir = _step_library_dir

    _scheduler = AsyncIOScheduler()
    _register_schedules(_pipelines)
    if _artifact_store is not None:
        _scheduler.add_job(
            _cleanup_artifacts,
            CronTrigger.from_crontab("0 2 * * *"),
            id="artifact:cleanup",
            replace_existing=True,
        )
        logger.info(
            "Artifact cleanup scheduled (daily at 02:00, retention: %d days)",
            _artifact_retention_days,
        )
    _scheduler.start()
    app.state.scheduler = _scheduler
    logger.info("Scheduler started")

    # SIGHUP triggers a pipeline config reload without restarting the process
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(
        signal.SIGHUP,
        lambda: asyncio.create_task(_sighup_reload()),
    )

    logger.info("Service ready — %d pipeline(s) loaded", len(_pipelines))
    yield

    _scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")

    if _poller_task:
        _poller_task.cancel()
        try:
            await _poller_task
        except asyncio.CancelledError:
            pass


async def _sighup_reload() -> None:
    try:
        count = _do_reload()
        logger.info("SIGHUP: reloaded %d pipeline(s)", count)
    except Exception as exc:
        logger.error("SIGHUP reload failed: %s", exc)


app = FastAPI(title="Porc", description="Pipeline Orchestration Service", lifespan=lifespan)
app.include_router(ui_router)


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@app.post("/webhook")
async def webhook(
    request: Request,
    source: str | None = Query(default=None),
    x_pipeline_source: str | None = Header(default=None),
    strategy: AlertmanagerStrategy = Query(default="most_severe"),
):
    payload = await request.json()

    resolved_source = source or x_pipeline_source
    if not resolved_source:
        raise HTTPException(
            status_code=400,
            detail="Source must be provided via ?source= query param or X-Pipeline-Source header",
        )

    parser_class = PARSERS.get(resolved_source)
    if not parser_class:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source '{resolved_source}'. Registered sources: {list(PARSERS.keys())}",
        )

    parser_kwargs = {}
    if resolved_source == "alertmanager":
        parser_kwargs["strategy"] = strategy

    parser = parser_class(**parser_kwargs)
    normalised = await parser.parse(payload)

    logger.info(
        "Webhook received: source=%s pipeline=%s severity=%s",
        normalised.source, normalised.pipeline, normalised.severity,
    )

    pipeline = resolve_pipeline(normalised, _pipelines)
    if not pipeline:
        raise HTTPException(
            status_code=422,
            detail=f"No pipeline matched for source='{normalised.source}' "
                   f"severity='{normalised.severity}' labels={normalised.labels}",
        )

    # Generate run_id here so it can be returned in the 202 response before the
    # background task starts. The runner accepts a pre-supplied run_id and uses it
    # directly rather than generating a new one.
    run_id = str(uuid.uuid4())
    asyncio.create_task(_run_pipeline(pipeline, normalised, run_id))

    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "run_id": run_id,
            "source": normalised.source,
            "pipeline": pipeline.name,
            "severity": normalised.severity,
            "summary": normalised.summary,
        },
    )


async def _run_pipeline(pipeline, normalised, run_id: str | None = None):
    from .run_events import publish_complete as _publish_complete
    try:
        result = await _runner.run(pipeline, normalised, run_id=run_id)
        logger.info(
            "Pipeline completed: id=%s pipeline=%s status=%s",
            result.run_id, result.pipeline_name, result.status,
        )
    except Exception as exc:
        logger.error("Pipeline run raised unhandled exception: %s", exc, exc_info=True)
        if run_id:
            _publish_complete(run_id, "failed")


# ---------------------------------------------------------------------------
# Status endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/pipelines")
async def list_pipelines():
    return {
        "pipelines": [
            {"name": p.name, "description": p.description, "version": p.version}
            for p in _pipelines
        ]
    }


@app.post("/reload")
async def reload_pipelines():
    """Re-read all YAML pipeline configs from disk without restarting the service."""
    try:
        count = _do_reload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")
    return {"status": "reloaded", "pipelines_loaded": count}


@app.get("/schedules")
async def list_schedules():
    """List all active scheduled pipeline jobs and their next fire times."""
    if _scheduler is None:
        return {"schedules": []}
    jobs = [
        job for job in _scheduler.get_jobs()
        if job.id.startswith("pipeline:")
    ]
    return {
        "schedules": [
            {
                "pipeline": job.name,
                "cron": next(
                    (p.schedule.cron for p in _pipelines if p.name == job.name and p.schedule),
                    None,
                ),
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in jobs
        ]
    }


# ---------------------------------------------------------------------------
# Runs API
# ---------------------------------------------------------------------------

def _format_step(step: PipelineStep) -> dict:
    return {
        "name": step.step_name,
        "index": step.step_index,
        "executor": step.executor,
        "agent": step.agent,
        "model": step.model,
        "status": step.status,
        "primary_confidence": step.primary_confidence,
        "verifier_confidence": step.verifier_confidence,
        "effective_confidence": step.effective_confidence,
        "duration_ms": step.duration_ms,
        "executed_at": step.executed_at.isoformat(),
        "parsed_output": json.loads(step.parsed_output) if step.parsed_output else None,
        "artifacts": json.loads(step.artifacts) if step.artifacts else None,
    }


def _format_run_summary(run: PipelineRun) -> dict:
    return {
        "id": run.id,
        "pipeline_name": run.pipeline_name,
        "source": run.source,
        "status": run.status,
        "triggered_at": run.triggered_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


def _format_run_detail(run: PipelineRun) -> dict:
    return {
        **_format_run_summary(run),
        "normalised_context": json.loads(run.normalised_context),
        "steps": [_format_step(s) for s in run.steps],
    }


@app.get("/runs")
async def list_runs(
    status: str | None = Query(default=None),
    pipeline: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    session_factory = get_session_factory()
    async with session_factory() as session:
        q = select(PipelineRun).order_by(PipelineRun.triggered_at.desc())
        if status:
            q = q.where(PipelineRun.status == status)
        if pipeline:
            q = q.where(PipelineRun.pipeline_name == pipeline)
        q = q.limit(limit).offset(offset)
        result = await session.execute(q)
        runs = result.scalars().all()

    return {
        "runs": [_format_run_summary(r) for r in runs],
        "limit": limit,
        "offset": offset,
    }


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.id == run_id)
            .options(selectinload(PipelineRun.steps))
        )
        run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return _format_run_detail(run)
