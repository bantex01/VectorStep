import asyncio
import json
import logging
import os
import signal
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from sqlalchemy import or_, select
from sqlalchemy.orm import selectinload

from .db.database import create_tables, get_session_factory, init_db, mark_interrupted_runs
from .db.models import PipelineRun, PipelineStep, RunFeedback
import functools

from .executors import EXECUTORS
from .executors.gateway import GatewayExecutor
from .executors.openclaw_ws import OpenClawWSExecutor, validate_identity
from .metrics import PorkCollector, fetch_metrics_data
from .models.context import NormalisedContext
from .models.pipeline import PipelineConfig
from .normaliser import PARSERS
from .normaliser.alertmanager import AlertmanagerStrategy
from .notifications.log import LogNotifier
from .notifications.telegram import TelegramNotifier
from .pipeline import PipelineRunner, load_pipelines, load_step_library, resolve_pipeline
from .tracing import setup_tracing, shutdown_tracing
from .ui import configure as configure_ui
from .ui import router as ui_router
from .utils import utc_now

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
_slack_poller_tasks: list[asyncio.Task] = []
_artifact_store = None
_artifact_retention_days: int = 7
_dedup_enabled: bool = True
_dedup_window_seconds: int = 300
_webhook_tokens: dict[str, str | None] = {}  # token -> team name (None = legacy/unattributed)
_active_runs: int = 0
_max_concurrent_runs: int = 10


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
    if _runner is not None:
        _runner.set_pipeline_registry({p.name: p for p in _pipelines})
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
        team=schedule.team if schedule else None,
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
    global _runner, _pipelines, _pipeline_dir, _step_library, _step_library_dir, _scheduler, _app_ref, _poller_task, _slack_poller_tasks, _artifact_store, _artifact_retention_days, _dedup_enabled, _dedup_window_seconds, _webhook_tokens, _max_concurrent_runs
    _app_ref = app

    config = _load_config()
    _setup_logging(config)
    setup_tracing(config)

    dedup_cfg = config.get("dedup", {})
    _dedup_enabled = dedup_cfg.get("enabled", True)
    _dedup_window_seconds = dedup_cfg.get("window_seconds", 300)

    auth_cfg = config.get("auth", {}) or {}
    teams_cfg = auth_cfg.get("teams") or []
    legacy_token = auth_cfg.get("token", "") or ""

    _webhook_tokens = {}
    if teams_cfg:
        for entry in teams_cfg:
            token = entry.get("token", "") or ""
            name = entry.get("name", "") or ""
            if not token or not name:
                logger.warning("auth.teams entry missing name or token (after env resolution) — skipping")
                continue
            if token in _webhook_tokens:
                logger.warning("auth.teams token reused by team '%s' — overriding previous team '%s'", name, _webhook_tokens[token])
            _webhook_tokens[token] = name
    elif legacy_token:
        _webhook_tokens[legacy_token] = None

    _max_concurrent_runs = config.get("concurrency", {}).get("max_runs", 10)
    if _webhook_tokens:
        team_count = sum(1 for v in _webhook_tokens.values() if v is not None)
        if team_count:
            logger.info("Webhook auth enabled — %d token(s) configured (%d team(s))", len(_webhook_tokens), team_count)
        else:
            logger.info("Webhook auth enabled — Bearer token required on POST /webhook (legacy single-token, unattributed)")
    else:
        logger.warning("Webhook auth disabled — POST /webhook is unauthenticated")

    # Database
    db_url = config.get("database", {}).get("url", "sqlite+aiosqlite:///./runs.db")
    init_db(db_url)
    await create_tables()
    logger.info("Database initialised: %s", db_url)

    interrupted = await mark_interrupted_runs()
    if interrupted:
        logger.warning("Marked %d run(s) as 'interrupted' after restart", interrupted)

    # Step library (load before pipelines so references can be resolved)
    _step_library_dir = config.get("step_library_dir", "./steps")
    _step_library = load_step_library(_step_library_dir)

    # Pipeline configs
    _pipeline_dir = config.get("pipeline_config_dir", "./pipelines")
    _pipelines = load_pipelines(_pipeline_dir, step_library=_step_library)

    # Notifiers — log is always available; telegram and webhook require config
    from .notifications.webhook import WebhookNotifier
    notifiers = {
        "webhook": WebhookNotifier(),
        "log": LogNotifier(),
    }

    port = config.get("server", {}).get("port", 8000)

    telegram_cfg = config.get("notifications", {}).get("telegram", {})
    bot_token = telegram_cfg.get("bot_token", "")
    chat_id = telegram_cfg.get("chat_id", "")
    if bot_token and chat_id:
        notifiers["telegram"] = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        logger.info("Telegram notifier configured")

        from .notifications.telegram_poller import poll_telegram_updates

        # /run commands POST to our own /webhook over HTTP, so they need a
        # token themselves once auth.teams is configured — same as any other
        # caller. notifications.telegram.team picks which team Telegram-
        # triggered runs are attributed to; falls back to the legacy single
        # token (no team) if auth.teams isn't in use.
        telegram_team = telegram_cfg.get("team", "") or ""
        telegram_webhook_token = None
        if telegram_team:
            telegram_webhook_token = next(
                (e.get("token") for e in teams_cfg if e.get("name") == telegram_team), None
            )
            if not telegram_webhook_token:
                logger.warning(
                    "notifications.telegram.team '%s' not found in auth.teams — "
                    "/run commands will be unauthenticated and may 401", telegram_team,
                )
        elif not teams_cfg and legacy_token:
            telegram_webhook_token = legacy_token

        _poller_task = asyncio.create_task(
            poll_telegram_updates(
                bot_token,
                webhook_base_url=f"http://localhost:{port}",
                allowed_chat_id=chat_id,
                webhook_token=telegram_webhook_token,
            )
        )
    else:
        logger.warning("Telegram notifier not configured — notifications will be skipped")

    # Human-in-the-loop approval channel routing (executor: human) — independent of the
    # notifications.telegram channel above, since a team may use human_approval without
    # any Telegram bot at all (e.g. Slack- or Teams-only). See README §"human executor".
    human_approval_cfg = config.get("human_approval", {})
    ui_base_url = human_approval_cfg.get("ui_base_url") or f"http://localhost:{port}"
    if not human_approval_cfg.get("ui_base_url"):
        logger.warning(
            "human_approval.ui_base_url not set — falling back to %s, which only works "
            "for approvers on the same machine. Set it to a URL reachable by whoever "
            "will click Teams approval links.", ui_base_url,
        )

    from .executors.human import configure as configure_human_executor
    configure_human_executor(
        human_approval=human_approval_cfg,
        legacy_telegram={"bot_token": bot_token, "chat_id": chat_id} if bot_token and chat_id else {},
        ui_base_url=ui_base_url,
    )

    # One Slack Socket Mode listener per distinct app_token referenced across
    # human_approval config — resolves Approve/Reject button clicks, no public endpoint.
    slack_app_tokens = set()
    for team_cfg in human_approval_cfg.get("teams", {}).values():
        if team_cfg.get("channel") == "slack":
            token = team_cfg.get("slack", {}).get("app_token", "")
            if token:
                slack_app_tokens.add(token)
    default_channel_cfg = human_approval_cfg.get("default") or {}
    if default_channel_cfg.get("channel") == "slack":
        token = default_channel_cfg.get("slack", {}).get("app_token", "")
        if token:
            slack_app_tokens.add(token)

    if slack_app_tokens:
        from .notifications.slack_poller import poll_slack_events
        _slack_poller_tasks = [
            asyncio.create_task(poll_slack_events(app_token)) for app_token in slack_app_tokens
        ]
        logger.info("Slack Socket Mode listener(s) started: %d app(s)", len(slack_app_tokens))

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
        team_count=len(teams_cfg),
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
        pipeline_registry={p.name: p for p in _pipelines},
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

    for task in _slack_poller_tasks:
        task.cancel()
    for task in _slack_poller_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass

    shutdown_tracing()


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

async def _find_duplicate_run(
    pipeline_name: str, fingerprint: str, window_seconds: int
) -> PipelineRun | None:
    """Return a recent or in-progress run with the same pipeline+fingerprint, if any.

    A run with status="running" is always considered a duplicate regardless of
    window_seconds — this is the race-prevention case (two overlapping triages for
    the same alert). window_seconds additionally suppresses re-fires shortly after
    a run has completed (Alertmanager repeat-fire on a flapping alert).
    """
    session_factory = get_session_factory()
    cutoff = utc_now() - timedelta(seconds=window_seconds)
    async with session_factory() as session:
        result = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.pipeline_name == pipeline_name)
            .where(PipelineRun.fingerprint == fingerprint)
            .where(or_(PipelineRun.status == "running", PipelineRun.triggered_at > cutoff))
            .order_by(PipelineRun.triggered_at.desc())
        )
        return result.scalars().first()


def _resolve_dedup_settings(pipeline: PipelineConfig) -> tuple[bool, int]:
    """Resolve effective (enabled, window_seconds) for a pipeline.

    trigger.dedup fields override the service-level config.yaml defaults when set;
    None means "fall back to the service default".
    """
    dedup_cfg = pipeline.trigger.dedup
    enabled = _dedup_enabled if dedup_cfg is None or dedup_cfg.enabled is None else dedup_cfg.enabled
    window_seconds = (
        _dedup_window_seconds
        if dedup_cfg is None or dedup_cfg.window_seconds is None
        else dedup_cfg.window_seconds
    )
    return enabled, window_seconds


def _resolve_team(auth_header: str) -> str | None:
    """Resolve the team owning this webhook call from its Bearer token.

    Raises HTTPException(401) if auth is enabled and the token is missing or
    unrecognized. Returns None if auth is disabled, or if the matching token
    is the legacy single-token (unattributed) entry.
    """
    if not _webhook_tokens:
        return None
    presented = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else None
    if presented is None or presented not in _webhook_tokens:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return _webhook_tokens[presented]


@app.post("/webhook")
async def webhook(
    request: Request,
    source: str | None = Query(default=None),
    x_pipeline_source: str | None = Header(default=None),
    strategy: AlertmanagerStrategy = Query(default="most_severe"),
):
    team = _resolve_team(request.headers.get("Authorization", ""))

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
    normalised = normalised.model_copy(update={"team": team})

    logger.info(
        "Webhook received: source=%s pipeline=%s severity=%s",
        normalised.source, normalised.pipeline, normalised.severity,
    )

    return await _trigger_run(normalised)


@app.post("/pipelines/{name}/run")
async def run_pipeline_now(name: str, request: Request):
    """Manually trigger a pipeline run — powers the 'Run now' button on the
    pipeline detail page.

    Deliberately not behind webhook Bearer auth: it's an internal/management
    action on the same trust boundary as /reload and /runs/{id}/rerun, not the
    public ingestion path that auth.teams is meant to gate. Runs triggered
    here are unattributed (team=None) — there's no login/session concept in
    the UI to derive a team from, unlike the webhook's Bearer-token
    attribution (see README §3b).
    """
    payload = await request.json()
    payload = {**payload, "pipeline": name}

    parser_class = PARSERS["generic"]
    try:
        normalised = await parser_class().parse(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info("Manual run triggered from UI: pipeline=%s", normalised.pipeline)

    return await _trigger_run(normalised)


async def _trigger_run(normalised: NormalisedContext) -> JSONResponse:
    """Resolve a pipeline for a normalised context and either return a
    dedup/overload response or kick off a new run. Shared by the public
    /webhook endpoint and the internal 'Run now' UI trigger.
    """
    pipeline = resolve_pipeline(normalised, _pipelines)
    if not pipeline:
        raise HTTPException(
            status_code=422,
            detail=f"No pipeline matched for source='{normalised.source}' "
                   f"severity='{normalised.severity}' labels={normalised.labels}",
        )

    if normalised.fingerprint:
        enabled, window_seconds = _resolve_dedup_settings(pipeline)
        if enabled:
            duplicate = await _find_duplicate_run(pipeline.name, normalised.fingerprint, window_seconds)
            if duplicate:
                logger.info(
                    "Run deduplicated: pipeline=%s fingerprint=%s matches run=%s (status=%s)",
                    pipeline.name, normalised.fingerprint, duplicate.id, duplicate.status,
                )
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "deduplicated",
                        "run_id": duplicate.id,
                        "source": normalised.source,
                        "pipeline": pipeline.name,
                        "severity": normalised.severity,
                        "summary": normalised.summary,
                        "reason": f"Duplicate of run {duplicate.id} (status={duplicate.status})",
                    },
                )

    global _active_runs
    if _active_runs >= _max_concurrent_runs:
        return JSONResponse(
            status_code=429,
            content={
                "status": "overloaded",
                "active_runs": _active_runs,
                "max_concurrent_runs": _max_concurrent_runs,
            },
        )

    # Generate run_id here so it can be returned in the 202 response before the
    # background task starts. The runner accepts a pre-supplied run_id and uses it
    # directly rather than generating a new one.
    run_id = str(uuid.uuid4())
    _active_runs += 1
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


async def _run_pipeline(pipeline, normalised, run_id: str | None = None,
                        from_step: str | None = None, initial_step_outputs: dict | None = None):
    global _active_runs
    from .run_events import publish_complete as _publish_complete
    try:
        result = await _runner.run(pipeline, normalised, run_id=run_id,
                                   from_step=from_step, initial_step_outputs=initial_step_outputs)
        logger.info(
            "Pipeline completed: id=%s pipeline=%s status=%s",
            result.run_id, result.pipeline_name, result.status,
        )
    except Exception as exc:
        logger.error("Pipeline run raised unhandled exception: %s", exc, exc_info=True)
        if run_id:
            _publish_complete(run_id, "failed")
    finally:
        _active_runs -= 1


# ---------------------------------------------------------------------------
# Status endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_runs": _active_runs,
        "max_concurrent_runs": _max_concurrent_runs,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus exposition of run/step counts, durations, and verifier stats."""
    data = await fetch_metrics_data(get_session_factory())
    collector = PorkCollector(data)
    REGISTRY.register(collector)
    try:
        output = generate_latest(REGISTRY)
    finally:
        REGISTRY.unregister(collector)
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)


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
        "team": run.team,
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
    team: str | None = Query(default=None),
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
        if team:
            q = q.where(PipelineRun.team == team)
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


@app.post("/runs/{run_id}/rerun")
async def rerun_from_step(run_id: str, request: Request):
    """Create a new pipeline run starting from a specific step.

    Prior step outputs are loaded from the original run's DB rows and injected
    into the new run so prompt templates referencing {{steps.prior.field}} and
    {{artifacts.prior.key}} resolve correctly without re-executing those steps.
    """
    from .models.llm import LLMOutput as _LLMOutput

    body = await request.json()
    from_step = body.get("from_step", "").strip()
    if not from_step:
        raise HTTPException(status_code=400, detail="from_step is required")

    # Load original run + its steps
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.id == run_id)
            .options(selectinload(PipelineRun.steps))
        )
        original_run = result.scalar_one_or_none()

    if not original_run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    # Resolve current pipeline config
    pipeline = next((p for p in _pipelines if p.name == original_run.pipeline_name), None)
    if not pipeline:
        raise HTTPException(
            status_code=422,
            detail=f"Pipeline '{original_run.pipeline_name}' is not currently loaded",
        )

    # Validate from_step is a sequential step in the pipeline
    from .models.pipeline import StepConfig as _StepConfig
    sequential_names = [s.name for s in pipeline.steps if isinstance(s, _StepConfig)]
    if from_step not in sequential_names:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{from_step}' is not a sequential step in pipeline "
                f"'{pipeline.name}'. Re-run supports sequential steps only. "
                f"Available: {sequential_names}"
            ),
        )

    # Reconstruct prior step outputs from DB rows that precede from_step
    sorted_steps = sorted(original_run.steps, key=lambda s: s.step_index)
    initial_step_outputs: dict[str, _LLMOutput] = {}
    for step_row in sorted_steps:
        if step_row.step_name == from_step:
            break
        if not step_row.parsed_output:
            continue
        parsed = json.loads(step_row.parsed_output)
        output = _LLMOutput.model_validate(parsed)
        if "/" in step_row.step_name:
            # Parallel branch stored as "group/branch" — register under branch name
            # so {{steps.branch_name.field}} resolves correctly downstream
            _, branch_name = step_row.step_name.split("/", 1)
            initial_step_outputs[branch_name] = output
        else:
            initial_step_outputs[step_row.step_name] = output

    # Reconstruct NormalisedContext from the original run, with a fresh timestamp
    # and no fingerprint (re-runs are never deduped against each other)
    nc_raw = json.loads(original_run.normalised_context)
    normalised = NormalisedContext(
        source="rerun",
        pipeline=nc_raw.get("pipeline", ""),
        severity=nc_raw.get("severity"),
        labels=nc_raw.get("labels", {}),
        summary=nc_raw.get("summary"),
        fingerprint=None,
        raw=nc_raw.get("raw", {}),
        metadata={
            **nc_raw.get("metadata", {}),
            "original_run_id": run_id,
            "from_step": from_step,
        },
    )

    global _active_runs
    if _active_runs >= _max_concurrent_runs:
        return JSONResponse(
            status_code=429,
            content={
                "status": "overloaded",
                "active_runs": _active_runs,
                "max_concurrent_runs": _max_concurrent_runs,
            },
        )

    new_run_id = str(uuid.uuid4())
    _active_runs += 1
    asyncio.create_task(
        _run_pipeline(pipeline, normalised, new_run_id,
                      from_step=from_step, initial_step_outputs=initial_step_outputs)
    )

    logger.info(
        "Re-run created: original=%s new=%s from_step=%s prior_steps=%d",
        run_id, new_run_id, from_step, len(initial_step_outputs),
    )

    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "run_id": new_run_id,
            "original_run_id": run_id,
            "from_step": from_step,
            "prior_steps_loaded": len(initial_step_outputs),
        },
    )


@app.post("/runs/{run_id}/feedback")
async def submit_run_feedback(run_id: str, request: Request):
    """Submit or update human accuracy feedback for a completed run."""
    body = await request.json()
    outcome = (body.get("outcome") or "").strip()
    if outcome not in ("correct", "partial", "incorrect"):
        raise HTTPException(status_code=400, detail="outcome must be 'correct', 'partial', or 'incorrect'")
    notes = (body.get("notes") or "").strip() or None

    session_factory = get_session_factory()
    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        result = await session.execute(select(RunFeedback).where(RunFeedback.run_id == run_id))
        feedback = result.scalar_one_or_none()

        if feedback:
            feedback.outcome = outcome
            feedback.notes = notes
            feedback.submitted_at = utc_now()
        else:
            feedback = RunFeedback(
                run_id=run_id,
                pipeline_name=run.pipeline_name,
                outcome=outcome,
                notes=notes,
            )
            session.add(feedback)
        await session.commit()

    return {
        "run_id": run_id,
        "outcome": feedback.outcome,
        "notes": feedback.notes,
        "submitted_at": feedback.submitted_at.isoformat(),
    }


@app.get("/runs/{run_id}/feedback")
async def get_run_feedback(run_id: str):
    """Get human accuracy feedback for a run, if any."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(select(RunFeedback).where(RunFeedback.run_id == run_id))
        feedback = result.scalar_one_or_none()

    if not feedback:
        return {"feedback": None}
    return {
        "feedback": {
            "run_id": feedback.run_id,
            "outcome": feedback.outcome,
            "notes": feedback.notes,
            "submitted_at": feedback.submitted_at.isoformat(),
        }
    }
