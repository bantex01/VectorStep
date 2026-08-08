import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker, create_async_engine

from ..utils import utc_now
from .models import Base, PendingApproval, PipelineRun

logger = logging.getLogger(__name__)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def init_db(database_url: str) -> None:
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def _alembic_config() -> Config:
    cfg = Config(str(_ALEMBIC_INI))
    # Tells migrations/env.py the exact URL our own engine was built from —
    # the only thing that ever needs a second, short-lived engine of its own
    # (see env.py's run_async_migrations docstring for why it can't just
    # reuse this engine object across threads).
    cfg.attributes["configured_url"] = _engine.url.render_as_string(hide_password=False)
    return cfg


async def _current_revision(conn: AsyncConnection) -> str | None:
    has_version_table = await conn.run_sync(lambda c: inspect(c).has_table("alembic_version"))
    if not has_version_table:
        return None
    result = await conn.exec_driver_sql("SELECT version_num FROM alembic_version")
    row = result.first()
    return row[0] if row else None


def _pending_revisions(cfg: Config, current: str | None, head: str) -> list[str]:
    script = ScriptDirectory.from_config(cfg)
    return [rev.revision for rev in script.iterate_revisions(head, current)]


async def create_tables(auto_migrate: bool = True) -> None:
    """Bring the schema to head, adopting whatever state the DB is already in.

    - Already at head → no-op, regardless of auto_migrate.
    - Behind head and auto_migrate is False → fail fast, naming the pending
      revisions, rather than risk the app running against a stale schema.
    - Behind head and auto_migrate is True: an empty DB (no `pipeline_runs`)
      upgrades from scratch; a DB with tables but no `alembic_version` (every
      pre-Alembic deployment) runs the legacy shim first. The shim's
      `Base.metadata.create_all` builds tables matching *current* db/models.py —
      which the drift test below enforces is always exactly the `head` schema —
      so the DB is stamped at `head`, not the literal "0001_baseline", before the
      (now no-op) upgrade. Stamping at a fixed "0001_baseline" would be correct
      only while that was the sole revision; it broke the moment a second
      revision touched a column the shim's create_all already creates.
    """
    assert _engine is not None, "Database not initialised — call init_db() first"

    cfg = _alembic_config()
    head = ScriptDirectory.from_config(cfg).get_current_head()

    async with _engine.connect() as conn:
        current = await _current_revision(conn)
        has_pipeline_runs = await conn.run_sync(lambda c: inspect(c).has_table("pipeline_runs"))

    if current == head:
        return

    if not auto_migrate:
        pending = _pending_revisions(cfg, current, head)
        raise RuntimeError(
            "Database schema is behind head and database.auto_migrate is false "
            f"(pending revision(s): {', '.join(pending)}). Run "
            "`cd service && alembic upgrade head` to apply them, or set "
            "database.auto_migrate: true to let the app migrate on boot."
        )

    if current is None and has_pipeline_runs:
        async with _engine.begin() as conn:
            await _run_legacy_shim(conn)
        await asyncio.to_thread(command.stamp, cfg, head)

    await asyncio.to_thread(command.upgrade, cfg, "head")


async def _run_legacy_shim(conn: AsyncConnection) -> None:
    """Pre-Alembic migration mechanism — adopts any pre-existing deployment.

    Kept verbatim from the original create_tables() so every historical DB
    shape (any prior version's partial application of these same statements)
    converges on exactly the 0001_baseline schema before being stamped there.
    Idempotent by construction, which is what makes that convergence safe.

    Retained for one release cycle's worth of specs after
    SPEC-alembic-migrations.md lands, then deletable — by then every
    deployment that boots at least once will have been stamped and adopted.
    """
    await conn.run_sync(Base.metadata.create_all)

    # Add columns introduced after initial schema — safe to run on every boot.
    # Postgres supports IF NOT EXISTS directly. SQLite's ALTER TABLE ADD COLUMN
    # has no IF NOT EXISTS form (confirmed unsupported as of SQLite 3.51), so it
    # falls back to attempt-and-ignore-if-already-there, narrowed to
    # OperationalError (the exception SQLite/aiosqlite actually raises for a
    # duplicate column) rather than a bare except, so unrelated DB errors aren't
    # silently swallowed.
    is_postgres = conn.dialect.name == "postgresql"
    for table, column, column_type in _COLUMN_MIGRATIONS:
        if is_postgres:
            await conn.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {column_type}"
            )
        else:
            try:
                await conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                )
            except OperationalError:
                logger.debug("Column %s.%s already exists, skipping", table, column)

    # CREATE [UNIQUE] INDEX IF NOT EXISTS is portable across both dialects.
    for statement in _INDEX_MIGRATIONS:
        await conn.exec_driver_sql(statement)


_COLUMN_MIGRATIONS = [
    ("pipeline_steps", "verifier_mode", "TEXT"),
    ("pipeline_runs", "logs", "TEXT"),
    ("pipeline_steps", "artifacts", "TEXT"),
    ("pipeline_steps", "agent_trace", "TEXT"),
    ("pipeline_runs", "fingerprint", "TEXT"),
    ("pipeline_runs", "parent_run_id", "TEXT"),
    ("pipeline_steps", "input_tokens", "INTEGER"),
    ("pipeline_steps", "output_tokens", "INTEGER"),
    ("pipeline_runs", "team", "TEXT"),
    ("pipeline_steps", "provider", "TEXT"),
    ("pipeline_runs", "stage", "TEXT DEFAULT 'production'"),
    ("pipeline_steps", "grounding_score", "FLOAT"),
    ("pipeline_steps", "trust_report", "TEXT"),
    ("pipeline_steps", "deterministic_passed", "BOOLEAN"),
    ("pipeline_steps", "verifier_agent", "TEXT"),
    ("pipeline_steps", "verifier_model", "TEXT"),
    ("pipeline_steps", "verifier_provider", "TEXT"),
    ("pipeline_steps", "verifier_prompt", "TEXT"),
    ("pipeline_steps", "prompt_hash", "TEXT"),
    ("pipeline_steps", "agent_version", "TEXT"),
    ("pipeline_steps", "verifier_input_tokens", "INTEGER"),
    ("pipeline_steps", "verifier_output_tokens", "INTEGER"),
    ("pipeline_steps", "cost", "FLOAT"),
    ("pipeline_steps", "grounding_model", "TEXT"),
    ("pipeline_steps", "grounding_provider", "TEXT"),
    ("pipeline_steps", "grounding_input_tokens", "INTEGER"),
    ("pipeline_steps", "grounding_output_tokens", "INTEGER"),
]

_INDEX_MIGRATIONS = [
    "CREATE INDEX IF NOT EXISTS ix_pipeline_runs_fingerprint ON pipeline_runs (fingerprint)",
    "CREATE INDEX IF NOT EXISTS ix_pipeline_runs_parent_run_id ON pipeline_runs (parent_run_id)",
    "CREATE INDEX IF NOT EXISTS ix_pipeline_runs_team ON pipeline_runs (team)",
    # Closes the dedup TOCTOU race (README §3a "Known limitation"): the DB itself now
    # refuses a second 'running' row for the same pipeline+fingerprint, regardless of
    # how close together two webhook deliveries land. NULLs are never considered equal
    # in a unique index, so pipelines/sources that opt out of dedup (fingerprint=None,
    # e.g. sub-pipelines, re-runs) are correctly unaffected.
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_pipeline_runs_running_fingerprint "
    "ON pipeline_runs (pipeline_name, fingerprint) WHERE status = 'running'",
    "CREATE INDEX IF NOT EXISTS ix_pipeline_runs_stage ON pipeline_runs (stage)",
    "CREATE INDEX IF NOT EXISTS ix_pipeline_steps_prompt_hash ON pipeline_steps (prompt_hash)",
    "CREATE INDEX IF NOT EXISTS ix_pipeline_steps_agent_version ON pipeline_steps (agent_version)",
]


def _append_log_event(logs: list, level: str, event: str, msg: str) -> None:
    logs.append({
        "ts": utc_now().isoformat(timespec="milliseconds") + "Z",
        "level": level,
        "event": event,
        "msg": msg,
    })


async def sweep_and_partition_running_runs(
    pipelines: "list[PipelineConfig]",
    default_max_resume_age_seconds: int = 3600,
) -> "list[tuple[PipelineConfig, NormalisedContext, str]]":
    """Startup sweep, replacing the old mark_interrupted_runs() (SPEC-durable-runs.md
    §2 "Resume happens in lifespan() startup, replacing part of the
    mark_interrupted_runs() sweep").

    A clean shutdown never leaves a run in 'running' — any such row on startup means
    the process died mid-run. Every such row is partitioned:
      - durable pipeline, config unchanged, within max_resume_age_seconds -> handed
        back as a (pipeline, normalised_context, run_id) tuple, ready to feed straight
        into the same asyncio.create_task(_run_pipeline(..., resume=True)) path a
        fresh run uses. The row itself is left exactly as-is (still 'running' — it
        never left that state, so the dedup partial-unique index keeps suppressing
        duplicate webhooks for it throughout the outage, same as before the crash).
      - everything else -> marked 'interrupted', exactly as mark_interrupted_runs
        always did, with a run-log event naming *why* it wasn't resumed when that's
        knowable (resume_skipped_config_changed / resume_skipped_max_age_exceeded) on
        top of the unconditional run_interrupted event.
    """
    from ..models.context import NormalisedContext
    from ..models.pipeline import PipelineConfig, pipeline_config_fingerprint

    assert _session_factory is not None, "Database not initialised — call init_db() first"
    by_name: dict[str, PipelineConfig] = {p.name: p for p in pipelines}
    resumable: list[tuple[PipelineConfig, NormalisedContext, str]] = []

    async with _session_factory() as session:
        result = await session.execute(
            select(PipelineRun).where(PipelineRun.status == "running")
        )
        runs = result.scalars().all()

        for run in runs:
            pipeline = by_name.get(run.pipeline_name)
            skip_event: str | None = None
            skip_msg = ""

            if pipeline is not None and pipeline.durable is not None:
                current_fp = pipeline_config_fingerprint(pipeline)
                if run.config_fingerprint != current_fp:
                    skip_event = "resume_skipped_config_changed"
                    skip_msg = (
                        "Pipeline config changed while this run was down — resuming "
                        "under different semantics would poison calibration data."
                    )
                else:
                    max_age = pipeline.durable.max_resume_age_seconds
                    if max_age is None:
                        max_age = default_max_resume_age_seconds
                    age_seconds = (utc_now() - run.triggered_at).total_seconds()
                    if age_seconds > max_age:
                        skip_event = "resume_skipped_max_age_exceeded"
                        skip_msg = (
                            f"Run has been down for {age_seconds:.0f}s, exceeding "
                            f"durable.max_resume_age_seconds={max_age}s — not resumed."
                        )
                    else:
                        try:
                            normalised = NormalisedContext.model_validate_json(run.normalised_context)
                        except Exception:
                            skip_event = "resume_skipped_config_changed"
                            skip_msg = "Persisted context could not be reconstructed — not resumed."
                        else:
                            resumable.append((pipeline, normalised, run.id))
                            continue

            logs = json.loads(run.logs) if run.logs else []
            if skip_event:
                _append_log_event(logs, "warn", skip_event, skip_msg)
            run.status = "interrupted"
            run.completed_at = utc_now()
            _append_log_event(
                logs, "warn", "run_interrupted",
                "Service restarted while this run was in progress.",
            )
            run.logs = json.dumps(logs)

        await session.commit()

    return resumable


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    assert _session_factory is not None, "Database not initialised — call init_db() first"
    async with _session_factory() as session:
        yield session


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    assert _session_factory is not None, "Database not initialised — call init_db() first"
    return _session_factory


def get_engine():
    assert _engine is not None, "Database not initialised — call init_db() first"
    return _engine


# ---------------------------------------------------------------------------
# Pending approvals — durable mirror of executors/human.py's in-memory
# _pending_approvals/_pending_meta dicts (SPEC-durable-runs.md §2). Unlike the
# functions above, these degrade to a no-op/empty result rather than asserting
# when no DB is configured: executors/human.py is exercised directly by unit
# tests that never call init_db(), and its own HITL wait logic works with zero
# DB dependency today — this must stay true when no session_factory exists.
# ---------------------------------------------------------------------------


async def save_pending_approval(
    token: str, run_id: str, step_name: str, pipeline_name: str | None,
    message: str, team: str | None, stage: str,
) -> None:
    if _session_factory is None:
        return
    async with _session_factory() as session:
        session.add(PendingApproval(
            token=token, run_id=run_id, step_name=step_name, pipeline_name=pipeline_name,
            message=message, team=team, stage=stage,
        ))
        await session.commit()


async def delete_pending_approval(token: str) -> None:
    if _session_factory is None:
        return
    async with _session_factory() as session:
        row = await session.get(PendingApproval, token)
        if row is not None:
            await session.delete(row)
            await session.commit()


async def get_pending_approval(token: str) -> PendingApproval | None:
    if _session_factory is None:
        return None
    async with _session_factory() as session:
        return await session.get(PendingApproval, token)


async def mark_run_resumed(run_id: str) -> None:
    """Stamp resumed_at the first time a run is resumed (SPEC-durable-runs.md) —
    never cleared, and never overwritten on a second resume of the same run.
    Drives vectorstep_runs_resumed_total in metrics.py.
    """
    if _session_factory is None:
        return
    async with _session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        if run is not None and run.resumed_at is None:
            run.resumed_at = utc_now()
            await session.commit()


async def get_pending_approvals_for_run(run_id: str) -> list[PendingApproval]:
    if _session_factory is None:
        return []
    async with _session_factory() as session:
        result = await session.execute(
            select(PendingApproval).where(PendingApproval.run_id == run_id)
        )
        return list(result.scalars().all())
