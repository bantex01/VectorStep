import json
import logging
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..utils import utc_now
from .models import Base, PipelineRun

logger = logging.getLogger(__name__)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_db(database_url: str) -> None:
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def create_tables() -> None:
    assert _engine is not None, "Database not initialised — call init_db() first"
    async with _engine.begin() as conn:
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
]


async def mark_interrupted_runs() -> int:
    """Sweep runs left in 'running' state after a crash or forced restart.

    A clean shutdown never leaves a run in 'running' — any such row on startup
    means the process died mid-run. Mark it 'interrupted' so it stops showing
    as in-progress forever and doesn't skew success-rate stats.
    """
    assert _session_factory is not None, "Database not initialised — call init_db() first"
    async with _session_factory() as session:
        result = await session.execute(
            select(PipelineRun).where(PipelineRun.status == "running")
        )
        runs = result.scalars().all()
        for run in runs:
            run.status = "interrupted"
            run.completed_at = utc_now()
            logs = json.loads(run.logs) if run.logs else []
            logs.append({
                "ts": utc_now().isoformat(timespec="milliseconds") + "Z",
                "level": "warn",
                "event": "run_interrupted",
                "msg": "Service restarted while this run was in progress.",
            })
            run.logs = json.dumps(logs)
        await session.commit()
        return len(runs)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    assert _session_factory is not None, "Database not initialised — call init_db() first"
    async with _session_factory() as session:
        yield session


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    assert _session_factory is not None, "Database not initialised — call init_db() first"
    return _session_factory
