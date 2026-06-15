import json
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..utils import utc_now
from .models import Base, PipelineRun

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
        # SQLite raises OperationalError if the column already exists; we ignore that.
        for statement in _MIGRATIONS:
            try:
                await conn.exec_driver_sql(statement)
            except Exception:
                pass


_MIGRATIONS = [
    "ALTER TABLE pipeline_steps ADD COLUMN verifier_mode TEXT",
    "ALTER TABLE pipeline_runs ADD COLUMN logs TEXT",
    "ALTER TABLE pipeline_steps ADD COLUMN artifacts TEXT",
    "ALTER TABLE pipeline_steps ADD COLUMN agent_trace TEXT",
    "ALTER TABLE pipeline_runs ADD COLUMN fingerprint TEXT",
    "CREATE INDEX IF NOT EXISTS ix_pipeline_runs_fingerprint ON pipeline_runs (fingerprint)",
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
