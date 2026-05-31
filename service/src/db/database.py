from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base

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
]


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    assert _session_factory is not None, "Database not initialised — call init_db() first"
    async with _session_factory() as session:
        yield session


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    assert _session_factory is not None, "Database not initialised — call init_db() first"
    return _session_factory
