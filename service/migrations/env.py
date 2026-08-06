import asyncio
import os
from logging.config import fileConfig

import yaml
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from src.db.models import Base

config = context.config

# disable_existing_loggers=False: the app's own logging config is already live
# by the time this runs from create_tables() on every boot — the fileConfig
# default of disabling every pre-existing logger would silently kill it.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _resolve_url() -> str:
    """Resolve the database URL to migrate.

    database.py sets config.attributes["configured_url"] to the exact URL the
    app's own engine was built from before invoking alembic.command.upgrade/
    stamp — used for every programmatic call (normal boot and the test suite).
    Only a standalone CLI invocation (`alembic upgrade head` run directly by
    an operator, e.g. with database.auto_migrate: false) falls through to
    VECTORSTEP_TEST_DATABASE_URL / config.yaml, matching how tests/conftest.py
    picks a backend.
    """
    configured = config.attributes.get("configured_url")
    if configured:
        return configured

    env_override = os.environ.get("VECTORSTEP_TEST_DATABASE_URL")
    if env_override:
        return env_override

    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    url = raw.get("database", {}).get("url", "sqlite+aiosqlite:///./runs.db")
    if isinstance(url, str) and url.startswith("${") and url.endswith("}"):
        url = os.environ.get(url[2:-1], "")
    return url


def run_migrations_offline() -> None:
    """Run migrations without a live DBAPI connection, emitting raw SQL."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode against a short-lived async engine.

    Always builds its own engine (NullPool, disposed immediately after) rather
    than reusing the app's live engine object: database.py invokes this via
    alembic.command.upgrade/stamp, which are sync entry points that call
    asyncio.run() internally (see the async template this file is based on) —
    run from the app's already-running event loop, that call has to happen on
    a worker thread (asyncio.to_thread), so a shared engine would end up with
    connections checked out across two different event loops.
    """
    connectable = async_engine_from_config(
        {"sqlalchemy.url": _resolve_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
