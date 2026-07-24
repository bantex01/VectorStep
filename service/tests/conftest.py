"""Shared test configuration for the P-Ork service test suite.

Stubs out optional/heavy native dependencies that aren't installed in the
standard dev environment so executor modules can be imported freely in tests.
The real modules are only needed when actually connecting to OpenClaw — all
OpenClaw-related tests should be integration tests in a fully provisioned env.
"""
import os
import sys
from unittest.mock import MagicMock

for _mod in [
    "cryptography",
    "cryptography.hazmat",
    "cryptography.hazmat.primitives",
    "cryptography.hazmat.primitives.serialization",
    "websockets",
    "websockets.legacy",
    "websockets.legacy.client",
]:
    sys.modules.setdefault(_mod, MagicMock())

import pytest

from src.db.database import create_tables, get_engine, init_db

PORK_TEST_DATABASE_URL = os.environ.get("PORK_TEST_DATABASE_URL")


@pytest.fixture
async def db(tmp_path):
    """Initialise an isolated, ready-to-use database for a test.

    Defaults to a SQLite file scoped to pytest's tmp_path — same behaviour as
    before this fixture existed: fast, zero-infra, free per-test isolation.

    Set PORK_TEST_DATABASE_URL (e.g.
    postgresql+asyncpg://user:pass@localhost:5432/pork_test) to run the exact
    same test bodies against Postgres instead. Postgres has no per-test temp
    file, so isolation is created explicitly by dropping and recreating the
    public schema around each test — this also means create_tables() runs
    against Postgres on every test, exercising its ADD COLUMN IF NOT EXISTS
    migration branch, which is otherwise never touched by SQLite-only tests.
    """
    if PORK_TEST_DATABASE_URL:
        init_db(PORK_TEST_DATABASE_URL)
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.exec_driver_sql("DROP SCHEMA public CASCADE")
            await conn.exec_driver_sql("CREATE SCHEMA public")
        await create_tables()
        yield
        await engine.dispose()
    else:
        init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
        await create_tables()
        yield
