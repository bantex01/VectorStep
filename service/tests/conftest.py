"""Shared test configuration for the VectorStep service test suite.

Stubs out optional/heavy native dependencies that aren't installed in the
standard dev environment so executor modules can be imported freely in tests.
The real modules are only needed when actually connecting to OpenClaw — all
OpenClaw-related tests should be integration tests in a fully provisioned env.
"""
import os
import sys
from unittest.mock import MagicMock

# TestClient(app) runs the real FastAPI lifespan, which loads config.yaml —
# gitignored (host-owned, never committed), so give tests a fixture config
# instead. setdefault: an explicitly-set CONFIG_PATH (e.g. a dev override) wins.
os.environ.setdefault(
    "CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "fixtures", "test_config.yaml"),
)

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

VECTORSTEP_TEST_DATABASE_URL = os.environ.get("VECTORSTEP_TEST_DATABASE_URL")


@pytest.fixture
async def raw_db(tmp_path):
    """Initialise an isolated, empty database for a test — schema not applied.

    Same backend-selection contract as `db` below (defaults to a per-test
    SQLite tmp file; set VECTORSTEP_TEST_DATABASE_URL to run against Postgres
    instead, isolated via a schema drop/recreate since there's no per-test
    temp file on a shared server). Split out from `db` so migration tests can
    drive create_tables() themselves — e.g. to build a pre-Alembic DB shape
    first — instead of it running as part of fixture setup.
    """
    if VECTORSTEP_TEST_DATABASE_URL:
        init_db(VECTORSTEP_TEST_DATABASE_URL)
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.exec_driver_sql("DROP SCHEMA public CASCADE")
            await conn.exec_driver_sql("CREATE SCHEMA public")
        yield
        await engine.dispose()
    else:
        init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
        yield


@pytest.fixture
async def db(raw_db):
    """Initialise an isolated, ready-to-use (schema-at-head) database for a test.

    Runs create_tables() against whatever `raw_db` set up — a fresh SQLite tmp
    file by default, or Postgres when VECTORSTEP_TEST_DATABASE_URL is set,
    exercising the Postgres-specific paths (e.g. the legacy shim's
    ADD COLUMN IF NOT EXISTS branch) that SQLite-only test runs never touch.
    """
    await create_tables()
    yield
