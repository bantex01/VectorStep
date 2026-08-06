"""Tests for the Alembic adoption logic in db/database.py (SPEC-alembic-migrations.md §5).

Uses the `raw_db` fixture (schema not yet applied) where a test needs to control
the DB's starting shape itself, and `db` (already at head) otherwise. Both run
under SQLite by default and Postgres when VECTORSTEP_TEST_DATABASE_URL is set —
see tests/conftest.py.
"""
import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext

from src.db.database import create_tables, get_engine, get_session_factory, _run_legacy_shim
from src.db.models import Base, PipelineRun


async def test_fresh_empty_db_boots_to_head(db):
    engine = get_engine()
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
        indexes = await conn.run_sync(
            lambda c: {idx["name"] for idx in inspect(c).get_indexes("pipeline_runs")}
        )

    assert {
        "alembic_version", "pipeline_runs", "pipeline_steps", "run_feedback",
        "step_prompt_versions", "agent_version_snapshots", "step_feedback",
    } <= tables
    assert "ix_pipeline_runs_running_fingerprint" in indexes


async def test_running_fingerprint_partial_index_enforces(db):
    # DB-level half of the dedup TOCTOU fix (README §3a) — the application-level
    # pre-check narrows the race, this constraint is what actually closes it.
    session_factory = get_session_factory()

    async with session_factory() as session:
        session.add(PipelineRun(
            id="run-a", pipeline_name="p", source="generic", status="running",
            normalised_context="{}", raw_payload="{}", fingerprint="fp-1",
        ))
        await session.commit()

    async with session_factory() as session:
        session.add(PipelineRun(
            id="run-b", pipeline_name="p", source="generic", status="running",
            normalised_context="{}", raw_payload="{}", fingerprint="fp-1",
        ))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


async def test_legacy_adoption_stamps_baseline_without_losing_data(raw_db):
    engine = get_engine()

    # Build a DB "the old way" — the pre-Alembic mechanism, still importable
    # as a private function purely to adopt deployments that predate this spec.
    async with engine.begin() as conn:
        await _run_legacy_shim(conn)

    session_factory = get_session_factory()
    async with session_factory() as session:
        session.add(PipelineRun(
            id="pre-migration-run", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", team="payments",
        ))
        await session.commit()

    async with engine.connect() as conn:
        has_alembic_version = await conn.run_sync(lambda c: inspect(c).has_table("alembic_version"))
    assert not has_alembic_version

    await create_tables()

    async with engine.connect() as conn:
        result = await conn.exec_driver_sql("SELECT version_num FROM alembic_version")
        stamped_at = result.first()[0]
    assert stamped_at == "0001_baseline"

    async with session_factory() as session:
        run = await session.get(PipelineRun, "pre-migration-run")
    assert run is not None
    assert run.team == "payments"

    # Subsequent boot is a no-op — nothing re-applied, nothing raised.
    await create_tables()


async def test_auto_migrate_false_raises_naming_pending_revision(raw_db):
    with pytest.raises(RuntimeError, match="0001_baseline"):
        await create_tables(auto_migrate=False)


async def test_auto_migrate_false_at_head_boots_fine(db):
    # `db` fixture already brought this DB to head via the default auto_migrate=True path.
    await create_tables(auto_migrate=False)


async def test_autogenerate_produces_no_diff_at_head(db):
    """Drift guard: Base.metadata and the DB must never disagree at head.

    Fails CI the moment a model field is added/changed without a matching
    Alembic revision — the whole point of adopting Alembic in the first place.
    """
    engine = get_engine()

    def _diff(sync_conn):
        context = MigrationContext.configure(sync_conn, opts={"compare_type": True})
        return compare_metadata(context, Base.metadata)

    async with engine.connect() as conn:
        diffs = await conn.run_sync(_diff)

    assert diffs == [], (
        "autogenerate detected drift between Base.metadata and the DB at head — "
        f"a model change shipped without a matching Alembic revision: {diffs}"
    )
