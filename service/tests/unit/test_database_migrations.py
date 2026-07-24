"""Tests for the team column migration (db/database.py _COLUMN_MIGRATIONS)."""
from sqlalchemy import inspect

from src.db.database import create_tables, get_session_factory
from src.db.models import PipelineRun


async def test_team_column_migration_idempotent(db):
    # Running create_tables() twice must not raise — the second run hits the
    # "column already exists" path that's swallowed via OperationalError.
    await create_tables()

    session_factory = get_session_factory()
    async with session_factory() as session:
        conn = await session.connection()
        # sqlalchemy.inspect (not PRAGMA) so this introspects portably on both
        # SQLite and Postgres.
        columns = await conn.run_sync(
            lambda c: {col["name"] for col in inspect(c).get_columns("pipeline_runs")}
        )
        indexes = await conn.run_sync(
            lambda c: {idx["name"] for idx in inspect(c).get_indexes("pipeline_runs")}
        )

    assert "team" in columns
    assert "ix_pipeline_runs_team" in indexes
    assert "stage" in columns
    assert "ix_pipeline_runs_stage" in indexes


async def test_team_column_queryable_after_migration(db):
    await create_tables()

    session_factory = get_session_factory()
    async with session_factory() as session:
        session.add(PipelineRun(
            id="run-1",
            pipeline_name="p",
            source="generic",
            status="completed",
            normalised_context="{}",
            raw_payload="{}",
            team="payments",
        ))
        await session.commit()

        run = await session.get(PipelineRun, "run-1")
        assert run.team == "payments"
