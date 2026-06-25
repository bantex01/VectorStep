"""Tests for the team column migration (db/database.py _COLUMN_MIGRATIONS)."""
from src.db.database import create_tables, init_db, get_session_factory
from src.db.models import PipelineRun


async def test_team_column_migration_idempotent(tmp_path):
    db_path = tmp_path / "runs.db"
    init_db(f"sqlite+aiosqlite:///{db_path}")

    # Running create_tables() twice must not raise — the second run hits the
    # "column already exists" path that's swallowed via OperationalError.
    await create_tables()
    await create_tables()

    session_factory = get_session_factory()
    async with session_factory() as session:
        conn = await session.connection()
        columns_result = await conn.exec_driver_sql("PRAGMA table_info(pipeline_runs)")
        columns = {row[1] for row in columns_result.fetchall()}
        indexes_result = await conn.exec_driver_sql("PRAGMA index_list(pipeline_runs)")
        indexes = {row[1] for row in indexes_result.fetchall()}

    assert "team" in columns
    assert "ix_pipeline_runs_team" in indexes


async def test_team_column_queryable_after_migration(tmp_path):
    db_path = tmp_path / "runs.db"
    init_db(f"sqlite+aiosqlite:///{db_path}")
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
