"""Tests for the team column migration (db/database.py _COLUMN_MIGRATIONS)."""
from sqlalchemy import inspect

from src.db.database import create_tables, get_session_factory
from src.db.models import PipelineRun, PipelineStep


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


async def test_prompt_hash_and_agent_version_column_migration_idempotent(db):
    # SPEC-prompt-versioning.md §4b — same idempotency contract as the team-column
    # migration above, for the two columns/indexes this spec adds to pipeline_steps.
    await create_tables()
    await create_tables()

    session_factory = get_session_factory()
    async with session_factory() as session:
        conn = await session.connection()
        columns = await conn.run_sync(
            lambda c: {col["name"] for col in inspect(c).get_columns("pipeline_steps")}
        )
        indexes = await conn.run_sync(
            lambda c: {idx["name"] for idx in inspect(c).get_indexes("pipeline_steps")}
        )

    assert "prompt_hash" in columns
    assert "agent_version" in columns
    assert "ix_pipeline_steps_prompt_hash" in indexes
    assert "ix_pipeline_steps_agent_version" in indexes


async def test_prompt_hash_and_agent_version_queryable_after_migration(db):
    await create_tables()

    session_factory = get_session_factory()
    async with session_factory() as session:
        session.add(PipelineRun(
            id="run-2", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}",
        ))
        session.add(PipelineStep(
            run_id="run-2", step_name="investigate", step_index=0, executor="gateway",
            prompt="rendered prompt", status="completed",
            prompt_hash="a3f2c9d81e04", agent_version="91f02ab3c7de",
        ))
        await session.commit()

        result = await session.execute(
            PipelineStep.__table__.select().where(PipelineStep.run_id == "run-2")
        )
        row = result.one()
        assert row.prompt_hash == "a3f2c9d81e04"
        assert row.agent_version == "91f02ab3c7de"


async def test_new_prompt_version_tables_created(db):
    from src.db.models import AgentVersionSnapshot, StepPromptVersion

    session_factory = get_session_factory()
    async with session_factory() as session:
        session.add(StepPromptVersion(
            prompt_hash="a3f2c9d81e04", step_name="investigate", template="You are triaging...",
        ))
        session.add(AgentVersionSnapshot(
            agent_version="91f02ab3c7de", agent="gateway:sre-investigation",
            soul_md="You are an SRE.", agent_yaml="name: sre-investigation\n",
        ))
        await session.commit()

        spv = await session.get(StepPromptVersion, "a3f2c9d81e04")
        avs = await session.get(AgentVersionSnapshot, "91f02ab3c7de")
        assert spv.step_name == "investigate"
        assert avs.agent == "gateway:sre-investigation"
