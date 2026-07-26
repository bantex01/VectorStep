"""Tests for analytics.get_step_versions/get_agent_versions and the
GET /steps/{name}/versions, GET /agents/{name}/versions endpoints
(SPEC-prompt-versioning.md §5a/§5c/§5d) — "did that prompt edit actually help?"
and "what does this agent_version's config actually look like?" made answerable
with data instead of an opaque hex string."""
from datetime import datetime, timedelta

import httpx
import pytest

import src.main as main
from src.analytics import get_agent_versions, get_step_versions
from src.db.database import get_session_factory
from src.db.models import (
    AgentVersionSnapshot,
    PipelineRun,
    PipelineStep,
    StepFeedback,
    StepPromptVersion,
)
from src.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


T0 = datetime(2026, 6, 1)
T1 = datetime(2026, 6, 15)
T2 = datetime(2026, 7, 3)


# ---------------------------------------------------------------------------
# get_step_versions
# ---------------------------------------------------------------------------

async def _seed_step_run(sf, run_id: str, step_name: str, prompt_hash: str, executed_at: datetime, feedback: str | None = None):
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name="p", source="test",
            normalised_context="{}", raw_payload="{}", stage="production",
        ))
        step = PipelineStep(
            run_id=run_id, step_name=step_name, step_index=0, executor="gateway",
            prompt="p", status="completed", prompt_hash=prompt_hash,
            effective_confidence=0.9, executed_at=executed_at,
        )
        session.add(step)
        await session.flush()
        if feedback is not None:
            session.add(StepFeedback(
                step_id=step.id, run_id=run_id, pipeline_name="p", step_name=step_name, outcome=feedback,
            ))
        await session.commit()


async def _seed_registry(sf, prompt_hash: str, step_name: str, template: str, first_seen: datetime, last_seen: datetime):
    async with sf() as session:
        session.add(StepPromptVersion(
            prompt_hash=prompt_hash, step_name=step_name, template=template,
            first_seen_at=first_seen, last_seen_at=last_seen,
        ))
        await session.commit()


async def test_get_step_versions_empty_for_unknown_step(db):
    versions = await get_step_versions(get_session_factory(), "does-not-exist")
    assert versions == []


async def test_get_step_versions_newest_first(db):
    sf = get_session_factory()
    await _seed_registry(sf, "hash-old", "triage", "Old template.", T0, T0)
    await _seed_registry(sf, "hash-new", "triage", "New template.", T1, T1)

    versions = await get_step_versions(sf, "triage")

    assert [v["prompt_hash"] for v in versions] == ["hash-new", "hash-old"]


async def test_get_step_versions_diff_from_previous_populated_except_oldest(db):
    sf = get_session_factory()
    await _seed_registry(sf, "hash-old", "triage", "line one\nline two", T0, T0)
    await _seed_registry(sf, "hash-new", "triage", "line one\nline THREE", T1, T1)

    versions = await get_step_versions(sf, "triage")

    newest, oldest = versions
    assert oldest["diff_from_previous"] is None
    assert newest["diff_from_previous"] is not None
    assert "hash-old" in newest["diff_from_previous"]
    assert "hash-new" in newest["diff_from_previous"]
    assert "line THREE" in newest["diff_from_previous"]


async def test_get_step_versions_runs_total_and_labelled_n(db):
    sf = get_session_factory()
    await _seed_registry(sf, "hash-a", "triage", "template a", T0, T0)
    await _seed_step_run(sf, "r1", "triage", "hash-a", T0, feedback="correct")
    await _seed_step_run(sf, "r2", "triage", "hash-a", T0, feedback=None)  # unlabelled

    versions = await get_step_versions(sf, "triage")

    assert versions[0]["runs_total"] == 2
    assert versions[0]["labelled_n"] == 1


async def test_get_step_versions_fan_out_branches_roll_up_to_group_name(db):
    """Registry rows and pipeline_steps rows for fan-out/parallel branches are
    written under the collapsed group name (runner.py's _db_save_branch), and
    runs_total must count branch rows ("group/0", "group/1", ...) too."""
    sf = get_session_factory()
    await _seed_registry(sf, "hash-a", "per-service", "investigate {{item}}", T0, T0)
    await _seed_step_run(sf, "r1", "per-service/0", "hash-a", T0)
    await _seed_step_run(sf, "r2", "per-service/1", "hash-a", T0)

    versions = await get_step_versions(sf, "per-service")

    assert versions[0]["runs_total"] == 2


async def test_get_step_versions_calibration_scoped_to_this_hash(db):
    sf = get_session_factory()
    await _seed_registry(sf, "hash-old", "triage", "old", T0, T0)
    await _seed_registry(sf, "hash-new", "triage", "new", T1, T1)
    await _seed_step_run(sf, "r1", "triage", "hash-old", T0, feedback="correct")
    await _seed_step_run(sf, "r2", "triage", "hash-new", T1, feedback="incorrect")

    versions = await get_step_versions(sf, "triage")

    newest, oldest = versions
    assert oldest["prompt_hash"] == "hash-old"
    assert oldest["calibration"][0]["total_n"] == 1
    assert newest["calibration"][0]["total_n"] == 1
    assert oldest["calibration"] != newest["calibration"]


async def test_endpoint_get_step_versions_200(db, monkeypatch, client):
    sf = get_session_factory()
    await _seed_registry(sf, "hash-a", "triage", "template a", T0, T0)
    monkeypatch.setattr(main, "_step_library", {"triage": {"name": "triage"}})

    resp = await client.get("/steps/triage/versions")

    assert resp.status_code == 200
    body = resp.json()
    assert body["step_name"] == "triage"
    assert body["versions"][0]["prompt_hash"] == "hash-a"


async def test_endpoint_get_step_versions_404_unknown_step(db, monkeypatch, client):
    monkeypatch.setattr(main, "_step_library", {})

    resp = await client.get("/steps/does-not-exist/versions")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# get_agent_versions
# ---------------------------------------------------------------------------

async def _seed_agent_snapshot(sf, agent_version, agent, soul_md, agent_yaml, note, first_seen, last_seen):
    async with sf() as session:
        session.add(AgentVersionSnapshot(
            agent_version=agent_version, agent=agent, soul_md=soul_md, agent_yaml=agent_yaml,
            note=note, first_seen_at=first_seen, last_seen_at=last_seen,
        ))
        await session.commit()


async def test_get_agent_versions_empty_for_unknown_agent(db):
    versions = await get_agent_versions(get_session_factory(), "does-not-exist")
    assert versions == []


async def test_get_agent_versions_newest_first(db):
    sf = get_session_factory()
    await _seed_agent_snapshot(sf, "v-old", "gateway:sre-triage", "old soul", "yaml", None, T0, T0)
    await _seed_agent_snapshot(sf, "v-new", "gateway:sre-triage", "new soul", "yaml", None, T1, T1)

    versions = await get_agent_versions(sf, "sre-triage")

    assert [v["agent_version"] for v in versions] == ["v-new", "v-old"]


async def test_get_agent_versions_diff_populated_when_both_have_text(db):
    sf = get_session_factory()
    await _seed_agent_snapshot(sf, "v-old", "gateway:sre-triage", "You are agent A.", "yaml", None, T0, T0)
    await _seed_agent_snapshot(sf, "v-new", "gateway:sre-triage", "You are agent B.", "yaml", None, T1, T1)

    versions = await get_agent_versions(sf, "sre-triage")

    newest, oldest = versions
    assert oldest["diff_from_previous"] is None
    assert "agent A" in newest["diff_from_previous"]
    assert "agent B" in newest["diff_from_previous"]


async def test_get_agent_versions_diff_null_when_either_side_has_no_soul_md(db):
    """The 'gateway unreachable'/'changed before snapshot' case from §4f — no
    text to diff, rendered as an honest gap, not a crash."""
    sf = get_session_factory()
    await _seed_agent_snapshot(
        sf, "v-old", "gateway:sre-triage", None, None,
        "gateway unreachable at snapshot time", T0, T0,
    )
    await _seed_agent_snapshot(sf, "v-new", "gateway:sre-triage", "You are agent B.", "yaml", None, T1, T1)

    versions = await get_agent_versions(sf, "sre-triage")

    newest, oldest = versions
    assert oldest["soul_md"] is None
    assert oldest["note"] == "gateway unreachable at snapshot time"
    assert newest["diff_from_previous"] is None  # predecessor had no text to diff against


async def test_get_agent_versions_used_by_steps(db):
    sf = get_session_factory()
    await _seed_agent_snapshot(sf, "v-old", "gateway:sre-triage", "old", "yaml", None, T0, T0)
    await _seed_agent_snapshot(sf, "v-new", "gateway:sre-triage", "new", "yaml", None, T1, T1)
    async with sf() as session:
        session.add(PipelineRun(id="r1", pipeline_name="p", source="test", normalised_context="{}", raw_payload="{}"))
        session.add(PipelineRun(id="r2", pipeline_name="p", source="test", normalised_context="{}", raw_payload="{}"))
        session.add(PipelineStep(
            run_id="r1", step_name="investigate", step_index=0, executor="gateway",
            agent="gateway:sre-triage", agent_version="v-old", prompt="p", status="completed",
        ))
        session.add(PipelineStep(
            run_id="r2", step_name="summarize", step_index=0, executor="gateway",
            agent="gateway:sre-triage", agent_version="v-new", prompt="p", status="completed",
        ))
        await session.commit()

    versions = await get_agent_versions(sf, "sre-triage")

    by_version = {v["agent_version"]: v for v in versions}
    assert by_version["v-old"]["used_by_steps"] == ["investigate"]
    assert by_version["v-new"]["used_by_steps"] == ["summarize"]


async def test_endpoint_get_agent_versions_200(db, client):
    sf = get_session_factory()
    await _seed_agent_snapshot(sf, "v-a", "gateway:sre-triage", "soul", "yaml", None, T0, T0)

    resp = await client.get("/agents/sre-triage/versions")

    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "gateway:sre-triage"
    assert body["versions"][0]["agent_version"] == "v-a"


async def test_endpoint_get_agent_versions_404_no_snapshots(db, client):
    resp = await client.get("/agents/does-not-exist/versions")

    assert resp.status_code == 404
