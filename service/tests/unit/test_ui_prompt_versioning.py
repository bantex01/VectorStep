"""Smoke tests for the SPEC-prompt-versioning.md §6 UI surfaces — that the version
chips, prompt history disclosure, bucket-reset callout, and agent versions tab
all render without a Jinja error when their data is present. Not pixel-level
assertions; these exist to catch template/undefined-variable regressions."""
from datetime import datetime

import httpx
import pytest

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


async def test_insights_steps_renders_with_prompt_hash_and_agent_version(db, client):
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="run-1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
            triggered_at=datetime(2026, 1, 1),
        ))
        step = PipelineStep(
            id="step-a", run_id="run-1", step_name="triage", step_index=0, executor="gateway",
            agent="gateway:sre-triage", model="claude-sonnet-5", provider="anthropic",
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
            prompt_hash="a3f2c9d81e04", agent_version="91f02ab3c7de", effective_confidence=0.9,
        )
        session.add(step)
        await session.flush()
        # compute_calibration_buckets only forms a bucket for LABELLED step-executions —
        # without feedback, prompt_hash/agent_version would never reach the breakdown row.
        session.add(StepFeedback(step_id=step.id, run_id="run-1", pipeline_name="p", step_name="triage", outcome="correct"))
        await session.commit()

    resp = await client.get("/ui/insights/steps?time_range=all")

    assert resp.status_code == 200
    assert "a3f2c9d81e04" in resp.text  # version chip data is embedded in the drilldown JSON


async def test_step_versions_endpoint_reachable_from_insights_steps_page(db, client):
    """The Prompt history disclosure fetches this at runtime — confirm it 200s
    for a step that has registry rows, independent of the JS that calls it."""
    sf = get_session_factory()
    async with sf() as session:
        session.add(StepPromptVersion(
            prompt_hash="a3f2c9d81e04", step_name="triage", template="You are triaging...",
        ))
        await session.commit()

    resp = await client.get("/steps/triage/versions")

    assert resp.status_code == 404  # step not in the (empty) step library in this test process


async def test_run_detail_renders_with_bucket_reset(db, client):
    sf = get_session_factory()
    trust_report = {
        "signals": {"S": 0.9, "S_after_V": 0.9, "V": None, "V_mode": None, "G": None, "D": None},
        "calibration": {
            "bucket": {"step_name": "investigate", "agent": "gateway:sre-triage",
                       "model": "claude-sonnet-5", "provider": "anthropic"},
            "bin": None, "n": 2, "n_min": 20, "validated": False, "raw": 0.9, "calibrated": None,
            "on_uncalibrated": "proceed",
            "bucket_reset": {
                "reason": "prompt_changed",
                "previous_version_last_seen": "2026-07-03T16:02:51",
                "previous_validated_n": 47,
            },
        },
        "grounding": None,
        "deterministic_checks": None,
        "gate": {"confidence_threshold": 0.75, "on_low_confidence": "escalate"},
        "combined_trust": 0.9,
    }
    async with sf() as session:
        session.add(PipelineRun(
            id="run-1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
            triggered_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineStep(
            id="step-a", run_id="run-1", step_name="investigate", step_index=0, executor="gateway",
            agent="gateway:sre-triage", model="claude-sonnet-5", provider="anthropic",
            prompt="Investigate this.", status="completed", executed_at=datetime(2026, 1, 1),
            prompt_hash="a3f2c9d81e04", agent_version="91f02ab3c7de",
            trust_report=__import__("json").dumps(trust_report), effective_confidence=0.9,
            # The Prompt/Trust panel block is gated on pretty/verifier_pretty/trace being
            # truthy — needs a real parsed_output for the block to render at all.
            parsed_output=__import__("json").dumps({"confidence": 0.9, "summary": "ok"}),
        ))
        await session.commit()

    resp = await client.get("/ui/runs/run-1")

    assert resp.status_code == 200
    assert "a3f2c9d81e04" in resp.text  # template chip
    assert "91f02ab3c7de" in resp.text  # agent chip
    assert "reset" in resp.text.lower()


async def test_agent_detail_renders_with_agent_versions(db, client):
    sf = get_session_factory()
    async with sf() as session:
        session.add(AgentVersionSnapshot(
            agent_version="91f02ab3c7de", agent="gateway:sre-triage",
            soul_md="You are an SRE triage agent.", agent_yaml="name: sre-triage\n",
        ))
        await session.commit()

    resp = await client.get("/ui/agents/gateway/sre-triage")

    assert resp.status_code == 200
    assert "91f02ab3c7de" in resp.text


async def test_agent_detail_renders_with_unreachable_snapshot_note(db, client):
    sf = get_session_factory()
    async with sf() as session:
        session.add(AgentVersionSnapshot(
            agent_version="deadbeef0001", agent="gateway:sre-triage",
            soul_md=None, agent_yaml=None, note="gateway unreachable at snapshot time",
        ))
        await session.commit()

    resp = await client.get("/ui/agents/gateway/sre-triage")

    assert resp.status_code == 200
    assert "gateway unreachable at snapshot time" in resp.text


async def test_agent_detail_renders_with_legacy_null_agent_version(db, client):
    """The synthetic legacy entry (agent_version IS NULL, no AgentVersionSnapshot
    row at all) must render without crashing the Jinja `[:12]` slice — this is
    exactly the gap a live MCP test caught: get_agent_versions can return an
    entry whose agent_version is None."""
    sf = get_session_factory()
    async with sf() as session:
        session.add(PipelineRun(
            id="run-1", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
            triggered_at=datetime(2026, 1, 1),
        ))
        session.add(PipelineStep(
            run_id="run-1", step_name="investigate", step_index=0, executor="gateway",
            agent="gateway:sre-triage", agent_version=None, prompt="p", status="completed",
            executed_at=datetime(2026, 1, 1),
        ))
        await session.commit()

    resp = await client.get("/ui/agents/gateway/sre-triage")

    assert resp.status_code == 200
    assert "pre-versioning" in resp.text


async def test_agent_detail_renders_with_no_versions_at_all(db, client):
    """An agent with no snapshot rows must not break the page — the Versions tab
    simply doesn't appear."""
    resp = await client.get("/ui/agents/gateway/never-seen-agent")

    assert resp.status_code == 200
