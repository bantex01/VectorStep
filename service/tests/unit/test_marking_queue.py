"""Tests for the marking queue (GET /ui/marking-queue) — the cross-pipeline review
queue of steps with no HUMAN accuracy feedback (StepFeedback), regardless of whether
an automatic label already exists from a failed deterministic check or an inherited
run-level rating. Mirrors the semantics of readiness.accuracy.min_human_marked."""
import re

import httpx
import pytest
from fastapi import FastAPI

from src.db.database import get_session_factory
from src.db.models import PipelineRun, PipelineStep, RunFeedback, StepFeedback
from src.ui import router as ui_router

app = FastAPI()
app.include_router(ui_router)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_run(sf, run_id: str, pipeline_name: str, *, stage="testing", team=None, status="completed") -> None:
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name=pipeline_name, source="test", status=status,
            normalised_context="{}", raw_payload="{}", stage=stage, team=team,
        ))
        await session.commit()


async def _seed_step(
    sf, run_id: str, pipeline_name: str, step_name: str, *,
    status="completed", deterministic_passed=None, step_feedback=None, index=0,
) -> str:
    async with sf() as session:
        step = PipelineStep(
            run_id=run_id, step_name=step_name, step_index=index, executor="gateway",
            prompt="p", status=status, deterministic_passed=deterministic_passed,
        )
        session.add(step)
        await session.flush()
        step_id = step.id
        if step_feedback is not None:
            session.add(StepFeedback(
                step_id=step_id, run_id=run_id, pipeline_name=pipeline_name,
                step_name=step_name, outcome=step_feedback,
            ))
        await session.commit()
    return step_id


async def _seed_run_feedback(sf, run_id: str, pipeline_name: str, outcome: str = "correct") -> None:
    async with sf() as session:
        session.add(RunFeedback(run_id=run_id, pipeline_name=pipeline_name, outcome=outcome))
        await session.commit()


def _stat_value(text: str, label: str) -> str:
    """Pull the number out of a stat card by its label, tolerant of the
    multi-line Jinja whitespace around the value."""
    m = re.search(re.escape(label) + r"</p>\s*<p[^>]*>\s*([^\s<]+)\s*</p>", text)
    assert m, f"stat card {label!r} not found"
    return m.group(1)


def _provenance_tag_count(text: str, tag: str) -> int:
    """Occurrences of a provenance tag OUTSIDE the fixed legend paragraph, which
    always mentions both tag words once regardless of what's in the queue."""
    return text.count(f'>{tag}</span>') - 1  # legend contributes exactly one


async def test_step_with_human_feedback_is_excluded(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r1", "p")
    await _seed_step(sf, "r1", "p", "s", step_feedback="correct")

    resp = await client.get("/ui/marking-queue?stage=testing")
    assert resp.status_code == 200
    assert "Nothing unmarked" in resp.text


async def test_step_with_no_feedback_appears_with_no_provenance_tag(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r1", "p")
    await _seed_step(sf, "r1", "p", "s")

    resp = await client.get("/ui/marking-queue?stage=testing")
    assert resp.status_code == 200
    assert ">p<" in resp.text
    assert ">s<" in resp.text
    assert _provenance_tag_count(resp.text, "failed check") == 0
    assert _provenance_tag_count(resp.text, "run feedback") == 0


async def test_failed_deterministic_check_tagged_as_provenance(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r1", "p")
    await _seed_step(sf, "r1", "p", "s", deterministic_passed=False)

    resp = await client.get("/ui/marking-queue?stage=testing")
    assert resp.status_code == 200
    assert _provenance_tag_count(resp.text, "failed check") == 1
    assert _provenance_tag_count(resp.text, "run feedback") == 0


async def test_run_level_feedback_tagged_as_provenance_when_no_step_feedback(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r1", "p")
    await _seed_step(sf, "r1", "p", "s")
    await _seed_run_feedback(sf, "r1", "p")

    resp = await client.get("/ui/marking-queue?stage=testing")
    assert resp.status_code == 200
    assert _provenance_tag_count(resp.text, "run feedback") == 1
    assert _provenance_tag_count(resp.text, "failed check") == 0


async def test_failed_check_takes_precedence_over_run_feedback_tag(db, client):
    """resolve_label's precedence (step > deterministic > run) means a step with
    both a failed check AND a run-level rating is labeled via the failed check —
    the queue's provenance tag should match that, not double-count or prefer
    the weaker signal."""
    sf = get_session_factory()
    await _seed_run(sf, "r1", "p")
    await _seed_step(sf, "r1", "p", "s", deterministic_passed=False)
    await _seed_run_feedback(sf, "r1", "p")

    resp = await client.get("/ui/marking-queue?stage=testing")
    assert resp.status_code == 200
    assert _provenance_tag_count(resp.text, "failed check") == 1
    assert _provenance_tag_count(resp.text, "run feedback") == 0


async def test_stage_defaults_to_testing(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r-test", "p", stage="testing")
    await _seed_step(sf, "r-test", "p", "s")
    await _seed_run(sf, "r-prod", "p", stage="production")
    await _seed_step(sf, "r-prod", "p", "s2")

    resp = await client.get("/ui/marking-queue")
    assert resp.status_code == 200
    assert ">s<" in resp.text
    assert ">s2<" not in resp.text


async def test_stage_filter_can_select_production(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r-test", "p", stage="testing")
    await _seed_step(sf, "r-test", "p", "s")
    await _seed_run(sf, "r-prod", "p", stage="production")
    await _seed_step(sf, "r-prod", "p", "s2")

    resp = await client.get("/ui/marking-queue?stage=production")
    assert resp.status_code == 200
    assert ">s2<" in resp.text
    assert ">s<" not in resp.text


async def test_empty_stage_filter_shows_all_stages(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r-test", "p", stage="testing")
    await _seed_step(sf, "r-test", "p", "s")
    await _seed_run(sf, "r-prod", "p", stage="production")
    await _seed_step(sf, "r-prod", "p", "s2")

    resp = await client.get("/ui/marking-queue?stage=")
    assert resp.status_code == 200
    assert ">s<" in resp.text
    assert ">s2<" in resp.text


async def test_pipeline_and_team_filters_are_additive(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r1", "pipeline-a", team="alpha")
    await _seed_step(sf, "r1", "pipeline-a", "s")
    await _seed_run(sf, "r2", "pipeline-a", team="beta")
    await _seed_step(sf, "r2", "pipeline-a", "s2")
    await _seed_run(sf, "r3", "pipeline-b", team="alpha")
    await _seed_step(sf, "r3", "pipeline-b", "s3")

    # pipeline filter alone
    resp = await client.get("/ui/marking-queue?stage=testing&pipeline=pipeline-a")
    assert ">s<" in resp.text and ">s2<" in resp.text and ">s3<" not in resp.text

    # team filter alone
    resp = await client.get("/ui/marking-queue?stage=testing&team=alpha")
    assert ">s<" in resp.text and ">s3<" in resp.text and ">s2<" not in resp.text

    # both together (additive AND)
    resp = await client.get("/ui/marking-queue?stage=testing&pipeline=pipeline-a&team=alpha")
    assert ">s<" in resp.text and ">s2<" not in resp.text and ">s3<" not in resp.text


async def test_fan_out_branches_collapse_to_group_name(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r1", "p")
    await _seed_step(sf, "r1", "p", "triage/0")
    await _seed_run(sf, "r2", "p")
    await _seed_step(sf, "r2", "p", "triage/1")

    resp = await client.get("/ui/marking-queue?stage=testing")
    assert resp.status_code == 200
    assert ">triage<" in resp.text
    assert "2 unmarked" in resp.text


async def test_stat_cards_and_coverage_percentage(db, client):
    sf = get_session_factory()
    # One marked, one unmarked, out of two eligible terminal steps.
    await _seed_run(sf, "r1", "p")
    await _seed_step(sf, "r1", "p", "marked-step", step_feedback="correct")
    await _seed_run(sf, "r2", "p")
    await _seed_step(sf, "r2", "p", "unmarked-step")

    resp = await client.get("/ui/marking-queue?stage=testing")
    assert resp.status_code == 200
    text = resp.text
    # 1 pipeline, 1 run, 1 step unmarked of 2 eligible -> 50% coverage.
    assert _stat_value(text, "Pipelines") == "1"
    assert _stat_value(text, "Runs") == "1"
    assert _stat_value(text, "Steps unmarked") == "1"
    assert "of 2 eligible" in text
    assert _stat_value(text, "Coverage") == "50%"


async def test_empty_state_when_everything_marked(db, client):
    sf = get_session_factory()
    await _seed_run(sf, "r1", "p")
    await _seed_step(sf, "r1", "p", "s", step_feedback="correct")

    resp = await client.get("/ui/marking-queue?stage=testing")
    assert resp.status_code == 200
    assert "Nothing unmarked" in resp.text


async def test_no_write_endpoint_referenced(db, client):
    """This page links out to run detail for marking — it must not itself submit
    feedback (no POST/PUT/DELETE calls of any kind, it's a plain GET/filter page)."""
    resp = await client.get("/ui/marking-queue")
    assert resp.status_code == 200
    assert "<form" in resp.text  # the filter form
    assert 'method="get"' in resp.text
    assert "fetch(" not in resp.text
