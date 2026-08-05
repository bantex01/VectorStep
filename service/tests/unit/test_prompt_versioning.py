"""Tests for src/pipeline/versioning.py (SPEC-prompt-versioning.md) — content
hashing for step prompt templates, and its persistence onto PipelineStep rows
via _db_save_step / _db_save_branch."""
from unittest.mock import patch

import httpx
from sqlalchemy import select

from src.db.database import get_session_factory
from src.db.models import AgentVersionSnapshot, PipelineRun, PipelineStep, StepPromptVersion
from src.models.llm import LLMOutput
from src.models.pipeline import ParallelStepConfig, StepConfig
from src.pipeline.runner import PipelineRunner, StepResult
from src.pipeline.versioning import (
    normalise_template,
    prompt_hash,
    record_agent_version,
    record_prompt_version,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

class TestNormaliseTemplate:
    def test_strips_trailing_whitespace_per_line(self):
        assert normalise_template("line one   \nline two\t\n") == "line one\nline two"

    def test_strips_leading_and_trailing_blank_lines(self):
        assert normalise_template("\n\nbody\n\n\n") == "body"

    def test_does_not_collapse_internal_whitespace(self):
        assert normalise_template("a    b") == "a    b"

    def test_does_not_touch_internal_blank_lines_or_indentation(self):
        text = "line one\n\n    indented line\n"
        assert normalise_template(text) == "line one\n\n    indented line"


class TestPromptHash:
    def test_whitespace_only_change_same_hash(self):
        h1 = prompt_hash("You are triaging an alert.\nBe concise.")
        h2 = prompt_hash("You are triaging an alert.   \nBe concise.\n\n")
        assert h1 == h2

    def test_trailing_newline_only_change_same_hash(self):
        h1 = prompt_hash("Investigate the alert.")
        h2 = prompt_hash("Investigate the alert.\n")
        assert h1 == h2

    def test_added_blank_line_mid_template_changes_hash(self):
        h1 = prompt_hash("line one\nline two")
        h2 = prompt_hash("line one\n\nline two")
        assert h1 != h2

    def test_reindent_changes_hash(self):
        h1 = prompt_hash("- item one\n- item two")
        h2 = prompt_hash("  - item one\n  - item two")
        assert h1 != h2

    def test_content_change_changes_hash(self):
        h1 = prompt_hash("Investigate this alert.")
        h2 = prompt_hash("Investigate this incident.")
        assert h1 != h2

    def test_none_template_returns_none(self):
        assert prompt_hash(None) is None

    def test_empty_string_returns_none(self):
        assert prompt_hash("") is None

    def test_whitespace_only_template_returns_none(self):
        assert prompt_hash("   \n\n   \n") is None

    def test_two_empty_templates_both_none_not_a_shared_hash(self):
        # None is a sentinel, not a hash — two non-LLM steps must not silently
        # collapse into a shared "empty template" bucket. See §2 and §4a note.
        h1 = prompt_hash("")
        h2 = prompt_hash(None)
        assert h1 is None and h2 is None

    def test_reverting_to_previous_template_reproduces_same_hash(self):
        original = "You are triaging an alert.\nBe concise."
        edited = "You are triaging an alert.\nBe thorough."
        reverted = "You are triaging an alert.\nBe concise."
        assert prompt_hash(original) == prompt_hash(reverted)
        assert prompt_hash(original) != prompt_hash(edited)


# ---------------------------------------------------------------------------
# Persistence — _db_save_step / _db_save_branch
# ---------------------------------------------------------------------------

async def _seed_run(sf, run_id: str):
    async with sf() as session:
        session.add(PipelineRun(
            id=run_id, pipeline_name="p", source="test",
            normalised_context="{}", raw_payload="{}",
        ))
        await session.commit()


async def _get_step(sf, run_id: str, step_name: str) -> PipelineStep:
    async with sf() as session:
        rows = (await session.execute(
            select(PipelineStep).where(
                PipelineStep.run_id == run_id, PipelineStep.step_name == step_name,
            )
        )).scalars().all()
        assert len(rows) == 1, f"expected exactly one row for {step_name}, got {len(rows)}"
        return rows[0]


async def test_db_save_step_writes_prompt_hash_and_agent_version(db):
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf)
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="Investigate this alert.\nBe concise.",
    )
    output = LLMOutput(
        confidence=0.9, summary="ok", next_step_context="", raw_response={},
        model="claude-sonnet-5", provider="anthropic", agent_version="91f02ab3c7de",
    )
    result = StepResult(
        step_name="investigate", step_index=0, status="completed",
        output=output, verifier_output=None, effective_confidence=0.9, duration_ms=10,
    )

    await runner._db_save_step("r1", step, result)

    saved = await _get_step(sf, "r1", "investigate")
    assert saved.prompt_hash == prompt_hash(step.prompt_template)
    assert saved.agent_version == "91f02ab3c7de"


async def test_db_save_step_no_output_leaves_agent_version_null(db):
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf)
    step = StepConfig(
        name="failed-step", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="Investigate this alert.",
    )
    result = StepResult(
        step_name="failed-step", step_index=0, status="failed",
        output=None, verifier_output=None, effective_confidence=None, duration_ms=10,
    )

    await runner._db_save_step("r1", step, result)

    saved = await _get_step(sf, "r1", "failed-step")
    # prompt_hash comes from step config, not the (missing) output — still populated.
    assert saved.prompt_hash == prompt_hash(step.prompt_template)
    assert saved.agent_version is None


async def test_db_save_step_non_llm_step_has_null_prompt_hash(db):
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf)
    step = StepConfig(name="notify", executor="webhook", prompt_template="")
    result = StepResult(
        step_name="notify", step_index=0, status="completed",
        output=None, verifier_output=None, effective_confidence=None, duration_ms=5,
    )

    await runner._db_save_step("r1", step, result)

    saved = await _get_step(sf, "r1", "notify")
    assert saved.prompt_hash is None


async def test_db_save_branch_hashes_branchs_own_template_not_groups(db):
    """SPEC-prompt-versioning.md §4d: a fan-out/parallel branch carries its own
    prompt_template distinct from the group's — hashing the group's executor_config
    dump (or nothing at all) would wrongly split a bucket that should be whole."""
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf)
    branch = ParallelStepConfig(
        name="branch-a", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="Branch-specific instructions.",
    )
    output = LLMOutput(
        confidence=0.9, summary="ok", next_step_context="", raw_response={},
        model="claude-sonnet-5", provider="anthropic", agent_version="91f02ab3c7de",
    )

    await runner._db_save_branch("r1", "triage", branch, 0, 0, output)

    saved = await _get_step(sf, "r1", "triage/branch-a")
    assert saved.prompt_hash == prompt_hash("Branch-specific instructions.")

    # Registry step_name uses the collapsed group name, matching calibration.py's
    # own step_name.split("/", 1)[0] collapse (SPEC-prompt-versioning.md §5a).
    h = prompt_hash("Branch-specific instructions.")
    async with sf() as session:
        registry_row = await session.get(StepPromptVersion, h)
    assert registry_row.step_name == "triage"
    assert saved.agent_version == "91f02ab3c7de"


async def test_db_save_branch_failed_output_still_hashes_prompt(db):
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf)
    branch = ParallelStepConfig(
        name="branch-b", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="Branch B instructions.",
    )
    failed_output = LLMOutput(
        confidence=0.0, summary="", next_step_context="", raw_response={}, failed=True,
    )

    await runner._db_save_branch("r1", "triage", branch, 0, 1, failed_output)

    saved = await _get_step(sf, "r1", "triage/branch-b")
    assert saved.prompt_hash == prompt_hash("Branch B instructions.")


async def test_db_save_step_writes_prompt_registry_row_same_transaction(db):
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf)
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="You are triaging an alert.",
    )
    output = LLMOutput(confidence=0.9, summary="ok", next_step_context="", raw_response={})
    result = StepResult(
        step_name="investigate", step_index=0, status="completed",
        output=output, verifier_output=None, effective_confidence=0.9, duration_ms=10,
    )

    await runner._db_save_step("r1", step, result)

    h = prompt_hash(step.prompt_template)
    async with sf() as session:
        registry_row = await session.get(StepPromptVersion, h)
    assert registry_row is not None
    assert registry_row.step_name == "investigate"
    assert registry_row.template == step.prompt_template


async def test_db_save_step_non_llm_step_writes_no_registry_row(db):
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf)
    step = StepConfig(name="notify", executor="webhook", prompt_template="")
    result = StepResult(
        step_name="notify", step_index=0, status="completed",
        output=None, verifier_output=None, effective_confidence=None, duration_ms=5,
    )

    await runner._db_save_step("r1", step, result)

    async with sf() as session:
        rows = (await session.execute(select(StepPromptVersion))).scalars().all()
    assert rows == []


# ---------------------------------------------------------------------------
# record_prompt_version — write-on-miss upsert
# ---------------------------------------------------------------------------

async def test_record_prompt_version_no_op_for_none_hash(db):
    sf = get_session_factory()
    async with sf() as session:
        await record_prompt_version(session, hash_=None, step_name="s", template="t")
        await session.commit()

    async with sf() as session:
        rows = (await session.execute(select(StepPromptVersion))).scalars().all()
    assert rows == []


async def test_record_prompt_version_second_sighting_refreshes_last_seen_not_duplicate(db):
    sf = get_session_factory()
    h = prompt_hash("template text")

    async with sf() as session:
        await record_prompt_version(session, hash_=h, step_name="investigate", template="template text")
        await session.commit()
    async with sf() as session:
        first = await session.get(StepPromptVersion, h)
        first_seen, last_seen = first.first_seen_at, first.last_seen_at

    async with sf() as session:
        await record_prompt_version(session, hash_=h, step_name="investigate", template="template text")
        await session.commit()

    async with sf() as session:
        rows = (await session.execute(select(StepPromptVersion))).scalars().all()
        second = await session.get(StepPromptVersion, h)

    assert len(rows) == 1  # no duplicate row
    assert second.first_seen_at == first_seen  # unchanged
    assert second.last_seen_at >= last_seen


# ---------------------------------------------------------------------------
# record_agent_version — best-effort snapshot fetch from the Gateway
# ---------------------------------------------------------------------------

def _fake_response(status_code=200, json_body=None):
    async def fake_request(self, method, url, **kwargs):
        return httpx.Response(status_code, json=json_body or {}, request=httpx.Request(method, url))
    return fake_request


async def test_record_agent_version_no_op_for_none_version(db):
    sf = get_session_factory()
    async with sf() as session:
        await record_agent_version(session, "http://gw", agent_version=None, agent="gateway:sre-triage")
        await session.commit()

    async with sf() as session:
        rows = (await session.execute(select(AgentVersionSnapshot))).scalars().all()
    assert rows == []


async def test_record_agent_version_stores_text_when_gateway_confirms_current(db):
    sf = get_session_factory()
    fake = _fake_response(json_body={
        "name": "sre-triage", "version": "91f02ab3c7de",
        "soul_md": "You are an SRE.", "agent_yaml": "name: sre-triage\n",
    })

    with patch.object(httpx.AsyncClient, "request", new=fake):
        async with sf() as session:
            await record_agent_version(
                session, "http://gw", agent_version="91f02ab3c7de", agent="gateway:sre-triage",
            )
            await session.commit()

    async with sf() as session:
        row = await session.get(AgentVersionSnapshot, "91f02ab3c7de")
    assert row.agent == "gateway:sre-triage"
    assert row.soul_md == "You are an SRE."
    assert row.agent_yaml == "name: sre-triage\n"
    assert row.note is None


async def test_record_agent_version_strips_gateway_prefix_when_calling_rest(db):
    sf = get_session_factory()
    called = {}

    async def fake_request(self, method, url, **kwargs):
        called["url"] = url
        return httpx.Response(200, json={"version": "91f02ab3c7de", "soul_md": "s", "agent_yaml": "y"},
                               request=httpx.Request(method, url))

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        async with sf() as session:
            await record_agent_version(
                session, "http://gw", agent_version="91f02ab3c7de", agent="gateway:sre-triage",
            )
            await session.commit()

    assert str(called["url"]) == "http://gw/agents/sre-triage"


async def test_record_agent_version_stores_note_when_gateway_reports_different_current(db):
    sf = get_session_factory()
    # Gateway confirms a DIFFERENT version is current than the one we're resolving —
    # the operator changed the agent again before we could snapshot this one.
    fake = _fake_response(json_body={"version": "some-newer-hash", "soul_md": "new soul", "agent_yaml": "y"})

    with patch.object(httpx.AsyncClient, "request", new=fake):
        async with sf() as session:
            await record_agent_version(
                session, "http://gw", agent_version="91f02ab3c7de", agent="gateway:sre-triage",
            )
            await session.commit()

    async with sf() as session:
        row = await session.get(AgentVersionSnapshot, "91f02ab3c7de")
    assert row.soul_md is None
    assert row.agent_yaml is None
    assert row.note == "agent config changed before VectorStep could snapshot this version"


async def test_record_agent_version_stores_note_when_gateway_unreachable(db):
    sf = get_session_factory()

    async def fake_request(self, method, url, **kwargs):
        raise httpx.ConnectError("connection refused", request=httpx.Request(method, url))

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        async with sf() as session:
            await record_agent_version(
                session, "http://gw", agent_version="91f02ab3c7de", agent="gateway:sre-triage",
            )
            await session.commit()

    async with sf() as session:
        row = await session.get(AgentVersionSnapshot, "91f02ab3c7de")
    assert row.soul_md is None
    assert row.note == "gateway unreachable at snapshot time"


async def test_record_agent_version_never_raises_even_on_unreachable_gateway(db):
    """Best-effort bookkeeping must never fail a run."""
    sf = get_session_factory()

    async def fake_request(self, method, url, **kwargs):
        raise httpx.ConnectError("connection refused", request=httpx.Request(method, url))

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        async with sf() as session:
            await record_agent_version(
                session, "http://gw", agent_version="abc123", agent="gateway:x",
            )
            await session.commit()  # would raise if the exception propagated


async def test_record_agent_version_already_known_version_makes_no_http_call(db):
    sf = get_session_factory()
    async with sf() as session:
        session.add(AgentVersionSnapshot(
            agent_version="91f02ab3c7de", agent="gateway:sre-triage",
            soul_md="existing", agent_yaml="existing",
        ))
        await session.commit()

    called = {"count": 0}

    async def fake_request(self, method, url, **kwargs):
        called["count"] += 1
        return httpx.Response(200, json={}, request=httpx.Request(method, url))

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        async with sf() as session:
            await record_agent_version(
                session, "http://gw", agent_version="91f02ab3c7de", agent="gateway:sre-triage",
            )
            await session.commit()

    assert called["count"] == 0


async def test_db_save_step_snapshots_agent_version_when_gateway_configured(db):
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf, gateway_rest_url="http://gw")
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="Investigate.",
    )
    output = LLMOutput(
        confidence=0.9, summary="ok", next_step_context="", raw_response={},
        agent_version="91f02ab3c7de",
    )
    result = StepResult(
        step_name="investigate", step_index=0, status="completed",
        output=output, verifier_output=None, effective_confidence=0.9, duration_ms=10,
    )
    fake = _fake_response(json_body={"version": "91f02ab3c7de", "soul_md": "soul", "agent_yaml": "yaml"})

    with patch.object(httpx.AsyncClient, "request", new=fake):
        await runner._db_save_step("r1", step, result)

    async with sf() as session:
        row = await session.get(AgentVersionSnapshot, "91f02ab3c7de")
    assert row is not None
    assert row.agent == "gateway:sre-investigation"


async def test_db_save_step_no_gateway_rest_url_skips_agent_snapshot(db):
    """No gateway_rest_url configured -> no HTTP call attempted at all, even though
    an agent_version is present (e.g. a non-gateway PipelineRunner in tests)."""
    sf = get_session_factory()
    await _seed_run(sf, "r1")
    runner = PipelineRunner(executors={}, session_factory=sf)  # no gateway_rest_url
    step = StepConfig(
        name="investigate", executor="gateway", executor_config={"agent": "sre-investigation"},
        prompt_template="Investigate.",
    )
    output = LLMOutput(
        confidence=0.9, summary="ok", next_step_context="", raw_response={},
        agent_version="91f02ab3c7de",
    )
    result = StepResult(
        step_name="investigate", step_index=0, status="completed",
        output=output, verifier_output=None, effective_confidence=0.9, duration_ms=10,
    )

    called = {"count": 0}

    async def fake_request(self, method, url, **kwargs):
        called["count"] += 1
        return httpx.Response(200, json={}, request=httpx.Request(method, url))

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        await runner._db_save_step("r1", step, result)

    assert called["count"] == 0
