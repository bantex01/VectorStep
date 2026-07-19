"""Tests for deterministic checks (D) + grounding as an opt-in gate (SPEC-hard-gates.md):
config parsing, the three check-type evaluators, request_decision, the HumanExecutor
regression guarantee, _run_deterministic_checks, the gate integration (the critical
tests), persistence, migration, and the pork_step_deterministic_check_total metric."""
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import select

from src.db.database import create_tables, get_session_factory, init_db
from src.db.models import PipelineStep
from src.executors import human
from src.metrics import MetricsData, PorkCollector, fetch_metrics_data
from src.models.context import NormalisedContext
from src.models.llm import LLMOutput
from src.models.pipeline import (
    GroundingConfig,
    HumanCheckConfig,
    PipelineConfig,
    ShellCheckConfig,
    StepConfig,
    TriggerConfig,
    WebhookCheckConfig,
)
from src.pipeline.runner import PipelineRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubExecutor:
    def __init__(self, output):
        self._output = output

    async def execute(self, step, ctx):
        return self._output


def _make_normalised(**kwargs) -> NormalisedContext:
    defaults = dict(
        source="test", pipeline="test-pipeline", severity="warning",
        summary="test alert", labels={}, metadata={}, raw={},
        received_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    defaults.update(kwargs)
    return NormalisedContext(**defaults)


def _make_output(confidence=0.9, raw_response=None) -> LLMOutput:
    return LLMOutput(
        confidence=confidence, summary="ok", next_step_context="",
        raw_response=raw_response or {},
    )


def _runner(executors=None) -> PipelineRunner:
    return PipelineRunner(executors=executors or {}, session_factory=None)


@pytest.fixture(autouse=True)
def _reset_human_state():
    """Human-check tests share the module-global pending state with test_human_executor.py."""
    human._pending_approvals.clear()
    human._pending_meta.clear()
    human.configure(human_approval={}, legacy_telegram={}, ui_base_url="http://localhost:8000")
    yield
    human._pending_approvals.clear()
    human._pending_meta.clear()


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------

def test_check_config_defaults():
    shell = ShellCheckConfig(name="x", run="echo hi")
    assert shell.expect == "exit_code == 0"
    assert shell.timeout_seconds == 30

    webhook = WebhookCheckConfig(name="y", url="https://example.com")
    assert webhook.method == "POST"
    assert webhook.expect == "response.status_code < 400"
    assert webhook.timeout_seconds == 30

    human_check = HumanCheckConfig(name="z", message="approve?")
    assert human_check.timeout_seconds == 300


def test_step_config_discriminates_check_types_from_raw_dict():
    step = StepConfig(
        name="s", executor="gateway", prompt_template="",
        deterministic_checks=[
            {"type": "shell", "name": "a", "run": "exit 0"},
            {"type": "webhook", "name": "b", "url": "https://x"},
            {"type": "human", "name": "c", "message": "ok?"},
        ],
    )
    assert isinstance(step.deterministic_checks[0], ShellCheckConfig)
    assert isinstance(step.deterministic_checks[1], WebhookCheckConfig)
    assert isinstance(step.deterministic_checks[2], HumanCheckConfig)


def test_grounding_config_enforce_defaults_false():
    assert GroundingConfig().enforce is False
    assert GroundingConfig(enforce=True).enforce is True


# ---------------------------------------------------------------------------
# 2. _eval_shell_check
# ---------------------------------------------------------------------------

async def test_eval_shell_check_default_expect_passes_and_captures_result():
    runner = _runner()
    check = ShellCheckConfig(name="x", run="echo hello")

    passed, detail = await runner._eval_shell_check(check, {})

    assert passed is True
    assert "hello" in detail


async def test_eval_shell_check_custom_expect_evaluated_against_result():
    runner = _runner()
    check = ShellCheckConfig(name="x", run="echo 0.9", expect="result | float > 0.5")

    passed, _ = await runner._eval_shell_check(check, {})

    assert passed is True

    check_low = ShellCheckConfig(name="x", run="echo 0.1", expect="result | float > 0.5")
    passed_low, _ = await runner._eval_shell_check(check_low, {})
    assert passed_low is False


async def test_eval_shell_check_nonzero_exit_fails_regardless_of_expect():
    runner = _runner()
    check = ShellCheckConfig(name="x", run="exit 1", expect="true")

    passed, detail = await runner._eval_shell_check(check, {})

    assert passed is False
    assert "exit code 1" in detail


async def test_eval_shell_check_timeout_kills_process_and_fails():
    runner = _runner()
    check = ShellCheckConfig(name="x", run="sleep 5", timeout_seconds=1)

    passed, detail = await runner._eval_shell_check(check, {})

    assert passed is False
    assert "timed out" in detail


# ---------------------------------------------------------------------------
# 3. _eval_webhook_check
# ---------------------------------------------------------------------------

async def test_eval_webhook_check_200_default_expect_passes():
    runner = _runner()
    check = WebhookCheckConfig(name="x", url="https://example.com/check")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"ok": True})

    async def fake_request(self, method, url, json=None, headers=None):
        return mock_response

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        passed, detail = await runner._eval_webhook_check(check, {})

    assert passed is True
    assert "200" in detail


async def test_eval_webhook_check_custom_expect_against_body():
    runner = _runner()
    check = WebhookCheckConfig(
        name="x", url="https://example.com/check",
        expect="response.body.value > 0.02",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"value": 0.05})

    async def fake_request(self, method, url, json=None, headers=None):
        return mock_response

    with patch.object(httpx.AsyncClient, "request", new=fake_request):
        passed, _ = await runner._eval_webhook_check(check, {})

    assert passed is True


async def test_eval_webhook_check_connection_error_fails_without_propagating_past_the_runner():
    """_eval_webhook_check itself lets the exception through — _run_deterministic_checks
    is the fail-closed boundary (see test_run_deterministic_checks_unexpected_exception_
    recorded_as_failed) — so this exercises the check end-to-end via that boundary."""
    runner = _runner()
    step = StepConfig(
        name="s", executor="gateway", prompt_template="",
        deterministic_checks=[WebhookCheckConfig(name="x", url="https://example.com/check")],
    )

    with patch.object(httpx.AsyncClient, "request", side_effect=httpx.ConnectError("unreachable")):
        results = await runner._run_deterministic_checks(step=step, ctx={}, run_log=[])

    assert results[0]["passed"] is False
    assert "unreachable" in results[0]["detail"]


# ---------------------------------------------------------------------------
# 4. _eval_human_check / request_decision
# ---------------------------------------------------------------------------

async def test_eval_human_check_approved_passes():
    human.configure(
        human_approval={"default": {"channel": "slack", "slack": {"bot_token": "t", "channel_id": "c"}}},
        legacy_telegram={}, ui_base_url="http://x",
    )
    runner = _runner()
    check = HumanCheckConfig(name="approve-check", message="ok?", timeout_seconds=2)

    async def fake_send(self, text, token):
        asyncio.get_running_loop().call_later(0, lambda: human.resolve_approval(token, True))

    with patch.object(human.SlackApprovalChannel, "send", new=fake_send):
        passed, detail = await runner._eval_human_check(check, {})

    assert passed is True
    assert "approved" in detail


async def test_eval_human_check_rejected_fails():
    human.configure(
        human_approval={"default": {"channel": "slack", "slack": {"bot_token": "t", "channel_id": "c"}}},
        legacy_telegram={}, ui_base_url="http://x",
    )
    runner = _runner()
    check = HumanCheckConfig(name="approve-check", message="ok?", timeout_seconds=2)

    async def fake_send(self, text, token):
        asyncio.get_running_loop().call_later(0, lambda: human.resolve_approval(token, False))

    with patch.object(human.SlackApprovalChannel, "send", new=fake_send):
        passed, detail = await runner._eval_human_check(check, {})

    assert passed is False
    assert "rejected" in detail


async def test_eval_human_check_timeout_in_production_fails_without_raising():
    human.configure(
        human_approval={"default": {"channel": "slack", "slack": {"bot_token": "t", "channel_id": "c"}}},
        legacy_telegram={}, ui_base_url="http://x",
    )
    runner = _runner()
    check = HumanCheckConfig(name="approve-check", message="ok?", timeout_seconds=1)

    with patch.object(human.SlackApprovalChannel, "send", new=AsyncMock()):
        passed, detail = await runner._eval_human_check(check, {})

    assert passed is False
    assert "timed out" in detail


async def test_eval_human_check_timeout_in_testing_auto_approves():
    runner = _runner()
    check = HumanCheckConfig(name="approve-check", message="ok?", timeout_seconds=1)

    passed, detail = await runner._eval_human_check(check, {"_testing": True})

    assert passed is True


# ---------------------------------------------------------------------------
# 5. HumanExecutor regression (critical — do not skip)
# ---------------------------------------------------------------------------

def test_human_executor_suite_still_passes_unmodified():
    """Runs the entire existing tests/test_human_executor.py file in-process to prove
    the request_decision() extraction changed zero observable HumanExecutor behaviour.
    (Also run directly via `pytest tests/test_human_executor.py` in CI/manual runs —
    this in-process run is a belt-and-braces guard inside this spec's own test file.)"""
    exit_code = pytest.main(["-q", "tests/test_human_executor.py"])
    assert exit_code == 0


# ---------------------------------------------------------------------------
# 6. _run_deterministic_checks
# ---------------------------------------------------------------------------

async def test_run_deterministic_checks_all_pass():
    runner = _runner()
    step = StepConfig(
        name="s", executor="gateway", prompt_template="",
        deterministic_checks=[
            ShellCheckConfig(name="a", run="exit 0"),
            ShellCheckConfig(name="b", run="echo ok"),
        ],
    )

    results = await runner._run_deterministic_checks(step=step, ctx={}, run_log=[])

    assert all(r["passed"] for r in results)
    assert [r["name"] for r in results] == ["a", "b"]
    assert all(r["type"] == "shell" for r in results)


async def test_run_deterministic_checks_one_failure_keeps_all_results():
    runner = _runner()
    step = StepConfig(
        name="s", executor="gateway", prompt_template="",
        deterministic_checks=[
            ShellCheckConfig(name="a", run="exit 0"),
            ShellCheckConfig(name="b", run="exit 1"),
        ],
    )

    results = await runner._run_deterministic_checks(step=step, ctx={}, run_log=[])

    assert len(results) == 2
    passed_by_name = {r["name"]: r["passed"] for r in results}
    assert passed_by_name == {"a": True, "b": False}


async def test_run_deterministic_checks_unexpected_exception_recorded_as_failed():
    runner = _runner()
    step = StepConfig(
        name="s", executor="gateway", prompt_template="",
        deterministic_checks=[WebhookCheckConfig(name="a", url="https://example.com")],
    )

    with patch.object(httpx.AsyncClient, "request", side_effect=RuntimeError("boom")):
        results = await runner._run_deterministic_checks(step=step, ctx={}, run_log=[])

    assert results[0]["passed"] is False
    assert "boom" in results[0]["detail"]


# ---------------------------------------------------------------------------
# 7. Gate integration (the critical tests — do not skip)
# ---------------------------------------------------------------------------

async def test_grounding_enforced_low_g_high_s_caps_trust_and_escalates():
    primary_output = _make_output(confidence=0.95, raw_response={"trace": [
        {"type": "tool_call", "name": "q", "input": {}},
        {"type": "tool_result", "name": "q", "content": "x", "is_error": False},
    ]})
    grounding_output = LLMOutput(confidence=0.1, summary="unsupported", next_step_context="", raw_response={})

    runner = _runner(executors={
        "gateway": lambda: _StubExecutor(primary_output),
        "grounding_stub": lambda: _StubExecutor(grounding_output),
    })
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        confidence_threshold=0.75, on_low_confidence="escalate",
        grounding=GroundingConfig(executor="grounding_stub", enforce=True),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )

    assert result.trust_report["combined_trust"] == 0.1
    assert result.status == "escalated"


async def test_grounding_shadow_enforce_false_default_no_gate_change():
    """Same setup as the enforced test above, but enforce omitted (False) — the
    step must complete exactly as it would with no grounding at all. Re-proves
    Phase 0's guarantee still holds after this spec's changes land."""
    primary_output = _make_output(confidence=0.95, raw_response={"trace": [
        {"type": "tool_call", "name": "q", "input": {}},
        {"type": "tool_result", "name": "q", "content": "x", "is_error": False},
    ]})
    grounding_output = LLMOutput(confidence=0.1, summary="unsupported", next_step_context="", raw_response={})

    runner = _runner(executors={
        "gateway": lambda: _StubExecutor(primary_output),
        "grounding_stub": lambda: _StubExecutor(grounding_output),
    })
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        confidence_threshold=0.75, on_low_confidence="escalate",
        grounding=GroundingConfig(executor="grounding_stub"),  # enforce defaults False
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )

    assert result.status == "completed"
    assert result.trust_report["combined_trust"] == 0.95
    assert result.trust_report["gate"]["policy"] == "legacy_confidence"


async def test_failed_deterministic_check_forces_escalate_regardless_of_confidence():
    primary_output = _make_output(confidence=0.95)
    runner = _runner(executors={"gateway": lambda: _StubExecutor(primary_output)})
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        confidence_threshold=0.5, on_low_confidence="escalate",
        deterministic_checks=[ShellCheckConfig(name="still_breaching", run="exit 1")],
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )

    assert result.trust_report["combined_trust"] == 0.0
    assert result.status != "completed"
    assert result.status == "escalated"


async def test_all_deterministic_checks_pass_no_behaviour_change():
    primary_output = _make_output(confidence=0.95)
    runner = _runner(executors={"gateway": lambda: _StubExecutor(primary_output)})
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        confidence_threshold=0.5, on_low_confidence="escalate",
        deterministic_checks=[ShellCheckConfig(name="still_breaching", run="exit 0")],
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )

    assert result.status == "completed"
    assert result.trust_report["combined_trust"] == result.effective_confidence == 0.95


async def test_core_invariant_no_new_config_means_identical_behaviour():
    """The single hardest requirement in this spec: a step declaring neither
    grounding: nor deterministic_checks: must behave byte-for-byte as before."""
    primary_output = _make_output(confidence=0.95)
    runner = _runner(executors={"gateway": lambda: _StubExecutor(primary_output)})
    step = StepConfig(
        name="plain", executor="gateway", prompt_template="",
        confidence_threshold=0.5, on_low_confidence="escalate",
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )

    assert result.status == "completed"
    assert result.effective_confidence == 0.95
    assert result.trust_report is None
    assert result.grounding_score is None
    assert result.deterministic_passed is None


def test_gate_policy_trust_vector_vs_legacy_confidence():
    trust_vector_report = PipelineRunner._build_trust_report(
        primary_confidence=0.9, effective_confidence=0.9, verifier_confidence=None,
        verifier_mode=None,
        grounding_score=0.9, grounding_report={"computed": True},
        deterministic_results=[{"name": "a", "type": "shell", "passed": True, "detail": "", "duration_ms": 1}],
        calibration_report=None,
        combined_trust=0.9, gate_policy="trust_vector",
    )
    assert trust_vector_report["gate"]["policy"] == "trust_vector"
    assert trust_vector_report["mode"] == "enforced"

    legacy_report = PipelineRunner._build_trust_report(
        primary_confidence=0.9, effective_confidence=0.9, verifier_confidence=None,
        verifier_mode=None,
        grounding_score=0.4, grounding_report={"computed": True},
        deterministic_results=None,
        calibration_report=None,
        combined_trust=0.9, gate_policy="legacy_confidence",
    )
    assert legacy_report["gate"]["policy"] == "legacy_confidence"
    assert legacy_report["mode"] == "shadow"


# ---------------------------------------------------------------------------
# 8. Persistence
# ---------------------------------------------------------------------------

async def test_deterministic_passed_persisted(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    sf = get_session_factory()

    failing_output = _make_output(confidence=0.95)
    plain_output = _make_output(confidence=0.9)

    runner = PipelineRunner(
        executors={
            "gateway": lambda: _StubExecutor(failing_output),
            "human": lambda: _StubExecutor(plain_output),
        },
        session_factory=sf,
    )
    checked_step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        on_low_confidence="proceed",
        deterministic_checks=[ShellCheckConfig(name="still_breaching", run="exit 1")],
    )
    plain_step = StepConfig(name="notify", executor="human", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[checked_step, plain_step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="run-d1")
    assert result.status == "completed"

    async with sf() as session:
        rows = (await session.execute(
            select(PipelineStep).where(PipelineStep.run_id == "run-d1").order_by(PipelineStep.step_index)
        )).scalars().all()

    checked_row, plain_row = rows
    assert checked_row.deterministic_passed is False
    parsed = json.loads(checked_row.trust_report)
    assert parsed["deterministic_checks"]
    assert len(parsed["deterministic_checks"]) == 1

    assert plain_row.deterministic_passed is None


# ---------------------------------------------------------------------------
# 9. Migration
# ---------------------------------------------------------------------------

async def test_deterministic_passed_column_exists_after_create_tables(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    await create_tables()  # idempotent

    sf = get_session_factory()
    async with sf() as session:
        conn = await session.connection()
        columns_result = await conn.exec_driver_sql("PRAGMA table_info(pipeline_steps)")
        columns = {row[1] for row in columns_result.fetchall()}

    assert "deterministic_passed" in columns


# ---------------------------------------------------------------------------
# 10. Metric
# ---------------------------------------------------------------------------

def _find_family(families, sample_name):
    return next(f for f in families if any(s.name == sample_name for s in f.samples))


def test_collect_emits_pork_step_deterministic_check_total():
    data = MetricsData(
        run_counts=[], runs_in_progress=0, step_counts=[], step_durations=[],
        verifier_counts=[], token_usage=[], human_decisions=[], feedback_counts=[],
        step_feedback_counts=[], grounding_scores=[],
        deterministic_check_counts=[
            ("p", "investigate", "passed", 3),
            ("p", "investigate", "failed", 1),
        ],
    )
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_step_deterministic_check_total")

    by_outcome = {s.labels["outcome"]: s.value for s in family.samples}
    assert by_outcome["passed"] == 3
    assert by_outcome["failed"] == 1


async def test_fetch_metrics_data_deterministic_check_counts_excludes_null_and_testing(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    sf = get_session_factory()

    from src.db.models import PipelineRun

    async with sf() as session:
        session.add(PipelineRun(
            id="run-prod", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="production",
        ))
        session.add(PipelineRun(
            id="run-test", pipeline_name="p", source="generic", status="completed",
            normalised_context="{}", raw_payload="{}", stage="testing",
        ))
        session.add(PipelineStep(
            id="step-checked", run_id="run-prod", step_name="investigate", step_index=0,
            executor="gateway", prompt="", status="completed", executed_at=datetime(2026, 1, 1),
            deterministic_passed=True,
        ))
        session.add(PipelineStep(
            id="step-null", run_id="run-prod", step_name="triage", step_index=1,
            executor="human", prompt="", status="completed", executed_at=datetime(2026, 1, 1),
            deterministic_passed=None,
        ))
        session.add(PipelineStep(
            id="step-testing", run_id="run-test", step_name="investigate", step_index=0,
            executor="gateway", prompt="", status="completed", executed_at=datetime(2026, 1, 1),
            deterministic_passed=False,
        ))
        await session.commit()

    metrics_data = await fetch_metrics_data(sf)

    assert metrics_data.deterministic_check_counts == [("p", "investigate", "passed", 1)]
