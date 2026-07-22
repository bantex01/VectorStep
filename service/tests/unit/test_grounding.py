"""Tests for shadow-mode grounding (SPEC-grounding-shadow.md): trace formatting,
_run_grounding, _build_trust_report, the shadow-means-no-gate-change guarantee,
persistence, migration, and the pork_step_grounding_score metric."""
import json
from datetime import datetime

from sqlalchemy import select

from src.db.database import create_tables, get_session_factory, init_db
from src.db.models import PipelineStep
from src.metrics import MetricsData, PorkCollector, fetch_metrics_data
from src.models.context import NormalisedContext
from src.models.llm import LLMOutput
from src.models.pipeline import GroundingConfig, PipelineConfig, StepConfig, TriggerConfig
from src.pipeline.runner import PipelineRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubExecutor:
    def __init__(self, output):
        self._output = output

    async def execute(self, step, ctx):
        return self._output


class _RaisingExecutor:
    async def execute(self, step, ctx):
        raise RuntimeError("gateway unreachable")


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


_TOOL_TRACE = [
    {"type": "tool_call", "name": "grafana_query", "input": {"q": "rate(http_5xx[5m])"}},
    {"type": "tool_result", "name": "grafana_query", "content": "0.9%", "is_error": False},
]


def _runner(executors=None) -> PipelineRunner:
    return PipelineRunner(executors=executors or {}, session_factory=None)


# ---------------------------------------------------------------------------
# 1. _format_trace_for_grounding
# ---------------------------------------------------------------------------

def test_format_trace_renders_tool_call_and_result_with_error_marker():
    runner = _runner()
    trace = [
        {"type": "tool_call", "name": "grafana_query", "input": {"q": "up"}},
        {"type": "tool_result", "name": "grafana_query", "content": "connection refused", "is_error": True},
    ]

    out = runner._format_trace_for_grounding(trace)

    assert "TOOL CALL: grafana_query" in out
    assert "TOOL RESULT [ERROR] (grafana_query): connection refused" in out


def test_format_trace_truncates_long_content():
    runner = _runner()
    trace = [{"type": "tool_result", "name": "logs", "content": "x" * 5000, "is_error": False}]

    out = runner._format_trace_for_grounding(trace, max_chars=100)

    assert len(out) < 200
    assert out.endswith("…")


def test_format_trace_with_no_tool_activity_returns_empty():
    runner = _runner()
    # An llm_call event carries no evidence — no tool_call/tool_result/text content.
    trace = [{"type": "llm_call", "iteration": 1}]

    assert runner._format_trace_for_grounding(trace) == ""
    assert runner._format_trace_for_grounding([]) == ""
    assert runner._format_trace_for_grounding(None) == ""


# ---------------------------------------------------------------------------
# 2. G extraction + report
# ---------------------------------------------------------------------------

async def test_run_grounding_extracts_score_and_claims():
    judge_output = LLMOutput(
        confidence=0.9,
        summary="3 of 4 claims supported",
        next_step_context="",
        reasoning={"claims": [{"claim": "pool exhaustion", "supported": True, "evidence": "pool-wait metric"}]},
        raw_response={},
        model="gpt-5",
        provider="azure",
    )
    runner = _runner(executors={"grounding_stub": lambda: _StubExecutor(judge_output)})
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        grounding=GroundingConfig(agent="grounding-judge", executor="grounding_stub"),
    )
    primary = _make_output(confidence=0.9, raw_response={"trace": _TOOL_TRACE})

    g, report, tokens = await runner._run_grounding(step=step, ctx={}, primary_output=primary, run_log=[])

    assert g == 0.9
    assert report["computed"] is True
    assert report["claims"] == [{"claim": "pool exhaustion", "supported": True, "evidence": "pool-wait metric"}]
    assert report["agent"] == "grounding-judge"
    assert report["model"] == "gpt-5"
    assert report["provider"] == "azure"


async def test_run_grounding_shares_original_task_prompt_with_judge():
    """Without the original task, the judge can't tell 'restates a fact it was given
    as input' (e.g. alert severity) apart from 'claims something it needed to discover'
    — it would mark given facts unsupported for lack of a matching tool result."""
    captured_ctx = {}

    class _CapturingExecutor:
        async def execute(self, step, ctx):
            captured_ctx.update(ctx)
            return LLMOutput(confidence=0.5, summary="x", next_step_context="", raw_response={})

    runner = _runner(executors={"grounding_stub": lambda: _CapturingExecutor()})
    step = StepConfig(
        name="investigate", executor="gateway",
        prompt_template="Alert severity: {{severity}}. Investigate.",
        grounding=GroundingConfig(executor="grounding_stub"),
    )
    primary = _make_output(confidence=0.9, raw_response={"trace": _TOOL_TRACE})

    await runner._run_grounding(
        step=step, ctx={"severity": "critical"}, primary_output=primary, run_log=[],
    )

    assert captured_ctx["primary_prompt"] == "Alert severity: critical. Investigate."


async def test_run_grounding_report_includes_judges_own_rendered_prompt():
    """The gateway executor stashes its rendered prompt on raw_response['prompt'] — the
    grounding report must carry that through, so a reviewer can see exactly what the
    judge itself was asked, not just what it decided."""
    judge_output = LLMOutput(
        confidence=0.8, summary="ok", next_step_context="",
        raw_response={"prompt": "You are a grounding auditor... (judge's own rendered prompt)"},
    )
    runner = _runner(executors={"grounding_stub": lambda: _StubExecutor(judge_output)})
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding_stub"),
    )
    primary = _make_output(confidence=0.9, raw_response={"trace": _TOOL_TRACE})

    g, report, tokens = await runner._run_grounding(step=step, ctx={}, primary_output=primary, run_log=[])

    assert report["prompt"] == "You are a grounding auditor... (judge's own rendered prompt)"


async def test_run_grounding_excludes_artifacts_but_keeps_other_extra_fields():
    """artifacts (e.g. a full markdown report) is presentation content, not a claim —
    excluding it keeps the grounding call cheap and stops the judge quoting a large
    blob back. Other extra fields (where claims actually live, e.g. patterns_found)
    must still reach the judge."""
    captured_ctx = {}

    class _CapturingExecutor:
        async def execute(self, step, ctx):
            captured_ctx.update(ctx)
            return LLMOutput(confidence=0.5, summary="x", next_step_context="", raw_response={})

    runner = _runner(executors={"grounding_stub": lambda: _CapturingExecutor()})
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding_stub"),
    )
    primary = LLMOutput(
        confidence=0.9, summary="ok", next_step_context="",
        raw_response={"trace": _TOOL_TRACE},
        artifacts={"report_markdown": "# huge report\n" + ("x" * 5000)},
        patterns_found=[{"title": "recurring 500s", "ticket_keys": ["OC-1", "OC-2"]}],
    )

    await runner._run_grounding(step=step, ctx={}, primary_output=primary, run_log=[])

    assert "report_markdown" not in captured_ctx["primary_response"]
    assert "patterns_found" in captured_ctx["primary_response"]
    assert "recurring 500s" in captured_ctx["primary_response"]


def test_grounding_config_max_trace_chars_defaults_to_1500():
    assert GroundingConfig().max_trace_chars == 1500


async def test_run_grounding_respects_custom_max_trace_chars():
    """A claim whose supporting evidence lands past the truncation cutoff is invisible
    to the judge — max_trace_chars lets a step raise that cutoff instead of silently
    producing false 'unsupported' verdicts on long tool results."""
    captured_ctx = {}

    class _CapturingExecutor:
        async def execute(self, step, ctx):
            captured_ctx.update(ctx)
            return LLMOutput(confidence=0.5, summary="x", next_step_context="", raw_response={})

    runner = _runner(executors={"grounding_stub": lambda: _CapturingExecutor()})
    long_content = "x" * 5000
    trace = [{"type": "tool_result", "name": "confluence", "content": long_content, "is_error": False}]
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding_stub", max_trace_chars=4000),
    )
    primary = _make_output(confidence=0.9, raw_response={"trace": trace})

    await runner._run_grounding(step=step, ctx={}, primary_output=primary, run_log=[])

    assert "x" * 4000 in captured_ctx["agent_trace"]
    assert "x" * 4001 not in captured_ctx["agent_trace"]


async def test_run_grounding_clamps_score_to_unit_interval():
    judge_output = LLMOutput(confidence=1.4, summary="x", next_step_context="", raw_response={})
    runner = _runner(executors={"grounding_stub": lambda: _StubExecutor(judge_output)})
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding_stub"),
    )
    primary = _make_output(raw_response={"trace": _TOOL_TRACE})

    g, _, _ = await runner._run_grounding(step=step, ctx={}, primary_output=primary, run_log=[])

    assert g == 1.0


# ---------------------------------------------------------------------------
# 3. No trace -> G is null, not zero
# ---------------------------------------------------------------------------

async def test_run_grounding_no_trace_returns_null_not_zero():
    runner = _runner(executors={"grounding_stub": lambda: _StubExecutor(_make_output())})
    step = StepConfig(
        name="human-step", executor="human", prompt_template="",
        grounding=GroundingConfig(executor="grounding_stub"),
    )
    primary = _make_output(raw_response={})  # no trace at all

    g, report, tokens = await runner._run_grounding(step=step, ctx={}, primary_output=primary, run_log=[])

    assert g is None
    assert report == {"computed": False, "reason": "no_trace", "agent": "grounding-judge", "enforce": False}
    assert tokens == 0


# ---------------------------------------------------------------------------
# 4. Soft failure
# ---------------------------------------------------------------------------

async def test_run_grounding_soft_fails_on_executor_error():
    runner = _runner(executors={"grounding_stub": lambda: _RaisingExecutor()})
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding_stub"),
    )
    primary = _make_output(raw_response={"trace": _TOOL_TRACE})
    run_log: list = []

    g, report, tokens = await runner._run_grounding(step=step, ctx={}, primary_output=primary, run_log=run_log)

    assert g is None
    assert report["computed"] is False
    assert report["reason"] == "error"
    assert "gateway unreachable" in report["error"]
    assert tokens == 0
    assert any(e["event"] == "grounding_failed" for e in run_log)


# ---------------------------------------------------------------------------
# 5. _build_trust_report shape
# ---------------------------------------------------------------------------

def test_build_trust_report_shape():
    report = PipelineRunner._build_trust_report(
        primary_confidence=0.88,
        effective_confidence=0.86,
        verifier_confidence=0.85,
        verifier_mode="critic",
        verifier_combination_strategy="minimum",
        verifier_veto_floor=None,
        grounding_score=0.15,
        grounding_report={"computed": True, "claims": []},
        deterministic_results=None,
        calibration_report=None,
        combined_trust=0.86,
        gate_policy="legacy_confidence",
        confidence_threshold=0.75,
        on_low_confidence="escalate",
    )

    assert report["version"] == 5
    assert report["mode"] == "shadow"
    assert report["signals"] == {
        "S": 0.88, "S_after_V": 0.86, "V": 0.85, "V_mode": "critic",
        "V_combination_strategy": "minimum", "V_veto_floor": None,
        "G": 0.15, "C": None, "D": None,
    }
    assert report["combined_trust"] == 0.86
    assert report["deterministic_checks"] is None
    assert report["gate"] == {
        "policy": "legacy_confidence", "confidence_threshold": 0.75, "on_low_confidence": "escalate",
    }


# ---------------------------------------------------------------------------
# 6. Shadow = no gate change (the critical test)
# ---------------------------------------------------------------------------

async def test_low_grounding_score_does_not_abort_or_escalate_the_step():
    """A step that clears its confidence threshold must complete exactly as it
    would without grounding, even when G comes back at 0.0 — shadow mode never
    gates. This is the whole point of the spec; don't skip it."""
    primary_output = _make_output(confidence=0.9, raw_response={"trace": _TOOL_TRACE})
    grounding_output = LLMOutput(confidence=0.0, summary="nothing supported", next_step_context="", raw_response={})

    runner = _runner(executors={
        "gateway": lambda: _StubExecutor(primary_output),
        "grounding_stub": lambda: _StubExecutor(grounding_output),
    })
    step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        confidence_threshold=0.75, on_low_confidence="escalate",
        grounding=GroundingConfig(executor="grounding_stub"),
    )
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )

    assert result.status == "completed"
    assert result.effective_confidence == 0.9
    assert result.grounding_score == 0.0
    assert result.trust_report["signals"]["G"] == 0.0


async def test_step_without_grounding_block_leaves_fields_none():
    primary_output = _make_output(confidence=0.9)
    runner = _runner(executors={"gateway": lambda: _StubExecutor(primary_output)})
    step = StepConfig(name="plain", executor="gateway", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[step])

    result = await runner._run_step_impl(
        step=step, index=0, pipeline=pipeline, normalised=_make_normalised(),
        run_id="r1", step_outputs={}, run_log=[],
    )

    assert result.status == "completed"
    assert result.grounding_score is None
    assert result.trust_report is None


# ---------------------------------------------------------------------------
# 7. Persistence
# ---------------------------------------------------------------------------

async def test_grounding_score_and_trust_report_persisted(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    sf = get_session_factory()

    primary_output = _make_output(confidence=0.9, raw_response={"trace": _TOOL_TRACE})
    grounding_output = LLMOutput(
        confidence=0.6, summary="partial", next_step_context="",
        reasoning={"claims": [{"claim": "x", "supported": True, "evidence": "y"}]},
        raw_response={},
    )
    plain_output = _make_output(confidence=0.9)

    runner = PipelineRunner(
        executors={
            "gateway": lambda: _StubExecutor(primary_output),
            "grounding_stub": lambda: _StubExecutor(grounding_output),
            "human": lambda: _StubExecutor(plain_output),
        },
        session_factory=sf,
    )
    grounded_step = StepConfig(
        name="investigate", executor="gateway", prompt_template="",
        grounding=GroundingConfig(executor="grounding_stub"),
    )
    ungrounded_step = StepConfig(name="notify", executor="human", prompt_template="")
    pipeline = PipelineConfig(name="p", trigger=TriggerConfig(), steps=[grounded_step, ungrounded_step])

    result = await runner.run(pipeline=pipeline, normalised=_make_normalised(), run_id="run-g1")
    assert result.status == "completed"

    async with sf() as session:
        rows = (await session.execute(
            select(PipelineStep).where(PipelineStep.run_id == "run-g1").order_by(PipelineStep.step_index)
        )).scalars().all()

    grounded_row, ungrounded_row = rows
    assert grounded_row.grounding_score == 0.6
    parsed_report = json.loads(grounded_row.trust_report)
    assert parsed_report["signals"]["G"] == 0.6

    assert ungrounded_row.grounding_score is None
    assert ungrounded_row.trust_report is None


# ---------------------------------------------------------------------------
# 8. Migration
# ---------------------------------------------------------------------------

async def test_grounding_columns_exist_after_create_tables(tmp_path):
    init_db(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await create_tables()
    await create_tables()  # idempotent, mirrors test_database_migrations.py

    sf = get_session_factory()
    async with sf() as session:
        conn = await session.connection()
        columns_result = await conn.exec_driver_sql("PRAGMA table_info(pipeline_steps)")
        columns = {row[1] for row in columns_result.fetchall()}

    assert "grounding_score" in columns
    assert "trust_report" in columns


# ---------------------------------------------------------------------------
# 9. Metric
# ---------------------------------------------------------------------------

def _find_family(families, name):
    return next(f for f in families if f.name == name)


def test_collect_emits_pork_step_grounding_score():
    data = MetricsData(
        run_counts=[], runs_in_progress=0, step_counts=[], step_durations=[],
        verifier_counts=[], token_usage=[], human_decisions=[], feedback_counts=[],
        step_feedback_counts=[],
        grounding_scores=[
            ("p", "investigate", "agent-x", "claude-sonnet-5", "anthropic", 0.9),
            ("p", "investigate", "agent-x", "claude-sonnet-5", "anthropic", 0.3),
        ],
        deterministic_check_counts=[],
    )
    families = list(PorkCollector(data).collect())
    family = _find_family(families, "pork_step_grounding_score")

    sum_sample = next(s for s in family.samples if s.name == "pork_step_grounding_score_sum")
    assert sum_sample.value == 0.9 + 0.3
    assert sum_sample.labels["pipeline"] == "p"
    assert sum_sample.labels["step_name"] == "investigate"
    assert sum_sample.labels["agent"] == "agent-x"
    assert sum_sample.labels["model"] == "claude-sonnet-5"
    assert sum_sample.labels["provider"] == "anthropic"

    count_sample = next(s for s in family.samples if s.name == "pork_step_grounding_score_count")
    assert count_sample.value == 2


async def test_fetch_metrics_data_excludes_null_and_testing_stage_grounding(tmp_path):
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
            id="step-scored", run_id="run-prod", step_name="investigate", step_index=0,
            executor="gateway", agent="agent-x", model="m", provider="anthropic",
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
            grounding_score=0.7,
        ))
        session.add(PipelineStep(
            id="step-null", run_id="run-prod", step_name="triage", step_index=1,
            executor="human", agent=None, model=None, provider=None,
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
            grounding_score=None,
        ))
        session.add(PipelineStep(
            id="step-testing", run_id="run-test", step_name="investigate", step_index=0,
            executor="gateway", agent="agent-x", model="m", provider="anthropic",
            prompt="", status="completed", executed_at=datetime(2026, 1, 1),
            grounding_score=0.5,
        ))
        await session.commit()

    metrics_data = await fetch_metrics_data(sf)

    assert metrics_data.grounding_scores == [("p", "investigate", "agent-x", "m", "anthropic", 0.7)]
