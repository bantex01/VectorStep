"""Tests for PipelineRunner._set_step_span_attributes — OTel span attributes
carrying prompt_hash/agent_version (SPEC-prompt-versioning.md follow-up).
Deliberately trace-only: see the spec discussion for why these are NOT added
as Prometheus metric labels (unbounded cardinality — every prompt/agent edit
mints a new, permanently-retained label value)."""
from unittest.mock import MagicMock

from src.models.llm import LLMOutput
from src.pipeline.runner import PipelineRunner, StepResult
from src.pipeline.versioning import prompt_hash


def _make_runner() -> PipelineRunner:
    return PipelineRunner(executors={})


def _make_result(output: LLMOutput | None, status: str = "completed") -> StepResult:
    return StepResult(
        step_name="investigate", step_index=0, status=status,
        output=output, verifier_output=None, effective_confidence=0.9, duration_ms=10,
    )


def _make_output(agent_version: str | None = None) -> LLMOutput:
    return LLMOutput(
        confidence=0.9, summary="ok", next_step_context="", raw_response={},
        model="claude-sonnet-5", provider="anthropic", agent_version=agent_version,
    )


class TestPromptHashAttribute:
    def test_sets_prompt_hash_when_template_given(self):
        runner = _make_runner()
        span = MagicMock()
        template = "Investigate this alert.\nBe concise."
        result = _make_result(_make_output())

        runner._set_step_span_attributes(span, result, prompt_template=template)

        span.set_attribute.assert_any_call("pork.prompt_hash", prompt_hash(template))

    def test_no_prompt_hash_attribute_when_template_not_passed(self):
        """Parallel groups pass no template — each branch has its own distinct
        one, so no single hash is meaningful at the group span level."""
        runner = _make_runner()
        span = MagicMock()
        result = _make_result(_make_output())

        runner._set_step_span_attributes(span, result)

        calls = [c.args[0] for c in span.set_attribute.call_args_list]
        assert "pork.prompt_hash" not in calls

    def test_no_prompt_hash_attribute_for_empty_template(self):
        """Non-LLM steps (webhook/notify/human) have no real prompt_template —
        prompt_hash() returns None for these, and no attribute should be set
        rather than a hash of the empty string."""
        runner = _make_runner()
        span = MagicMock()
        result = _make_result(_make_output())

        runner._set_step_span_attributes(span, result, prompt_template="")

        calls = [c.args[0] for c in span.set_attribute.call_args_list]
        assert "pork.prompt_hash" not in calls

    def test_fan_out_group_template_produces_same_hash_as_a_branch_with_it(self):
        """Sanity check that this is the exact same hashing function used
        everywhere else — no drift between what a span shows and what the DB
        records for the same template text."""
        runner = _make_runner()
        span = MagicMock()
        template = "Investigate {{item}}."
        result = _make_result(_make_output())

        runner._set_step_span_attributes(span, result, prompt_template=template)

        span.set_attribute.assert_any_call("pork.prompt_hash", prompt_hash(template))


class TestAgentVersionAttribute:
    def test_sets_agent_version_when_present(self):
        runner = _make_runner()
        span = MagicMock()
        result = _make_result(_make_output(agent_version="91f02ab3c7de"))

        runner._set_step_span_attributes(span, result)

        span.set_attribute.assert_any_call("pork.agent_version", "91f02ab3c7de")

    def test_no_agent_version_attribute_when_none(self):
        runner = _make_runner()
        span = MagicMock()
        result = _make_result(_make_output(agent_version=None))

        runner._set_step_span_attributes(span, result)

        calls = [c.args[0] for c in span.set_attribute.call_args_list]
        assert "pork.agent_version" not in calls

    def test_no_agent_version_attribute_when_no_output_at_all(self):
        runner = _make_runner()
        span = MagicMock()
        result = _make_result(output=None, status="failed")

        runner._set_step_span_attributes(span, result)

        calls = [c.args[0] for c in span.set_attribute.call_args_list]
        assert "pork.agent_version" not in calls
        assert "pork.prompt_hash" not in calls
