"""Tests for _run_verifier_impl (SPEC-verifier-semantics.md §8): a 'critic' verifier
sees the primary's prompt+response, an 'independent' verifier runs blind — and the
legacy 'challenger' alias behaves identically to 'independent'."""
from src.models.llm import LLMOutput
from src.models.pipeline import StepConfig, VerifierConfig
from src.pipeline.runner import PipelineRunner


class _CapturingExecutor:
    def __init__(self):
        self.captured_ctx: dict = {}

    async def execute(self, step, ctx):
        self.captured_ctx = ctx
        return LLMOutput(confidence=0.8, summary="ok", next_step_context="", raw_response={})


def _runner(executor) -> PipelineRunner:
    return PipelineRunner(executors={"stub": lambda: executor}, session_factory=None)


def _step_with_mode(mode: str) -> StepConfig:
    return StepConfig(
        name="investigate",
        executor="stub",
        prompt_template="Investigate {{summary}}",
        verifier=VerifierConfig(executor="stub", mode=mode),
    )


def _primary_output() -> LLMOutput:
    return LLMOutput(confidence=0.9, summary="root cause found", next_step_context="", raw_response={})


async def test_independent_mode_runs_blind_no_primary_context():
    executor = _CapturingExecutor()
    runner = _runner(executor)
    step = _step_with_mode("independent")

    await runner._run_verifier_impl(
        step=step, ctx={"summary": "high error rate"}, primary_output=_primary_output(), run_log=[],
    )

    assert "primary_prompt" not in executor.captured_ctx
    assert "primary_response" not in executor.captured_ctx


async def test_critic_mode_shares_primary_prompt_and_response():
    executor = _CapturingExecutor()
    runner = _runner(executor)
    step = _step_with_mode("critic")

    await runner._run_verifier_impl(
        step=step, ctx={"summary": "high error rate"}, primary_output=_primary_output(), run_log=[],
    )

    assert "primary_prompt" in executor.captured_ctx
    assert "primary_response" in executor.captured_ctx


async def test_legacy_challenger_alias_behaves_like_independent():
    executor = _CapturingExecutor()
    runner = _runner(executor)
    step = _step_with_mode("challenger")  # coerced to "independent" at parse time

    assert step.verifier.mode == "independent"

    await runner._run_verifier_impl(
        step=step, ctx={"summary": "high error rate"}, primary_output=_primary_output(), run_log=[],
    )

    assert "primary_prompt" not in executor.captured_ctx
    assert "primary_response" not in executor.captured_ctx
