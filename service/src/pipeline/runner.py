import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from opentelemetry.trace import Status, StatusCode
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..artifacts.store import ArtifactStore
from ..db.models import PipelineRun, PipelineStep
from ..executors.base import BaseExecutor
from ..models.context import NormalisedContext
from ..models.llm import LLMOutput
from .. import run_events
from ..tracing import record_event, start_root_span, tracer
from ..utils import utc_now
from ..models.pipeline import (
    FanOutConfig,
    FanOutGroupConfig,
    LoopConfig,
    ParallelGroupConfig,
    ParallelGroupInner,
    ParallelStepConfig,
    PipelineConfig,
    RetryConfig,
    StepConfig,
    VerifierConfig,
)
from ..notifications.telegram import TelegramNotifier
from .context import build_context

logger = logging.getLogger(__name__)


def _compute_backoff(retry: RetryConfig | None, attempt: int) -> float:
    """Return delay in seconds before the next retry attempt."""
    if retry is None:
        return 1.0
    if retry.backoff == "fixed":
        return retry.delay_seconds
    return retry.delay_seconds * (2 ** (attempt - 1))


class _LiveRunLog(list):
    """list subclass that publishes each appended event to the live event bus.

    Using a subclass means every existing _log_event(run_log, ...) call site
    automatically gets live-publishing behaviour with no other changes.
    """
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def append(self, entry: object) -> None:
        super().append(entry)
        if isinstance(entry, dict):
            run_events.publish(self._run_id, entry)


def _log_event(run_log: list, level: str, event: str, msg: str, **extra) -> None:
    run_log.append({
        "ts": utc_now().isoformat(timespec="milliseconds") + "Z",
        "level": level,
        "event": event,
        "msg": msg,
        **extra,
    })
    record_event(event, attributes={"level": level, "msg": msg, **extra})


# Built-in prompt used when a verifier fires.
# Rendered with the same Jinja2 context as the primary step, plus:
#   {{primary_prompt}}   — the rendered prompt that was sent to the primary agent
#   {{primary_response}} — the primary agent's full JSON response text
_VERIFIER_PROMPT_TEMPLATE = """\
You are an independent reviewer assessing the quality and confidence of another \
agent's analysis.

Original task given to the primary agent:
---
{{primary_prompt}}
---

Primary agent's response:
---
{{primary_response}}
---

Review the reasoning above. Assess whether the primary agent's conclusion is \
well-supported, considers the right evidence, and has appropriate confidence.

Return JSON only, no other text:
{
  "confidence": 0.0,
  "summary": "One sentence: your overall verdict on the primary agent's response",
  "next_step_context": "",
  "reasoning": {
    "assessment": "Your detailed assessment of the primary agent's response",
    "gaps": "Any evidence gaps, logical leaps, or concerns",
    "confidence_rationale": "Why you scored confidence higher or lower than the primary agent"
  }
}
"""


@dataclass
class StepResult:
    step_name: str
    step_index: int
    status: Literal["completed", "stopped", "aborted", "escalated", "failed"]
    output: LLMOutput | None
    verifier_output: LLMOutput | None
    effective_confidence: float | None
    duration_ms: int
    # Populated for parallel groups — each branch output keyed by branch name.
    # Empty for sequential steps.
    branch_outputs: dict[str, "LLMOutput | None"] = field(default_factory=dict)
    # Token usage from the primary executor (+ verifier for sequential steps;
    # sum of all branches for parallel/fan-out groups). 0 when unavailable.
    total_tokens: int = 0


@dataclass
class PipelineRunResult:
    run_id: str
    pipeline_name: str
    status: Literal["completed", "stopped", "aborted", "escalated", "failed"]
    steps: list[StepResult] = field(default_factory=list)
    final_output: LLMOutput | None = None
    abort_reason: str | None = None


class PipelineRunner:
    def __init__(
        self,
        executors: dict[str, type[BaseExecutor]],
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        notifiers: dict[str, TelegramNotifier] | None = None,
        artifact_store: ArtifactStore | None = None,
        pipeline_registry: "dict[str, PipelineConfig] | None" = None,
    ):
        self._executor_classes = executors
        self._executor_instances: dict[str, BaseExecutor] = {}
        self._session_factory = session_factory
        self._notifiers: dict[str, TelegramNotifier] = notifiers or {}
        self._artifact_store = artifact_store
        self._pipeline_registry = pipeline_registry

    def set_pipeline_registry(self, registry: "dict[str, PipelineConfig]") -> None:
        self._pipeline_registry = registry

    @staticmethod
    def _extract_usage(raw_response: dict) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) from a gateway raw_response, or (0, 0)."""
        usage = ((raw_response or {}).get("meta") or {}).get("agentMeta", {}).get("usage") or {}
        return (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))

    async def run(
        self,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str | None = None,
        from_step: str | None = None,
        initial_step_outputs: "dict[str, LLMOutput] | None" = None,
        parent_run_id: str | None = None,
    ) -> PipelineRunResult:
        run_id = run_id or str(uuid.uuid4())
        run_log: _LiveRunLog = _LiveRunLog(run_id)

        with start_root_span(
            "pipeline.run",
            attributes={
                "pork.pipeline.name": pipeline.name,
                "pork.run.id": run_id,
                "pork.source": normalised.source,
                **({"pork.parent_run_id": parent_run_id} if parent_run_id else {}),
            },
        ) as run_span:
            result = await self._run_pipeline_body(
                pipeline, normalised, run_id, run_log,
                from_step=from_step,
                initial_step_outputs=initial_step_outputs,
                parent_run_id=parent_run_id,
            )
            run_span.set_attribute("pork.run.status", result.status)
            if result.abort_reason:
                run_span.set_attribute("pork.run.abort_reason", result.abort_reason)
            if result.status == "failed":
                run_span.set_status(Status(StatusCode.ERROR))

        return result

    async def _run_pipeline_body(
        self,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        run_log: "_LiveRunLog",
        from_step: str | None = None,
        initial_step_outputs: "dict[str, LLMOutput] | None" = None,
        parent_run_id: str | None = None,
    ) -> PipelineRunResult:
        logger.info("Pipeline run started: id=%s pipeline=%s", run_id, pipeline.name)
        _log_event(run_log, "info", "run_started",
                   f"Pipeline started: {pipeline.name} (source: {normalised.source})")

        if from_step:
            n = len(initial_step_outputs or {})
            logger.info("Re-run from step '%s': %d prior step output(s) pre-loaded", from_step, n)
            _log_event(run_log, "info", "rerun_started",
                       f"Re-run from step: {from_step} ({n} prior step output(s) replayed)")

        await self._db_create_run(run_id, pipeline, normalised, parent_run_id=parent_run_id)

        result = PipelineRunResult(
            run_id=run_id,
            pipeline_name=pipeline.name,
            status="completed",
        )

        step_outputs: dict[str, LLMOutput] = dict(initial_step_outputs or {})
        skipping = from_step is not None
        accumulated_tokens = 0

        for index, step in enumerate(pipeline.steps):
            if isinstance(step, ParallelGroupConfig):
                step_name = step.parallel.name
                step_when = step.parallel.when
                step_on_abort = step.parallel.on_abort
                step_threshold = step.parallel.confidence_threshold
            elif isinstance(step, FanOutGroupConfig):
                step_name = step.fan_out.name
                step_when = step.fan_out.when
                step_on_abort = step.fan_out.on_abort
                step_threshold = step.fan_out.confidence_threshold
            else:
                step_name = step.name
                step_when = step.when
                step_on_abort = step.on_abort
                step_threshold = step.confidence_threshold

            # Skip steps before from_step when replaying a re-run
            if skipping:
                if step_name == from_step:
                    skipping = False
                else:
                    continue

            # Evaluate when: condition before doing any work
            if step_when is not None:
                when_ctx = await build_context(pipeline, normalised, run_id, step_name, step_outputs)
                if not self._eval_when(step_when, when_ctx):
                    logger.info("Step '%s' skipped — when: condition was false", step_name)
                    _log_event(run_log, "info", "step_skipped",
                               f"Step skipped: {step_name} — when: condition was false",
                               step=step_name)
                    continue

            if isinstance(step, ParallelGroupConfig):
                step_result = await self._run_parallel_group(
                    group=step.parallel,
                    index=index,
                    pipeline=pipeline,
                    normalised=normalised,
                    run_id=run_id,
                    step_outputs=step_outputs,
                    run_log=run_log,
                )
                # Branches saved inside _run_parallel_group; register each individually
                # so downstream steps can reference {{steps.branch_name.field}}
                for branch_name, branch_output in step_result.branch_outputs.items():
                    if branch_output:
                        step_outputs[branch_name] = branch_output
            elif isinstance(step, FanOutGroupConfig):
                step_result = await self._run_fan_out(
                    fan_out=step.fan_out,
                    index=index,
                    pipeline=pipeline,
                    normalised=normalised,
                    run_id=run_id,
                    step_outputs=step_outputs,
                    run_log=run_log,
                )
                for branch_name, branch_output in step_result.branch_outputs.items():
                    if branch_output:
                        step_outputs[branch_name] = branch_output
            else:
                step_result = await self._run_step(
                    step=step,
                    index=index * 1000,
                    pipeline=pipeline,
                    normalised=normalised,
                    run_id=run_id,
                    step_outputs=step_outputs,
                    run_log=run_log,
                )
                await self._db_save_step(run_id, step, step_result)
                if step_result.output:
                    step_outputs[step.name] = step_result.output

            result.steps.append(step_result)

            if step_result.status == "stopped":
                result.status = "stopped"
                result.final_output = step_result.output
                logger.info(
                    "Pipeline stopped at step '%s': %s",
                    step_name,
                    step_result.output.summary if step_result.output else "",
                )
                stopped_ctx = await build_context(pipeline, normalised, run_id, step_name, step_outputs)
                if step_result.output:
                    stopped_ctx["step_summary"] = step_result.output.summary
                    stopped_ctx["proceed_reason"] = step_result.output.proceed_reason or ""
                await self._dispatch_notification(
                    pipeline=pipeline,
                    action="stopped",
                    context=stopped_ctx,
                    run_log=run_log,
                )
                break

            if step_result.status in ("aborted", "escalated", "failed"):
                result.status = step_result.status
                result.abort_reason = (
                    step_result.output.summary if step_result.output else "unknown"
                )
                action = step_on_abort if step_result.status == "aborted" else "escalate"
                escalation_ctx = self._build_escalation_context(step_threshold, step_result)
                await self._dispatch_notification(
                    pipeline=pipeline,
                    action=action,
                    context={
                        **await build_context(pipeline, normalised, run_id, step_name, step_outputs),
                        **escalation_ctx,
                    },
                    run_log=run_log,
                )
                break

            result.final_output = step_result.output

            # Budget guardrail — check after registering the completed step's output
            accumulated_tokens += step_result.total_tokens
            if (
                pipeline.budget
                and pipeline.budget.max_tokens
                and accumulated_tokens > pipeline.budget.max_tokens
            ):
                msg = (
                    f"Token budget exceeded: {accumulated_tokens:,} tokens used "
                    f"(limit: {pipeline.budget.max_tokens:,})"
                )
                logger.warning("Pipeline '%s' %s", pipeline.name, msg)
                _log_event(run_log, "warn", "budget_exceeded", msg)
                result.status = "aborted"
                result.abort_reason = msg
                budget_ctx = await build_context(pipeline, normalised, run_id, step_name, step_outputs)
                budget_ctx["step_summary"] = msg
                await self._dispatch_notification(
                    pipeline=pipeline, action="notify", context=budget_ctx, run_log=run_log,
                )
                break

        _log_event(run_log, "info", "run_finished",
                   f"Pipeline finished: {result.status}")
        await self._db_complete_run(run_id, result.status, run_log)
        run_events.publish_complete(run_id, result.status)

        logger.info(
            "Pipeline run finished: id=%s pipeline=%s status=%s",
            run_id, pipeline.name, result.status,
        )
        return result

    # ------------------------------------------------------------------
    # Sequential step execution
    # ------------------------------------------------------------------

    def _inject_pork_context(self, ctx: dict, normalised: "NormalisedContext") -> None:
        """Inject internal runner references so executor: pipeline can call sub-pipelines."""
        if self._pipeline_registry is not None:
            ctx["_pork_runner"] = self
            ctx["_pork_normalised"] = normalised
            ctx["_pork_registry"] = self._pipeline_registry

    @staticmethod
    def _set_step_span_attributes(span, result: "StepResult") -> None:
        span.set_attribute("pork.step.status", result.status)
        if result.effective_confidence is not None:
            span.set_attribute("pork.confidence.effective", result.effective_confidence)
        if result.output:
            span.set_attribute("pork.confidence.primary", result.output.confidence)
            if result.output.model:
                span.set_attribute("pork.model", result.output.model)
        if result.verifier_output:
            span.set_attribute("pork.confidence.verifier", result.verifier_output.confidence)
        if result.status == "failed":
            span.set_status(Status(StatusCode.ERROR))

    async def _run_step(
        self,
        step: StepConfig,
        index: int,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        step_outputs: dict[str, LLMOutput],
        run_log: list,
    ) -> StepResult:
        agent = step.executor_config.get("agent", "")
        with tracer.start_as_current_span(
            step.name,
            attributes={
                "pork.span.kind": "step",
                "pork.executor": step.executor,
                "pork.agent": agent,
            },
        ) as span:
            result = await self._run_step_impl(
                step, index, pipeline, normalised, run_id, step_outputs, run_log
            )
            self._set_step_span_attributes(span, result)
            return result

    async def _run_step_impl(
        self,
        step: StepConfig,
        index: int,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        step_outputs: dict[str, LLMOutput],
        run_log: list,
    ) -> StepResult:
        start_ms = int(time.time() * 1000)

        agent = step.executor_config.get("agent", "")
        agent_label = f"{step.executor} / {agent}" if agent else step.executor
        logger.info("Executing step: run_id=%s step=%s (%d/%d)", run_id, step.name, index + 1, len(pipeline.steps))
        _log_event(run_log, "info", "step_started",
                   f"Step started: {step.name} [{agent_label}]",
                   step=step.name, executor=step.executor, agent=agent or None)

        loop_cfg = step.loop_until
        max_iters = loop_cfg.max_iterations if loop_cfg else 1
        primary_output: LLMOutput | None = None
        verifier_output: LLMOutput | None = None
        effective_confidence: float = 0.0

        for iteration in range(1, max_iters + 1):
            if loop_cfg and iteration > 1:
                _log_event(run_log, "info", "loop_iteration",
                           f"Refinement loop: {step.name} — iteration {iteration}/{max_iters} "
                           f"(prior confidence {effective_confidence:.0%})",
                           step=step.name, iteration=iteration, max_iterations=max_iters)

            ctx = await build_context(
                pipeline, normalised, run_id, step.name, step_outputs,
                artifact_store=self._artifact_store,
            )
            self._inject_pork_context(ctx, normalised)
            if loop_cfg:
                ctx["loop"] = {
                    "iteration": iteration,
                    "max_iterations": max_iters,
                    "prior_confidence": effective_confidence if iteration > 1 else None,
                    "prior_output": primary_output.model_dump(exclude={"raw_response"}) if iteration > 1 and primary_output else None,
                }

            max_attempts = step.retry.attempts if step.retry else 1
            last_error: str | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    executor = self._get_executor(step.executor)
                    coro = executor.execute(step, ctx)
                    if step.timeout_seconds:
                        primary_output = await asyncio.wait_for(coro, timeout=step.timeout_seconds)
                    else:
                        primary_output = await coro
                    if self._artifact_store:
                        primary_output = await self._intercept_artifacts(
                            primary_output, run_id, step.name
                        )
                    last_error = None
                    break
                except asyncio.TimeoutError:
                    last_error = f"Step timed out after {step.timeout_seconds}s"
                    logger.error(
                        "Step '%s' run_id=%s %s (attempt %d/%d)",
                        step.name, run_id, last_error, attempt, max_attempts,
                    )
                    _log_event(run_log, "error", "step_error",
                               f"Step error: {step.name} — {last_error}", step=step.name)
                except Exception as exc:
                    last_error = f"Executor error: {type(exc).__name__}: {exc}"
                    logger.error(
                        "Step '%s' run_id=%s %s (attempt %d/%d)",
                        step.name, run_id, last_error, attempt, max_attempts,
                    )
                    _log_event(run_log, "error", "step_error",
                               f"Step error: {step.name} — {last_error}", step=step.name)

                if attempt < max_attempts:
                    delay = _compute_backoff(step.retry, attempt)
                    logger.info(
                        "Retrying step '%s' in %.1fs (attempt %d/%d)",
                        step.name, delay, attempt + 1, max_attempts,
                    )
                    _log_event(run_log, "warn", "step_retrying",
                               f"Retrying {step.name} in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})",
                               step=step.name)
                    await asyncio.sleep(delay)

            if last_error:
                return StepResult(
                    step_name=step.name,
                    step_index=index,
                    status="failed",
                    output=LLMOutput(
                        confidence=0.0,
                        summary=last_error,
                        next_step_context="",
                        raw_response={},
                    ),
                    verifier_output=None,
                    effective_confidence=None,
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

            logger.debug(
                "Step '%s' parsed output: confidence=%.2f summary=%s",
                step.name, primary_output.confidence, primary_output.summary,
            )

            verifier_output = None
            effective_confidence = primary_output.confidence

            if step.verifier and self._should_verify(step.verifier, primary_output.confidence):
                verifier_output = await self._run_verifier(
                    step=step,
                    ctx=ctx,
                    primary_output=primary_output,
                    run_log=run_log,
                )
                if verifier_output:
                    effective_confidence = self._combine_confidence(
                        step=step,
                        primary_confidence=primary_output.confidence,
                        verifier_confidence=verifier_output.confidence,
                    )
                    logger.info(
                        "Step '%s' verifier: primary=%.2f verifier=%.2f effective=%.2f",
                        step.name,
                        primary_output.confidence,
                        verifier_output.confidence,
                        effective_confidence,
                    )
                    _log_event(run_log, "info", "verifier_ran",
                               f"Verifier ran: {step.name} — primary {primary_output.confidence:.0%} / "
                               f"verifier {verifier_output.confidence:.0%} → effective {effective_confidence:.0%}",
                               step=step.name)

            if loop_cfg and effective_confidence < loop_cfg.confidence and iteration < max_iters:
                continue

            break

        if loop_cfg and max_iters > 1:
            if effective_confidence >= loop_cfg.confidence:
                _log_event(run_log, "info", "loop_converged",
                           f"Refinement loop converged: {step.name} — confidence {effective_confidence:.0%} "
                           f"after {iteration} iteration(s)",
                           step=step.name, iterations=iteration)
            else:
                _log_event(run_log, "info", "loop_exhausted",
                           f"Refinement loop exhausted: {step.name} — best confidence {effective_confidence:.0%} "
                           f"after {max_iters} iteration(s)",
                           step=step.name, iterations=max_iters)

        _in_tok, _out_tok = self._extract_usage(primary_output.raw_response)
        if verifier_output:
            _vi, _vo = self._extract_usage(verifier_output.raw_response)
            _in_tok += _vi
            _out_tok += _vo
        _step_tokens = _in_tok + _out_tok

        if effective_confidence < step.confidence_threshold:
            action = step.on_low_confidence
            logger.info(
                "Step '%s' below confidence threshold (%.2f < %.2f): action=%s",
                step.name, effective_confidence, step.confidence_threshold, action,
            )
            conf_msg = (f"confidence {effective_confidence:.0%} < "
                        f"threshold {step.confidence_threshold:.0%}")
            if action == "abort":
                _log_event(run_log, "warn", "step_aborted",
                           f"Step aborted: {step.name} — {conf_msg}", step=step.name)
                return StepResult(
                    step_name=step.name,
                    step_index=index,
                    status="aborted",
                    output=primary_output,
                    verifier_output=verifier_output,
                    effective_confidence=effective_confidence,
                    duration_ms=int(time.time() * 1000) - start_ms,
                    total_tokens=_step_tokens,
                )
            if action == "escalate":
                _log_event(run_log, "warn", "step_escalated",
                           f"Step escalated: {step.name} — {conf_msg}", step=step.name)
                return StepResult(
                    step_name=step.name,
                    step_index=index,
                    status="escalated",
                    output=primary_output,
                    verifier_output=verifier_output,
                    effective_confidence=effective_confidence,
                    duration_ms=int(time.time() * 1000) - start_ms,
                    total_tokens=_step_tokens,
                )

        if not primary_output.proceed:
            logger.info(
                "Step '%s' returned proceed=false (confidence=%.2f) — resolving pipeline",
                step.name, effective_confidence,
            )
            _log_event(run_log, "info", "step_stopped",
                       f"Step stopped pipeline: {step.name} — "
                       f"{primary_output.proceed_reason or 'proceed=false'}",
                       step=step.name)
            return StepResult(
                step_name=step.name,
                step_index=index,
                status="stopped",
                output=primary_output,
                verifier_output=verifier_output,
                effective_confidence=effective_confidence,
                duration_ms=int(time.time() * 1000) - start_ms,
                total_tokens=_step_tokens,
            )

        duration_ms = int(time.time() * 1000) - start_ms
        _log_event(run_log, "info", "step_completed",
                   f"Step completed: {step.name} — confidence {effective_confidence:.0%} "
                   f"in {duration_ms / 1000:.1f}s",
                   step=step.name)
        return StepResult(
            step_name=step.name,
            step_index=index,
            status="completed",
            output=primary_output,
            verifier_output=verifier_output,
            effective_confidence=effective_confidence,
            duration_ms=duration_ms,
            total_tokens=_step_tokens,
        )

    # ------------------------------------------------------------------
    # Parallel group execution
    # ------------------------------------------------------------------

    async def _run_parallel_group(
        self,
        group: ParallelGroupInner,
        index: int,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        step_outputs: dict[str, LLMOutput],
        run_log: list,
    ) -> StepResult:
        with tracer.start_as_current_span(
            group.name,
            attributes={
                "pork.span.kind": "parallel_group",
                "pork.join_strategy": group.join,
                "pork.branch_count": len(group.steps),
            },
        ) as span:
            result = await self._run_parallel_group_impl(
                group, index, pipeline, normalised, run_id, step_outputs, run_log
            )
            self._set_step_span_attributes(span, result)
            return result

    async def _run_parallel_group_impl(
        self,
        group: ParallelGroupInner,
        index: int,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        step_outputs: dict[str, LLMOutput],
        run_log: list,
    ) -> StepResult:
        start_ms = int(time.time() * 1000)
        ctx = await build_context(
            pipeline, normalised, run_id, group.name, step_outputs,
            artifact_store=self._artifact_store,
        )
        self._inject_pork_context(ctx, normalised)

        logger.info(
            "Executing parallel group '%s' — %d branch(es)", group.name, len(group.steps)
        )
        _log_event(run_log, "info", "parallel_group_started",
                   f"Parallel group started: {group.name} ({len(group.steps)} branches)",
                   group=group.name)

        branch_coros = [
            self._run_parallel_branch(branch, ctx, run_log, run_id) for branch in group.steps
        ]

        try:
            if group.timeout_seconds:
                raw_results = await asyncio.wait_for(
                    asyncio.gather(*branch_coros, return_exceptions=True),
                    timeout=group.timeout_seconds,
                )
            else:
                raw_results = await asyncio.gather(*branch_coros, return_exceptions=True)
        except asyncio.TimeoutError:
            msg = f"Parallel group timed out after {group.timeout_seconds}s"
            logger.error("Parallel group '%s' %s", group.name, msg)
            _log_event(run_log, "error", "parallel_group_failed",
                       f"Parallel group timed out: {group.name}", group=group.name)
            return StepResult(
                step_name=group.name,
                step_index=index,
                status="failed",
                output=LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}),
                verifier_output=None,
                effective_confidence=None,
                duration_ms=int(time.time() * 1000) - start_ms,
            )

        branch_outputs: dict[str, LLMOutput] = {}
        confidences: list[float] = []
        weights: list[float] = []

        for i, (branch, raw) in enumerate(zip(group.steps, raw_results)):
            if isinstance(raw, Exception):
                msg = f"Executor error: {type(raw).__name__}: {raw}"
                logger.error("Branch '%s' raised unhandled exception: %s", branch.name, raw)
                _log_event(run_log, "error", "branch_failed",
                           f"Branch failed: {branch.name} — {msg}",
                           branch=branch.name, group=group.name)
                output = LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True)
            else:
                output = raw
                if not getattr(output, "failed", False):
                    _log_event(run_log, "info", "branch_completed",
                               f"Branch completed: {branch.name} — confidence {output.confidence:.0%}",
                               branch=branch.name, group=group.name)

            branch_outputs[branch.name] = output
            confidences.append(output.confidence)
            weights.append(branch.weight)

            await self._db_save_branch(run_id, group.name, branch, index, i, output)

        effective_confidence = self._join_confidences(group.join, confidences, weights)

        logger.info(
            "Parallel group '%s' join=%s confidences=%s effective=%.2f",
            group.name,
            group.join,
            [f"{c:.2f}" for c in confidences],
            effective_confidence,
        )
        _log_event(run_log, "info", "parallel_group_completed",
                   f"Parallel group completed: {group.name} — effective confidence "
                   f"{effective_confidence:.0%} (join: {group.join})",
                   group=group.name)

        completed = sum(1 for o in branch_outputs.values() if not getattr(o, "failed", False))
        group_output = LLMOutput(
            confidence=effective_confidence,
            summary=f"Parallel group '{group.name}': {completed}/{len(group.steps)} branches completed",
            next_step_context="",
            raw_response={},
        )

        _group_tokens = sum(sum(self._extract_usage(o.raw_response)) for o in branch_outputs.values())

        if effective_confidence < group.confidence_threshold:
            action = group.on_low_confidence
            logger.info(
                "Parallel group '%s' below threshold (%.2f < %.2f): action=%s",
                group.name, effective_confidence, group.confidence_threshold, action,
            )
            if action in ("abort", "escalate"):
                return StepResult(
                    step_name=group.name,
                    step_index=index,
                    status="aborted" if action == "abort" else "escalated",
                    output=group_output,
                    verifier_output=None,
                    effective_confidence=effective_confidence,
                    duration_ms=int(time.time() * 1000) - start_ms,
                    branch_outputs=branch_outputs,
                    total_tokens=_group_tokens,
                )

        return StepResult(
            step_name=group.name,
            step_index=index,
            status="completed",
            output=group_output,
            verifier_output=None,
            effective_confidence=effective_confidence,
            duration_ms=int(time.time() * 1000) - start_ms,
            branch_outputs=branch_outputs,
            total_tokens=_group_tokens,
        )

    # ------------------------------------------------------------------
    # Fan-out execution
    # ------------------------------------------------------------------

    async def _run_fan_out(
        self,
        fan_out: FanOutConfig,
        index: int,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        step_outputs: dict[str, LLMOutput],
        run_log: list,
    ) -> StepResult:
        with tracer.start_as_current_span(
            fan_out.name,
            attributes={
                "pork.span.kind": "fan_out",
                "pork.join_strategy": fan_out.join,
            },
        ) as span:
            result = await self._run_fan_out_impl(
                fan_out, index, pipeline, normalised, run_id, step_outputs, run_log
            )
            self._set_step_span_attributes(span, result)
            return result

    async def _run_fan_out_impl(
        self,
        fan_out: FanOutConfig,
        index: int,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        step_outputs: dict[str, LLMOutput],
        run_log: list,
    ) -> StepResult:
        import ast
        from jinja2 import Environment

        start_ms = int(time.time() * 1000)

        base_ctx = await build_context(
            pipeline, normalised, run_id, fan_out.name, step_outputs,
            artifact_store=self._artifact_store,
        )
        self._inject_pork_context(base_ctx, normalised)

        # Resolve `over` — Jinja2 render always returns a string; ast.literal_eval handles
        # Python repr of lists (e.g. "['a', 'b']"); json.loads covers valid JSON arrays.
        try:
            rendered = Environment().from_string(fan_out.over).render(**base_ctx)
            try:
                items = ast.literal_eval(rendered)
            except (ValueError, SyntaxError):
                items = json.loads(rendered)
            if not isinstance(items, list):
                raise ValueError(
                    f"'over' expression did not produce a list (got {type(items).__name__})"
                )
        except Exception as exc:
            msg = f"Fan-out '{fan_out.name}': failed to resolve 'over' expression: {exc}"
            logger.error(msg)
            _log_event(run_log, "error", "fan_out_failed", msg, step=fan_out.name)
            return StepResult(
                step_name=fan_out.name,
                step_index=index,
                status="failed",
                output=LLMOutput(
                    confidence=0.0, summary=msg, next_step_context="", raw_response={}
                ),
                verifier_output=None,
                effective_confidence=None,
                duration_ms=int(time.time() * 1000) - start_ms,
            )

        # Empty list handling
        if not items:
            if fan_out.on_empty == "abort":
                msg = f"Fan-out '{fan_out.name}': empty list — on_empty=abort"
                logger.warning(msg)
                _log_event(run_log, "warn", "fan_out_empty", msg, step=fan_out.name)
                return StepResult(
                    step_name=fan_out.name,
                    step_index=index,
                    status="failed",
                    output=LLMOutput(
                        confidence=0.0, summary=msg, next_step_context="", raw_response={}
                    ),
                    verifier_output=None,
                    effective_confidence=None,
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
            empty_msg = f"Fan-out '{fan_out.name}': empty list"
            if fan_out.on_empty == "skip":
                _log_event(run_log, "info", "fan_out_skipped",
                           f"{empty_msg} — skipped", step=fan_out.name)
            else:
                logger.warning("Fan-out '%s': empty list — treating as completed", fan_out.name)
                _log_event(run_log, "warn", "fan_out_empty", empty_msg, step=fan_out.name)
            return StepResult(
                step_name=fan_out.name,
                step_index=index,
                status="completed",
                output=LLMOutput(
                    confidence=1.0, summary=empty_msg, next_step_context="", raw_response={}
                ),
                verifier_output=None,
                effective_confidence=1.0,
                duration_ms=int(time.time() * 1000) - start_ms,
            )

        # Hard cap on item count
        if len(items) > fan_out.max_items:
            msg = (
                f"Fan-out '{fan_out.name}': {len(items)} items exceeds "
                f"max_items={fan_out.max_items}"
            )
            logger.error(msg)
            _log_event(run_log, "error", "fan_out_failed", msg, step=fan_out.name)
            return StepResult(
                step_name=fan_out.name,
                step_index=index,
                status="failed",
                output=LLMOutput(
                    confidence=0.0, summary=msg, next_step_context="", raw_response={}
                ),
                verifier_output=None,
                effective_confidence=None,
                duration_ms=int(time.time() * 1000) - start_ms,
            )

        total = len(items)
        logger.info("Fan-out '%s': %d item(s)", fan_out.name, total)
        _log_event(run_log, "info", "fan_out_started",
                   f"Fan-out started: {fan_out.name} ({total} items)", step=fan_out.name)

        branch_coros = []
        for i, item in enumerate(items):
            branch_ctx = {
                **base_ctx,
                fan_out.as_var: item,
                "fan_out_index": i,
                "fan_out_total": total,
            }
            branch_step = ParallelStepConfig(
                name=f"{fan_out.name}/{i}",
                executor=fan_out.executor,
                executor_config=fan_out.executor_config,
                prompt_template=fan_out.prompt_template,
                timeout_seconds=fan_out.timeout_seconds,
                verifier=fan_out.verifier,
            )
            branch_coros.append(
                self._run_parallel_branch(branch_step, branch_ctx, run_log, run_id)
            )

        raw_results = await asyncio.gather(*branch_coros, return_exceptions=True)

        branch_outputs: dict[str, LLMOutput] = {}
        confidences: list[float] = []
        weights: list[float] = []

        for i, (item, raw) in enumerate(zip(items, raw_results)):
            branch_name = f"{fan_out.name}/{i}"
            if isinstance(raw, Exception):
                msg = f"Executor error: {type(raw).__name__}: {raw}"
                output = LLMOutput(
                    confidence=0.0, summary=msg, next_step_context="", raw_response={},
                    failed=True,
                )
                _log_event(run_log, "error", "fan_out_branch_failed",
                           f"Fan-out branch failed: {branch_name} — {msg}",
                           step=fan_out.name, branch=branch_name)
            else:
                output = raw
                if not getattr(output, "failed", False):
                    _log_event(run_log, "info", "fan_out_branch_completed",
                               f"Fan-out branch completed: {branch_name} — "
                               f"confidence {output.confidence:.0%}",
                               step=fan_out.name, branch=branch_name)

            branch_outputs[branch_name] = output
            confidences.append(output.confidence)
            weights.append(1.0)

            # DB branch name is str(i) so stored step_name = "{fan_out.name}/{i}"
            db_branch = ParallelStepConfig(
                name=str(i),
                executor=fan_out.executor,
                executor_config=fan_out.executor_config,
                prompt_template=fan_out.prompt_template,
            )
            await self._db_save_branch(run_id, fan_out.name, db_branch, index, i, output)

        effective_confidence = self._join_confidences(fan_out.join, confidences, weights)
        logger.info(
            "Fan-out '%s' join=%s confidences=%s effective=%.2f",
            fan_out.name,
            fan_out.join,
            [f"{c:.2f}" for c in confidences],
            effective_confidence,
        )
        _log_event(run_log, "info", "fan_out_completed",
                   f"Fan-out completed: {fan_out.name} — {total} branches, "
                   f"effective confidence {effective_confidence:.0%}",
                   step=fan_out.name)

        completed = sum(1 for o in branch_outputs.values() if not getattr(o, "failed", False))
        group_output = LLMOutput(
            confidence=effective_confidence,
            summary=f"Fan-out '{fan_out.name}': {completed}/{total} branches completed",
            next_step_context="",
            raw_response={},
        )

        _fan_out_tokens = sum(sum(self._extract_usage(o.raw_response)) for o in branch_outputs.values())

        if effective_confidence < fan_out.confidence_threshold:
            action = fan_out.on_low_confidence
            logger.info(
                "Fan-out '%s' below threshold (%.2f < %.2f): action=%s",
                fan_out.name, effective_confidence, fan_out.confidence_threshold, action,
            )
            if action in ("abort", "escalate"):
                return StepResult(
                    step_name=fan_out.name,
                    step_index=index,
                    status="aborted" if action == "abort" else "escalated",
                    output=group_output,
                    verifier_output=None,
                    effective_confidence=effective_confidence,
                    duration_ms=int(time.time() * 1000) - start_ms,
                    branch_outputs=branch_outputs,
                    total_tokens=_fan_out_tokens,
                )

        return StepResult(
            step_name=fan_out.name,
            step_index=index,
            status="completed",
            output=group_output,
            verifier_output=None,
            effective_confidence=effective_confidence,
            duration_ms=int(time.time() * 1000) - start_ms,
            branch_outputs=branch_outputs,
            total_tokens=_fan_out_tokens,
        )

    async def _run_parallel_branch(
        self,
        branch: ParallelStepConfig,
        ctx: dict,
        run_log: list,
        run_id: str,
    ) -> LLMOutput:
        agent = branch.executor_config.get("agent", "")
        with tracer.start_as_current_span(
            branch.name,
            attributes={
                "pork.span.kind": "branch",
                "pork.executor": branch.executor,
                "pork.agent": agent,
            },
        ) as span:
            output = await self._run_parallel_branch_impl(branch, ctx, run_log, run_id)
            span.set_attribute("pork.confidence", output.confidence)
            if output.model:
                span.set_attribute("pork.model", output.model)
            if getattr(output, "failed", False):
                span.set_status(Status(StatusCode.ERROR))
            return output

    async def _run_parallel_branch_impl(
        self,
        branch: ParallelStepConfig,
        ctx: dict,
        run_log: list,
        run_id: str,
    ) -> LLMOutput:
        # Synthesise a StepConfig so we can reuse executor dispatch and verifier logic.
        # confidence_threshold=0.0 because gating happens at the group level, not per-branch.
        branch_step = StepConfig(
            name=branch.name,
            executor=branch.executor,
            executor_config=branch.executor_config,
            confidence_threshold=0.0,
            prompt_template=branch.prompt_template,
            timeout_seconds=branch.timeout_seconds,
            verifier=branch.verifier,
        )

        try:
            executor = self._get_executor(branch.executor)
            coro = executor.execute(branch_step, ctx)
            if branch.timeout_seconds:
                output = await asyncio.wait_for(coro, timeout=branch.timeout_seconds)
            else:
                output = await coro
            if self._artifact_store:
                output = await self._intercept_artifacts(output, run_id, branch.name)
        except asyncio.TimeoutError:
            msg = f"Branch timed out after {branch.timeout_seconds}s"
            logger.error("Branch '%s' %s", branch.name, msg)
            return LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True)
        except Exception as exc:
            msg = f"Executor error: {type(exc).__name__}: {exc}"
            logger.error("Branch '%s' %s", branch.name, msg)
            return LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True)

        if branch.verifier and self._should_verify(branch.verifier, output.confidence):
            verifier_output = await self._run_verifier(branch_step, ctx, output, run_log)
            if verifier_output:
                adjusted = self._combine_confidence(
                    branch_step, output.confidence, verifier_output.confidence
                )
                output = output.model_copy(update={"confidence": adjusted})
                logger.info(
                    "Branch '%s' verifier: primary=%.2f verifier=%.2f effective=%.2f",
                    branch.name, output.confidence, verifier_output.confidence, adjusted,
                )
                _log_event(run_log, "info", "verifier_ran",
                           f"Verifier ran: {branch.name} — primary {output.confidence:.0%} / "
                           f"verifier {verifier_output.confidence:.0%} → effective {adjusted:.0%}",
                           branch=branch.name)

        logger.debug(
            "Branch '%s' confidence=%.2f summary=%s",
            branch.name, output.confidence, output.summary,
        )
        return output

    def _join_confidences(
        self,
        strategy: str,
        confidences: list[float],
        weights: list[float],
    ) -> float:
        if not confidences:
            return 0.0
        if strategy == "all_must_pass":
            return min(confidences)
        if strategy == "any_must_pass":
            return max(confidences)
        # weighted_average
        total = sum(weights)
        if total == 0:
            return 0.0
        return sum(c * w for c, w in zip(confidences, weights)) / total

    # ------------------------------------------------------------------
    # Conditional step evaluation
    # ------------------------------------------------------------------

    def _eval_when(self, condition: str, ctx: dict) -> bool:
        """Evaluate a when: expression using Jinja2 so dot-notation on dicts works.

        Renders {% if <condition> %}true{% else %}false{% endif %} against the
        step context. Any evaluation error is treated as false with a warning.
        """
        from jinja2 import Environment
        try:
            tmpl = Environment().from_string(
                "{%- if " + condition + " -%}true{%- else -%}false{%- endif -%}"
            )
            return tmpl.render(**ctx) == "true"
        except Exception as exc:
            logger.warning(
                "when: '%s' could not be evaluated (%s) — step skipped", condition, exc
            )
            return False

    # ------------------------------------------------------------------
    # Verifier logic (shared by sequential steps and parallel branches)
    # ------------------------------------------------------------------

    def _should_verify(self, verifier: VerifierConfig, primary_confidence: float) -> bool:
        if verifier.trigger.always:
            return True
        return (
            primary_confidence < verifier.trigger.confidence_below
            and primary_confidence > verifier.trigger.confidence_above
        )

    async def _run_verifier(
        self,
        step: StepConfig,
        ctx: dict,
        primary_output: LLMOutput,
        run_log: list,
    ) -> LLMOutput | None:
        verifier = step.verifier
        assert verifier is not None
        span_name = f"{step.name}:challenger" if verifier.mode == "challenger" else f"{step.name}:verifier"
        with tracer.start_as_current_span(
            span_name,
            attributes={"pork.span.kind": "verifier", "pork.verifier.mode": verifier.mode},
        ) as span:
            output = await self._run_verifier_impl(step, ctx, primary_output, run_log)
            if output is not None:
                span.set_attribute("pork.confidence", output.confidence)
            else:
                span.set_status(Status(StatusCode.ERROR))
            return output

    async def _run_verifier_impl(
        self,
        step: StepConfig,
        ctx: dict,
        primary_output: LLMOutput,
        run_log: list,
    ) -> LLMOutput | None:
        verifier = step.verifier
        assert verifier is not None

        from jinja2 import Environment

        if verifier.mode == "challenger":
            # Challenger: run the same task independently — no primary output shared.
            # Both agents tackle the problem blind; combination strategy reconciles the scores.
            verifier_step = StepConfig(
                name=f"{step.name}:challenger",
                executor=verifier.executor,
                executor_config=verifier.executor_config,
                confidence_threshold=0.0,
                prompt_template=step.prompt_template,
            )
            verifier_ctx = ctx
        else:
            # Reviewer (default): share primary output so verifier can critique the reasoning.
            verifier_step = StepConfig(
                name=f"{step.name}:verifier",
                executor=verifier.executor,
                executor_config=verifier.executor_config,
                confidence_threshold=0.0,
                prompt_template=_VERIFIER_PROMPT_TEMPLATE,
            )
            primary_prompt = Environment().from_string(step.prompt_template).render(**ctx)
            verifier_ctx = {
                **ctx,
                "primary_prompt": primary_prompt,
                "primary_response": json.dumps(
                    primary_output.model_dump(exclude={"raw_response"}), indent=2
                ),
            }

        try:
            executor = self._get_executor(verifier.executor)
            return await executor.execute(verifier_step, verifier_ctx)
        except Exception as exc:
            logger.warning(
                "Verifier for step '%s' failed: %s — using primary confidence only",
                step.name, exc,
            )
            _log_event(run_log, "warn", "verifier_failed",
                       f"Verifier failed for {step.name}: {exc}", step=step.name)
            return None

    def _combine_confidence(
        self,
        step: StepConfig,
        primary_confidence: float,
        verifier_confidence: float,
    ) -> float:
        verifier = step.verifier
        assert verifier is not None

        if verifier.combination_strategy == "minimum":
            return min(primary_confidence, verifier_confidence)

        if verifier_confidence < verifier.veto_floor:
            logger.info(
                "Verifier veto triggered for step '%s': verifier_confidence=%.2f < veto_floor=%.2f",
                step.name, verifier_confidence, verifier.veto_floor,
            )
            return verifier_confidence

        return primary_confidence

    # ------------------------------------------------------------------
    # Artifact interception
    # ------------------------------------------------------------------

    async def _intercept_artifacts(
        self, output: LLMOutput, run_id: str, step_name: str
    ) -> LLMOutput:
        """Write any artifacts in the output to the store, replacing content with references."""
        artifacts_raw = output.model_extra.get("artifacts") if output.model_extra else None
        if not isinstance(artifacts_raw, dict) or not artifacts_raw:
            return output
        refs: dict[str, str] = {}
        for key, content in artifacts_raw.items():
            if not isinstance(content, str):
                continue
            try:
                ref = await self._artifact_store.write(run_id, step_name, key, content)
                refs[key] = ref
                logger.info(
                    "Artifact written: run=%s step=%s key=%s (%d chars)",
                    run_id, step_name, key, len(content),
                )
            except Exception as exc:
                logger.error("Failed to write artifact %s/%s: %s", step_name, key, exc)
        return output.model_copy(update={"artifacts": refs})

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    async def _db_create_run(
        self,
        run_id: str,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        parent_run_id: str | None = None,
    ) -> None:
        if not self._session_factory:
            return
        async with self._session_factory() as session:
            session.add(PipelineRun(
                id=run_id,
                pipeline_name=pipeline.name,
                source=normalised.source,
                triggered_at=normalised.received_at,
                status="running",
                normalised_context=normalised.model_dump_json(),
                raw_payload=json.dumps(normalised.raw),
                fingerprint=normalised.fingerprint,
                parent_run_id=parent_run_id,
            ))
            await session.commit()

    async def _db_save_step(
        self,
        run_id: str,
        step: StepConfig,
        result: StepResult,
    ) -> None:
        if not self._session_factory:
            return
        async with self._session_factory() as session:
            _agent = step.executor_config.get("agent")
            _artifact_refs = None
            if result.output and result.output.model_extra:
                _ar = result.output.model_extra.get("artifacts")
                if isinstance(_ar, dict) and _ar:
                    _artifact_refs = json.dumps(_ar)
            _trace = None
            if result.output:
                _t = (result.output.raw_response or {}).get("trace")
                if _t is not None:
                    _trace = json.dumps(_t)
            _in_tok, _out_tok = self._extract_usage(result.output.raw_response if result.output else {})
            session.add(PipelineStep(
                run_id=run_id,
                step_name=result.step_name,
                step_index=result.step_index,
                executor=step.executor,
                agent=f"{step.executor}:{_agent}" if _agent else None,
                model=result.output.model if result.output else None,
                prompt=json.dumps(step.executor_config),
                raw_output=json.dumps(result.output.raw_response) if result.output else None,
                parsed_output=result.output.model_dump_json(exclude={"raw_response"}) if result.output else None,
                verifier_output=result.verifier_output.model_dump_json(exclude={"raw_response"}) if result.verifier_output else None,
                verifier_mode=step.verifier.mode if step.verifier and result.verifier_output else None,
                status=result.status,
                primary_confidence=result.output.confidence if result.output else None,
                verifier_confidence=result.verifier_output.confidence if result.verifier_output else None,
                effective_confidence=result.effective_confidence,
                duration_ms=result.duration_ms,
                executed_at=utc_now(),
                artifacts=_artifact_refs,
                agent_trace=_trace,
                input_tokens=_in_tok or None,
                output_tokens=_out_tok or None,
            ))
            await session.commit()

    async def _db_save_branch(
        self,
        run_id: str,
        group_name: str,
        branch: ParallelStepConfig,
        group_index: int,
        branch_index: int,
        output: LLMOutput,
    ) -> None:
        if not self._session_factory:
            return
        branch_failed = getattr(output, "failed", False)
        _agent = branch.executor_config.get("agent")
        _artifact_refs = None
        if not branch_failed and output.model_extra:
            _ar = output.model_extra.get("artifacts")
            if isinstance(_ar, dict) and _ar:
                _artifact_refs = json.dumps(_ar)
        _trace = None
        _t = (output.raw_response or {}).get("trace")
        if _t is not None:
            _trace = json.dumps(_t)
        _in_tok, _out_tok = self._extract_usage(output.raw_response)
        async with self._session_factory() as session:
            session.add(PipelineStep(
                run_id=run_id,
                step_name=f"{group_name}/{branch.name}",
                # Encode group position + branch position so DB ordering mirrors execution order.
                # Branches share the same group_index prefix and sort together.
                step_index=group_index * 1000 + branch_index,
                executor=branch.executor,
                agent=f"{branch.executor}:{_agent}" if _agent else None,
                model=output.model,
                prompt=json.dumps(branch.executor_config),
                raw_output=json.dumps(output.raw_response),
                parsed_output=output.model_dump_json(exclude={"raw_response"}),
                verifier_output=None,
                status="failed" if branch_failed else "completed",
                primary_confidence=None if branch_failed else output.confidence,
                verifier_confidence=None,
                effective_confidence=None if branch_failed else output.confidence,
                duration_ms=None,
                executed_at=utc_now(),
                artifacts=_artifact_refs,
                agent_trace=_trace,
                input_tokens=_in_tok or None,
                output_tokens=_out_tok or None,
            ))
            await session.commit()

    async def _db_complete_run(self, run_id: str, status: str, run_log: list) -> None:
        if not self._session_factory:
            return
        from sqlalchemy import select
        async with self._session_factory() as session:
            run = await session.get(PipelineRun, run_id)
            if run:
                run.status = status
                run.completed_at = utc_now()
                run.logs = json.dumps(run_log)
                await session.commit()

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _build_escalation_context(
        self,
        confidence_threshold: float,
        step_result: StepResult,
    ) -> dict:
        output = step_result.output
        reasoning = output.reasoning or {} if output else {}

        if step_result.effective_confidence is not None:
            if step_result.effective_confidence < confidence_threshold:
                escalation_reason = "low_confidence"
            else:
                escalation_reason = step_result.status
        else:
            escalation_reason = step_result.status

        return {
            "escalation_reason": escalation_reason,
            "confidence": step_result.effective_confidence,
            "confidence_threshold": confidence_threshold,
            "contradicts": reasoning.get("contradicts", ""),
            "step_summary": output.summary if output else "",
        }

    async def _dispatch_notification(
        self,
        pipeline: PipelineConfig,
        action: str,
        context: dict,
        run_log: list,
    ) -> None:
        notifications = pipeline.notifications.get(action)
        if not notifications:
            logger.debug("No notification config for action '%s' — skipping", action)
            return

        for notification in notifications:
            notifier = self._notifiers.get(notification.channel)
            if not notifier:
                logger.warning(
                    "No notifier registered for channel '%s' — skipping notification for action '%s'",
                    notification.channel, action,
                )
                continue
            await notifier.send(notification, context)
            _log_event(run_log, "info", "notification_sent",
                       f"Notification sent: {action} → {notification.channel}",
                       action=action, channel=notification.channel)

    # ------------------------------------------------------------------
    # Executor cache
    # ------------------------------------------------------------------

    def _get_executor(self, name: str) -> BaseExecutor:
        if name not in self._executor_instances:
            executor_class = self._executor_classes.get(name)
            if not executor_class:
                raise ValueError(
                    f"Unknown executor '{name}'. "
                    f"Registered: {list(self._executor_classes.keys())}"
                )
            self._executor_instances[name] = executor_class()
        return self._executor_instances[name]
