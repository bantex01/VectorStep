import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

import httpx
from jinja2 import Environment, Undefined
from opentelemetry.trace import Status, StatusCode
from sqlalchemy.exc import IntegrityError
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
    HumanCheckConfig,
    LoopConfig,
    ParallelGroupConfig,
    ParallelGroupInner,
    ParallelStepConfig,
    PipelineConfig,
    RetryConfig,
    ShellCheckConfig,
    StepConfig,
    VerifierConfig,
    WebhookCheckConfig,
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


# Built-in prompt for the shadow-mode grounding pass. Rendered with:
#   {{primary_response}} — the primary agent's JSON output (no raw_response)
#   {{agent_trace}}      — a formatted transcript of the primary's tool calls + results
_GROUNDING_PROMPT_TEMPLATE = """\
You are a grounding auditor. You are shown another agent's structured output and the \
execution trace it produced (its tool calls and the results those tools returned). Your \
ONLY job is to check whether the agent's load-bearing claims are supported by evidence \
that actually appears in the trace. You cannot add outside knowledge, you cannot browse, \
and you are NOT assessing whether the conclusion is correct — only whether it is anchored \
to evidence that the trace actually returned.

A "load-bearing claim" is an assertion the output depends on: a stated root cause, a \
metric value, a causal link ("X because Y"), a referenced ticket/dashboard/id. Ignore \
hedging, restatements of the task, and generic advice.

For each load-bearing claim, decide if a tool result in the trace supports it. A claim \
reached with zero supporting tool results — or whose supporting tool call errored — is \
NOT supported.

Primary agent's output:
---
{{primary_response}}
---

Execution trace:
---
{{agent_trace}}
---

Return JSON only, no other text:
{
  "confidence": 0.0,
  "summary": "One sentence: how many load-bearing claims are supported, and which key one is not",
  "next_step_context": "",
  "reasoning": {
    "claims": [
      {"claim": "the exact claim", "supported": true, "evidence": "which tool result supports it, or why it is unsupported"}
    ]
  }
}

Set "confidence" to the FRACTION of load-bearing claims that are supported (supported \
count / total load-bearing claims), as a number from 0.0 to 1.0. This number is the \
grounding score — it is about the evidence, not about how sure you feel.
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
    grounding_score: float | None = None       # G ∈ [0,1], or None when not computed
    trust_report: dict | None = None           # full TrustReport (§5e), or None
    deterministic_passed: bool | None = None   # None = no checks declared; else all-checks-passed


@dataclass
class PipelineRunResult:
    run_id: str
    pipeline_name: str
    status: Literal["completed", "stopped", "aborted", "escalated", "failed", "deduplicated"]
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
        calibration_n_min: int = 20,
        calibration_bin_width: float = 0.1,
        calibration_cache_ttl_seconds: int = 300,
    ):
        self._executor_classes = executors
        self._executor_instances: dict[str, BaseExecutor] = {}
        self._session_factory = session_factory
        self._notifiers: dict[str, TelegramNotifier] = notifiers or {}
        self._artifact_store = artifact_store
        self._pipeline_registry = pipeline_registry

        from .calibration import CalibrationCache
        self._calibration_cache: "CalibrationCache | None" = (
            CalibrationCache(
                session_factory, bin_width=calibration_bin_width,
                n_min=calibration_n_min, ttl_seconds=calibration_cache_ttl_seconds,
            )
            if session_factory is not None else None
        )

    def set_pipeline_registry(self, registry: "dict[str, PipelineConfig]") -> None:
        self._pipeline_registry = registry

    @staticmethod
    def _extract_usage(raw_response: dict) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) from a gateway raw_response, or (0, 0)."""
        usage = ((raw_response or {}).get("meta") or {}).get("agentMeta", {}).get("usage") or {}
        return (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))

    @staticmethod
    def _format_trace_for_grounding(trace: list, max_chars: int = 1500) -> str:
        """Render tool_call/tool_result/text events as a readable transcript for the
        grounding judge. Returns '' when the trace has no evidence-bearing events."""
        lines: list[str] = []
        for ev in trace or []:
            if not isinstance(ev, dict):
                continue
            t = ev.get("type")
            if t == "tool_call":
                args = json.dumps(ev.get("input") or {})
                if len(args) > 300:
                    args = args[:300] + "…"
                lines.append(f"TOOL CALL: {ev.get('name', '')}({args})")
            elif t == "tool_result":
                c = str(ev.get("content", ""))
                if len(c) > max_chars:
                    c = c[:max_chars] + "…"
                err = " [ERROR]" if ev.get("is_error") else ""
                lines.append(f"TOOL RESULT{err} ({ev.get('name', '')}): {c}")
            elif t == "text":
                c = str(ev.get("content", ""))
                if c.strip():
                    lines.append(f"AGENT TEXT: {c[:max_chars]}")
        return "\n".join(lines)

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

        _span_name = (
            f"pipeline.run: {normalised.team}/{pipeline.name}"
            if normalised.team
            else f"pipeline.run: {pipeline.name}"
        )
        with start_root_span(
            _span_name,
            attributes={
                "pork.pipeline.name": pipeline.name,
                "pork.run.id": run_id,
                "pork.source": normalised.source,
                **({"pork.team": normalised.team} if normalised.team else {}),
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

        created = await self._db_create_run(run_id, pipeline, normalised, parent_run_id=parent_run_id)
        if not created:
            _log_event(run_log, "warn", "run_deduplicated",
                       "Duplicate run rejected at insert time — another run is already "
                       "in flight for this pipeline+fingerprint.")
            run_events.publish_complete(run_id, "deduplicated")
            return PipelineRunResult(
                run_id=run_id,
                pipeline_name=pipeline.name,
                status="deduplicated",
            )

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
                # on_failure hook applies only to sequential StepConfig executor errors
                if isinstance(step, StepConfig) and step_result.status == "failed":
                    if step.on_failure.webhook:
                        await self._fire_step_failure_webhook(
                            step, step_result, pipeline, normalised, run_id, step_outputs, run_log
                        )
                    if step.on_failure.policy == "continue":
                        _log_event(run_log, "warn", "step_failed_continuing",
                                   f"Step failed (continuing): {step_name} — "
                                   f"{step_result.output.summary if step_result.output else ''}",
                                   step=step_name)
                        accumulated_tokens += step_result.total_tokens
                        continue

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
            if result.output.provider:
                span.set_attribute("pork.provider", result.output.provider)
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

        grounding_score: float | None = None
        grounding_report: dict | None = None
        grounding_tokens = 0
        deterministic_results: list[dict] | None = None
        deterministic_passed: bool | None = None
        trust_report: dict | None = None
        combined_trust = effective_confidence
        gate_policy = "legacy_confidence"
        calibration_report: dict | None = None

        if step.calibration is not None and step.calibration.enforce:
            gate_policy = "trust_vector"
            agent_key = (
                f"{step.executor}:{step.executor_config.get('agent')}"
                if step.executor_config.get("agent") else None
            )
            bucket = None
            if self._calibration_cache is not None:
                bucket = await self._calibration_cache.get(
                    step_name=step.name, agent=agent_key,
                    model=primary_output.model, provider=primary_output.provider,
                )
            calib_bin = bucket.lookup(combined_trust) if bucket is not None else None

            if calib_bin is not None and calib_bin.validated:
                calibration_report = {
                    "bucket": {"step_name": step.name, "agent": agent_key,
                               "model": primary_output.model, "provider": primary_output.provider},
                    "bin": {"lo": calib_bin.lo, "hi": calib_bin.hi},
                    "n": calib_bin.n,
                    "n_min": self._calibration_cache.n_min if self._calibration_cache else None,
                    "validated": True,
                    "raw": combined_trust,
                    "calibrated": calib_bin.mean_label,
                    "on_uncalibrated": step.calibration.on_uncalibrated,
                }
                combined_trust = calib_bin.mean_label
            else:
                calibration_report = {
                    "bucket": {"step_name": step.name, "agent": agent_key,
                               "model": primary_output.model, "provider": primary_output.provider},
                    "bin": {"lo": calib_bin.lo, "hi": calib_bin.hi} if calib_bin is not None else None,
                    "n": calib_bin.n if calib_bin is not None else 0,
                    "n_min": self._calibration_cache.n_min if self._calibration_cache else None,
                    "validated": False,
                    "raw": combined_trust,
                    "calibrated": None,
                    "on_uncalibrated": step.calibration.on_uncalibrated,
                }
                if step.calibration.on_uncalibrated == "escalate":
                    combined_trust = 0.0

        if step.grounding is not None:
            grounding_score, grounding_report, grounding_tokens = await self._run_grounding(
                step=step, ctx=ctx, primary_output=primary_output, run_log=run_log,
            )
            if step.grounding.enforce and grounding_score is not None:
                combined_trust = min(combined_trust, grounding_score)
                gate_policy = "trust_vector"

        if step.deterministic_checks:
            deterministic_results = await self._run_deterministic_checks(
                step=step, ctx=ctx, run_log=run_log,
            )
            deterministic_passed = all(r["passed"] for r in deterministic_results)
            gate_policy = "trust_vector"
            if not deterministic_passed:
                combined_trust = 0.0

        if (
            step.grounding is not None
            or step.deterministic_checks
            or (step.calibration is not None and step.calibration.enforce)
        ):
            trust_report = self._build_trust_report(
                primary_confidence=primary_output.confidence,
                effective_confidence=effective_confidence,
                verifier_confidence=verifier_output.confidence if verifier_output else None,
                verifier_mode=step.verifier.mode if step.verifier and verifier_output else None,
                grounding_score=grounding_score,
                grounding_report=grounding_report,
                deterministic_results=deterministic_results,
                calibration_report=calibration_report,
                combined_trust=combined_trust,
                gate_policy=gate_policy,
            )

        _in_tok, _out_tok = self._extract_usage(primary_output.raw_response)
        if verifier_output:
            _vi, _vo = self._extract_usage(verifier_output.raw_response)
            _in_tok += _vi
            _out_tok += _vo
        _step_tokens = _in_tok + _out_tok + grounding_tokens

        if combined_trust < step.confidence_threshold:
            action = step.on_low_confidence
            logger.info(
                "Step '%s' below confidence threshold (trust %.2f < %.2f; self-report was %.2f): action=%s",
                step.name, combined_trust, step.confidence_threshold, effective_confidence, action,
            )
            conf_msg = (f"trust {combined_trust:.0%} < threshold {step.confidence_threshold:.0%}"
                        + (f" (self-report was {effective_confidence:.0%})"
                           if combined_trust != effective_confidence else ""))
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
                    grounding_score=grounding_score,
                    trust_report=trust_report,
                    deterministic_passed=deterministic_passed,
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
                    grounding_score=grounding_score,
                    trust_report=trust_report,
                    deterministic_passed=deterministic_passed,
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
                grounding_score=grounding_score,
                trust_report=trust_report,
                deterministic_passed=deterministic_passed,
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
            grounding_score=grounding_score,
            trust_report=trust_report,
            deterministic_passed=deterministic_passed,
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
        # TODO(grounding phase): branches — grounding is not yet wired into fan-out/
        # parallel branches, only sequential StepConfig steps (see SPEC-grounding-shadow.md).
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
            if output.provider:
                span.set_attribute("pork.provider", output.provider)
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
        span_name = f"{step.name}:independent" if verifier.mode == "independent" else f"{step.name}:verifier"
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

        if verifier.mode == "independent":
            # Independent: run the same task blind — no primary output shared.
            # Both agents tackle the problem blind; combination strategy reconciles the scores.
            verifier_step = StepConfig(
                name=f"{step.name}:independent",
                executor=verifier.executor,
                executor_config=verifier.executor_config,
                confidence_threshold=0.0,
                prompt_template=step.prompt_template,
            )
            verifier_ctx = ctx
        else:
            # Critic (default): share primary output so verifier can critique the reasoning.
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
            # min(), not verifier_confidence alone — the veto score is only guaranteed
            # to be the lower of the two when primary is at/above veto_floor. If primary
            # is already below veto_floor too, returning verifier_confidence unconditionally
            # could raise trust above primary, violating the downward-only invariant.
            return min(primary_confidence, verifier_confidence)

        return primary_confidence

    async def _run_grounding(
        self,
        step: StepConfig,
        ctx: dict,
        primary_output: LLMOutput,
        run_log: list,
    ) -> tuple[float | None, dict, int]:
        """Shadow-mode grounding pass. Returns (G_or_None, grounding_report, tokens).
        Never raises — grounding must not break a run."""
        grounding = step.grounding
        assert grounding is not None

        trace = (primary_output.raw_response or {}).get("trace") or []
        transcript = self._format_trace_for_grounding(trace)
        if not transcript:
            # No evidence trail (non-gateway step, or a trace with no tool activity):
            # "nothing to check" is null, not zero.
            return None, {"computed": False, "reason": "no_trace", "agent": grounding.agent}, 0

        grounding_ctx = {
            **ctx,
            # `artifacts` is presentation content (e.g. a full markdown report), not a
            # claim itself — the load-bearing claims live in the structured fields
            # (summary, reasoning, and whatever extra fields the agent returns
            # alongside them, e.g. patterns_found). Excluding it keeps the grounding
            # call cheap and stops the judge quoting a large blob back at us.
            "primary_response": json.dumps(
                primary_output.model_dump(exclude={"raw_response", "artifacts"}), indent=2
            ),
            "agent_trace": transcript,
        }
        grounding_step = StepConfig(
            name=f"{step.name}:grounding",
            executor=grounding.executor,
            executor_config={"agent": grounding.agent, **grounding.executor_config},
            confidence_threshold=0.0,
            prompt_template=_GROUNDING_PROMPT_TEMPLATE,
            timeout_seconds=grounding.timeout_seconds,
        )

        with tracer.start_as_current_span(
            f"{step.name}:grounding",
            attributes={"pork.span.kind": "grounding", "pork.agent": grounding.agent},
        ) as span:
            try:
                executor = self._get_executor(grounding.executor)
                coro = executor.execute(grounding_step, grounding_ctx)
                out = await asyncio.wait_for(coro, timeout=grounding.timeout_seconds)
            except Exception as exc:
                logger.warning(
                    "Grounding pass for step '%s' failed: %s — recording G=null", step.name, exc
                )
                _log_event(run_log, "warn", "grounding_failed",
                           f"Grounding failed for {step.name}: {exc}", step=step.name)
                span.set_status(Status(StatusCode.ERROR))
                return None, {"computed": False, "reason": "error", "error": str(exc),
                              "agent": grounding.agent}, 0

            g = max(0.0, min(1.0, float(out.confidence)))
            span.set_attribute("pork.grounding.score", g)
            claims = (out.reasoning or {}).get("claims") if out.reasoning else None
            _gi, _go = self._extract_usage(out.raw_response)
            _log_event(run_log, "info", "grounding_ran",
                       f"Grounding (shadow): {step.name} — G {g:.0%} vs self-report "
                       f"{primary_output.confidence:.0%}", step=step.name)
            report = {
                "computed": True,
                "agent": grounding.agent,
                "score": g,
                "summary": out.summary,
                "claims": claims if isinstance(claims, list) else [],
            }
            return g, report, _gi + _go

    async def _run_deterministic_checks(
        self,
        step: StepConfig,
        ctx: dict,
        run_log: list,
    ) -> list[dict]:
        """Evaluate every declared deterministic check. Fail-closed: an execution
        error, timeout, or evaluation exception counts as passed=False, never as
        skipped — see §2 of SPEC-hard-gates.md for why this differs from grounding's
        soft-fail philosophy."""
        results: list[dict] = []
        for check in step.deterministic_checks:
            start = time.time()
            try:
                if check.type == "shell":
                    passed, detail = await self._eval_shell_check(check, ctx)
                elif check.type == "webhook":
                    passed, detail = await self._eval_webhook_check(check, ctx)
                else:
                    passed, detail = await self._eval_human_check(check, ctx)
            except Exception as exc:
                passed, detail = False, f"check errored: {exc}"
            duration_ms = int((time.time() - start) * 1000)
            results.append({
                "name": check.name, "type": check.type, "passed": passed,
                "detail": detail, "duration_ms": duration_ms,
            })
            _log_event(
                run_log, "info" if passed else "warn", "deterministic_check_ran",
                f"Deterministic check {'passed' if passed else 'FAILED'}: "
                f"{check.name} ({check.type}) — {detail}",
                step=step.name,
            )
        return results

    async def _eval_shell_check(self, check: ShellCheckConfig, ctx: dict) -> tuple[bool, str]:
        proc = await asyncio.create_subprocess_shell(
            check.run,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=check.timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"timed out after {check.timeout_seconds}s"

        result = stdout.decode(errors="replace").strip()
        exit_code = proc.returncode
        if exit_code != 0:
            err = stderr.decode(errors="replace").strip()
            return False, f"exit code {exit_code}: {(err or result)[:300]}"

        passed = self._eval_when(check.expect, {**ctx, "result": result, "exit_code": exit_code})
        return passed, f"exit 0, result={result[:200]!r}"

    async def _eval_webhook_check(self, check: WebhookCheckConfig, ctx: dict) -> tuple[bool, str]:
        env = Environment(undefined=Undefined)

        def _render(obj: object) -> object:
            if isinstance(obj, str):
                return env.from_string(obj).render(**ctx)
            if isinstance(obj, dict):
                return {k: _render(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_render(item) for item in obj]
            return obj

        url = self._resolve_env(env.from_string(check.url).render(**ctx))
        headers = {k: self._resolve_env(v) for k, v in check.headers.items()}
        payload = _render(check.payload)

        async with httpx.AsyncClient(timeout=check.timeout_seconds) as client:
            resp = await client.request(check.method.upper(), url, json=payload, headers=headers)

        try:
            body = resp.json()
        except Exception:
            body = resp.text

        response = {"status_code": resp.status_code, "body": body}
        passed = self._eval_when(check.expect, {**ctx, "response": response})
        return passed, f"HTTP {resp.status_code}"

    async def _eval_human_check(self, check: HumanCheckConfig, ctx: dict) -> tuple[bool, str]:
        from ..executors.human import request_decision

        message = Environment(undefined=Undefined).from_string(check.message).render(**ctx)
        decision, _token = await request_decision(
            message=message,
            step_name=check.name,
            pipeline_name=ctx.get("pipeline_name"),
            run_id=ctx.get("pipeline_run_id"),
            team=ctx.get("team"),
            testing=ctx.get("_testing", False),
            timeout=check.timeout_seconds,
        )
        if decision is None:
            return False, "timed out awaiting human decision"
        return decision, "approved by human" if decision else "rejected by human"

    @staticmethod
    def _build_trust_report(
        primary_confidence: float,
        effective_confidence: float,
        verifier_confidence: float | None,
        verifier_mode: str | None,
        grounding_score: float | None,
        grounding_report: dict | None,
        deterministic_results: list[dict] | None,
        calibration_report: dict | None,
        combined_trust: float,
        gate_policy: str,
    ) -> dict:
        deterministic_passed = (
            all(r["passed"] for r in deterministic_results) if deterministic_results else None
        )
        return {
            "version": 3,   # bumped from 2 — calibration is new
            "mode": "enforced" if gate_policy == "trust_vector" else "shadow",
            "signals": {
                "S": primary_confidence,
                "S_after_V": effective_confidence,
                "V": verifier_confidence,
                "V_mode": verifier_mode,   # NEW — "critic" | "independent" | null (no verifier ran)
                "G": grounding_score,
                "C": None,                      # consistency — still a later phase
                "D": deterministic_passed,       # NEW — bool, or null if no checks declared
            },
            "combined_trust": combined_trust,    # NEW — what the gate actually compared
            "grounding": grounding_report,
            "deterministic_checks": deterministic_results,   # NEW — full per-check detail, or null
            "calibration": calibration_report,   # NEW — see calibration.py, or null if not enforced
            "gate": {"policy": gate_policy},      # "legacy_confidence" | "trust_vector"
        }

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
    ) -> bool:
        """Insert the run row. Returns False if a 'running' row for the same
        pipeline+fingerprint already exists (the DB's unique partial index — see
        database.py — rejects the insert), True otherwise (including when no DB is
        configured, e.g. unit tests with session_factory=None).

        This closes the dedup TOCTOU race documented in README §3a: the pre-check in
        main.py's webhook handler narrows the window, but two requests can still both
        pass it before either's row is committed. The DB constraint is the actual
        guarantee; this is the last line of defence so a race never results in two
        pipelines executing concurrently for the same alert.
        """
        if not self._session_factory:
            return True
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
                team=normalised.team,
                stage=pipeline.stage,
            ))
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                logger.warning(
                    "Duplicate run insert rejected by DB: pipeline=%s fingerprint=%s run_id=%s "
                    "— another run is already in flight for this fingerprint",
                    pipeline.name, normalised.fingerprint, run_id,
                )
                return False
        return True

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
                provider=result.output.provider if result.output else None,
                prompt=json.dumps(step.executor_config),
                raw_output=json.dumps(result.output.raw_response) if result.output else None,
                parsed_output=result.output.model_dump_json(exclude={"raw_response"}) if result.output else None,
                verifier_output=result.verifier_output.model_dump_json(exclude={"raw_response"}) if result.verifier_output else None,
                verifier_mode=step.verifier.mode if step.verifier and result.verifier_output else None,
                status=result.status,
                primary_confidence=result.output.confidence if result.output else None,
                verifier_confidence=result.verifier_output.confidence if result.verifier_output else None,
                effective_confidence=result.effective_confidence,
                grounding_score=result.grounding_score,
                trust_report=json.dumps(result.trust_report) if result.trust_report else None,
                deterministic_passed=result.deterministic_passed,
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
                provider=output.provider,
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
        trust = step_result.trust_report or {}
        combined_trust = trust.get("combined_trust", step_result.effective_confidence)

        if combined_trust is not None and combined_trust < confidence_threshold:
            escalation_reason = "low_confidence"
        else:
            escalation_reason = step_result.status

        failed_checks = [
            c["name"] for c in (trust.get("deterministic_checks") or []) if not c["passed"]
        ]

        return {
            "escalation_reason": escalation_reason,
            "confidence": step_result.effective_confidence,
            "confidence_threshold": confidence_threshold,
            "contradicts": reasoning.get("contradicts", ""),
            "step_summary": output.summary if output else "",
            "gate_policy": trust.get("gate", {}).get("policy", "legacy_confidence"),
            "combined_trust": combined_trust,
            "grounding_score": step_result.grounding_score,
            "failed_checks": failed_checks,
        }

    async def _fire_step_failure_webhook(
        self,
        step: StepConfig,
        result: "StepResult",
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        step_outputs: "dict[str, LLMOutput]",
        run_log: list,
    ) -> None:
        """Fire the step-level on_failure.webhook callback. Never raises — failures are logged."""
        webhook = step.on_failure.webhook
        assert webhook is not None

        if pipeline.stage == "testing":
            _log_event(
                run_log, "info", "step_failure_webhook_suppressed_testing",
                f"[testing] Step-failure webhook suppressed: {step.name} → {webhook.url}",
                step=step.name,
            )
            return

        try:
            ctx = await build_context(pipeline, normalised, run_id, step.name, step_outputs)
            ctx["step_failure"] = {
                "step": step.name,
                "summary": result.output.summary if result.output else "",
                "status": result.status,
            }

            env = Environment(undefined=Undefined)

            def _render(obj: object) -> object:
                if isinstance(obj, str):
                    return env.from_string(obj).render(**ctx)
                if isinstance(obj, dict):
                    return {k: _render(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_render(item) for item in obj]
                return obj

            rendered_payload = _render(webhook.payload)
            headers = {k: self._resolve_env(v) for k, v in webhook.headers.items()}
            headers.setdefault("Content-Type", "application/json")
            url = self._resolve_env(webhook.url)

            async with httpx.AsyncClient(timeout=webhook.timeout_seconds) as client:
                resp = await client.request(
                    webhook.method.upper(),
                    url,
                    content=json.dumps(rendered_payload).encode(),
                    headers=headers,
                )
            _log_event(run_log, "info", "step_failure_webhook_sent",
                       f"Step failure webhook: {step.name} → HTTP {resp.status_code}",
                       step=step.name)
        except Exception as exc:
            logger.warning("Step failure webhook for '%s' failed: %s — ignored", step.name, exc)
            _log_event(run_log, "warn", "step_failure_webhook_failed",
                       f"Step failure webhook failed: {step.name} — {exc}",
                       step=step.name)

    @staticmethod
    def _resolve_env(value: str) -> str:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

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

        testing = pipeline.stage == "testing"
        for notification in notifications:
            channel = "log" if testing else notification.channel
            notifier = self._notifiers.get(channel)
            if not notifier:
                logger.warning(
                    "No notifier registered for channel '%s' — skipping notification for action '%s'",
                    channel, action,
                )
                continue
            await notifier.send(notification, context)
            if testing:
                _log_event(
                    run_log, "info", "notification_suppressed_testing",
                    f"[testing] Notification routed to log: {action} → would have been "
                    f"{notification.channel}",
                    action=action, channel=notification.channel,
                )
            else:
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
