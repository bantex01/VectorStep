import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..db.models import PipelineRun, PipelineStep
from ..executors.base import BaseExecutor
from ..models.context import NormalisedContext
from ..models.llm import LLMOutput
from ..models.pipeline import (
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
    ):
        self._executor_classes = executors
        self._executor_instances: dict[str, BaseExecutor] = {}
        self._session_factory = session_factory
        self._notifiers: dict[str, TelegramNotifier] = notifiers or {}

    async def run(
        self,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str | None = None,
    ) -> PipelineRunResult:
        run_id = run_id or str(uuid.uuid4())
        logger.info("Pipeline run started: id=%s pipeline=%s", run_id, pipeline.name)

        await self._db_create_run(run_id, pipeline, normalised)

        result = PipelineRunResult(
            run_id=run_id,
            pipeline_name=pipeline.name,
            status="completed",
        )

        step_outputs: dict[str, LLMOutput] = {}

        for index, step in enumerate(pipeline.steps):
            if isinstance(step, ParallelGroupConfig):
                step_name = step.parallel.name
                step_when = step.parallel.when
                step_on_abort = step.parallel.on_abort
                step_threshold = step.parallel.confidence_threshold
            else:
                step_name = step.name
                step_when = step.when
                step_on_abort = step.on_abort
                step_threshold = step.confidence_threshold

            # Evaluate when: condition before doing any work
            if step_when is not None:
                when_ctx = build_context(pipeline, normalised, run_id, step_name, step_outputs)
                if not self._eval_when(step_when, when_ctx):
                    logger.info("Step '%s' skipped — when: condition was false", step_name)
                    continue

            if isinstance(step, ParallelGroupConfig):
                step_result = await self._run_parallel_group(
                    group=step.parallel,
                    index=index,
                    pipeline=pipeline,
                    normalised=normalised,
                    run_id=run_id,
                    step_outputs=step_outputs,
                )
                # Branches saved inside _run_parallel_group; register each individually
                # so downstream steps can reference {{steps.branch_name.field}}
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
                stopped_ctx = build_context(pipeline, normalised, run_id, step_name, step_outputs)
                if step_result.output:
                    stopped_ctx["step_summary"] = step_result.output.summary
                    stopped_ctx["proceed_reason"] = step_result.output.proceed_reason or ""
                await self._dispatch_notification(
                    pipeline=pipeline,
                    action="stopped",
                    context=stopped_ctx,
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
                        **build_context(pipeline, normalised, run_id, step_name, step_outputs),
                        **escalation_ctx,
                    },
                )
                break

            result.final_output = step_result.output

        await self._db_complete_run(run_id, result.status)

        logger.info(
            "Pipeline run finished: id=%s pipeline=%s status=%s",
            run_id, pipeline.name, result.status,
        )
        return result

    # ------------------------------------------------------------------
    # Sequential step execution
    # ------------------------------------------------------------------

    async def _run_step(
        self,
        step: StepConfig,
        index: int,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
        run_id: str,
        step_outputs: dict[str, LLMOutput],
    ) -> StepResult:
        start_ms = int(time.time() * 1000)

        ctx = build_context(pipeline, normalised, run_id, step.name, step_outputs)

        logger.info("Executing step: %s (%d/%d)", step.name, index + 1, len(pipeline.steps))

        max_attempts = step.retry.attempts if step.retry else 1
        last_error: str | None = None
        primary_output: LLMOutput | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                executor = self._get_executor(step.executor)
                coro = executor.execute(step, ctx)
                if step.timeout_seconds:
                    primary_output = await asyncio.wait_for(coro, timeout=step.timeout_seconds)
                else:
                    primary_output = await coro
                last_error = None
                break
            except asyncio.TimeoutError:
                last_error = f"Step timed out after {step.timeout_seconds}s"
                logger.error(
                    "Step '%s' %s (attempt %d/%d)",
                    step.name, last_error, attempt, max_attempts,
                )
            except Exception as exc:
                last_error = f"Executor error: {type(exc).__name__}: {exc}"
                logger.error(
                    "Step '%s' %s (attempt %d/%d)",
                    step.name, last_error, attempt, max_attempts,
                )

            if attempt < max_attempts:
                delay = _compute_backoff(step.retry, attempt)
                logger.info(
                    "Retrying step '%s' in %.1fs (attempt %d/%d)",
                    step.name, delay, attempt + 1, max_attempts,
                )
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

        verifier_output: LLMOutput | None = None
        effective_confidence = primary_output.confidence

        if step.verifier and self._should_verify(step.verifier, primary_output.confidence):
            verifier_output = await self._run_verifier(
                step=step,
                ctx=ctx,
                primary_output=primary_output,
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

        if effective_confidence < step.confidence_threshold:
            action = step.on_low_confidence
            logger.info(
                "Step '%s' below confidence threshold (%.2f < %.2f): action=%s",
                step.name, effective_confidence, step.confidence_threshold, action,
            )
            if action == "abort":
                return StepResult(
                    step_name=step.name,
                    step_index=index,
                    status="aborted",
                    output=primary_output,
                    verifier_output=verifier_output,
                    effective_confidence=effective_confidence,
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
            if action == "escalate":
                return StepResult(
                    step_name=step.name,
                    step_index=index,
                    status="escalated",
                    output=primary_output,
                    verifier_output=verifier_output,
                    effective_confidence=effective_confidence,
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

        if not primary_output.proceed:
            logger.info(
                "Step '%s' returned proceed=false (confidence=%.2f) — resolving pipeline",
                step.name, effective_confidence,
            )
            return StepResult(
                step_name=step.name,
                step_index=index,
                status="stopped",
                output=primary_output,
                verifier_output=verifier_output,
                effective_confidence=effective_confidence,
                duration_ms=int(time.time() * 1000) - start_ms,
            )

        return StepResult(
            step_name=step.name,
            step_index=index,
            status="completed",
            output=primary_output,
            verifier_output=verifier_output,
            effective_confidence=effective_confidence,
            duration_ms=int(time.time() * 1000) - start_ms,
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
    ) -> StepResult:
        start_ms = int(time.time() * 1000)
        ctx = build_context(pipeline, normalised, run_id, group.name, step_outputs)

        logger.info(
            "Executing parallel group '%s' — %d branch(es)", group.name, len(group.steps)
        )

        branch_coros = [self._run_parallel_branch(branch, ctx) for branch in group.steps]

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
                # Unhandled exception escaped _run_parallel_branch — synthesise failure output
                msg = f"Executor error: {type(raw).__name__}: {raw}"
                logger.error("Branch '%s' raised unhandled exception: %s", branch.name, raw)
                output = LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True)
            else:
                output = raw

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

        completed = sum(1 for o in branch_outputs.values() if not getattr(o, "failed", False))
        group_output = LLMOutput(
            confidence=effective_confidence,
            summary=f"Parallel group '{group.name}': {completed}/{len(group.steps)} branches completed",
            next_step_context="",
            raw_response={},
        )

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
        )

    async def _run_parallel_branch(
        self,
        branch: ParallelStepConfig,
        ctx: dict,
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
        except asyncio.TimeoutError:
            msg = f"Branch timed out after {branch.timeout_seconds}s"
            logger.error("Branch '%s' %s", branch.name, msg)
            return LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True)
        except Exception as exc:
            msg = f"Executor error: {type(exc).__name__}: {exc}"
            logger.error("Branch '%s' %s", branch.name, msg)
            return LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True)

        if branch.verifier and self._should_verify(branch.verifier, output.confidence):
            verifier_output = await self._run_verifier(branch_step, ctx, output)
            if verifier_output:
                adjusted = self._combine_confidence(
                    branch_step, output.confidence, verifier_output.confidence
                )
                output = output.model_copy(update={"confidence": adjusted})
                logger.info(
                    "Branch '%s' verifier: primary=%.2f verifier=%.2f effective=%.2f",
                    branch.name, output.confidence, verifier_output.confidence, adjusted,
                )

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
    # DB persistence
    # ------------------------------------------------------------------

    async def _db_create_run(
        self,
        run_id: str,
        pipeline: PipelineConfig,
        normalised: NormalisedContext,
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
                executed_at=datetime.utcnow(),
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
                executed_at=datetime.utcnow(),
            ))
            await session.commit()

    async def _db_complete_run(self, run_id: str, status: str) -> None:
        if not self._session_factory:
            return
        from sqlalchemy import select
        async with self._session_factory() as session:
            run = await session.get(PipelineRun, run_id)
            if run:
                run.status = status
                run.completed_at = datetime.utcnow()
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
