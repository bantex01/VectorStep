import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import httpx
from jinja2 import Environment, Undefined
from opentelemetry.trace import Status, StatusCode
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..artifacts.store import ArtifactStore
from ..db.database import get_pending_approvals_for_run, mark_run_resumed
from ..db.models import PendingApproval, PipelineRun, PipelineStep
from ..executors.base import BaseExecutor, LLMParseError
from ..models.context import NormalisedContext
from ..models.llm import LLMOutput
from .. import live_pricing
from .. import pricing
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
    pipeline_config_fingerprint,
)
from ..notifications.telegram import TelegramNotifier
from .context import build_context
from .versioning import prompt_hash, record_agent_version, record_prompt_version

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
#   {{agent_trace}}      — a formatted transcript of the primary's tool calls + results
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

Execution trace (the primary agent's actual tool calls and results):
---
{{agent_trace}}
---

Review the reasoning above. Assess whether the primary agent's conclusion is \
well-supported, considers the right evidence, and has appropriate confidence. Check \
specific factual claims (a ticket was created, a document was read, a value was found) \
against the execution trace above — a claim with no matching tool call or result is a \
real gap, not just a stylistic concern. If the trace is empty, say so explicitly rather \
than guessing at plausibility.

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
You are a grounding auditor. You are shown the task another agent was given, its \
structured output, and the execution trace it produced (its tool calls and the results \
those tools returned). Your ONLY job is to check whether the agent's load-bearing claims \
are supported by evidence — either the original task input below, or a tool result in the \
trace. You cannot add outside knowledge, you cannot browse, and you are NOT assessing \
whether the conclusion is correct — only whether it is anchored to evidence actually \
available to the agent.

A "load-bearing claim" is an assertion the output depends on: a stated root cause, a \
metric value, a causal link ("X because Y"), a referenced ticket/dashboard/id. Ignore \
hedging, restatements of the task, and generic advice.

IMPORTANT — the task given to the agent below (severity, service, environment, summary, \
and anything else it was handed as input) is GIVEN, trusted context, not something the \
agent needed to discover. A claim that merely restates a fact already present in the \
original task needs NO trace evidence — it was told, not found. Only claims that go \
BEYOND what the agent was given — a root cause, a specific metric value read from a tool, \
a causal link, a referenced ticket/dashboard id it created or looked up — need a \
supporting tool result in the trace.

Original task given to the primary agent:
---
{{primary_prompt}}
---

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
    # Cost in pricing.currency's units (primary + verifier once priced), or None when
    # unpriced/no token data. Set by _db_save_step/_db_save_branch (+group callers) at
    # persistence time — see SPEC-cost-accounting.md. Feeds the budget.max_usd
    # accumulator the same way total_tokens feeds max_tokens (None treated as 0 there).
    cost: float | None = None
    # Best-effort estimate from OpenRouter's live catalog (SPEC-live-pricing.md),
    # computed only when `cost` is None. Never persisted — ephemeral, in-memory-only,
    # so it's available to the budget.max_usd accumulator during THIS run if the
    # pipeline/step opts in via include_approx_cost. Display elsewhere (UI) recomputes
    # its own approximation fresh against the current catalog rather than reusing this.
    approx_cost: float | None = None
    grounding_score: float | None = None       # G ∈ [0,1], or None when not computed
    # Grounding judge's own model/provider/token usage (SPEC-cost-accounting.md) — all
    # None when grounding didn't run (no grounding: block, no trace, or the call errored).
    # Priced into `cost` (and `approx_cost`) alongside primary/verifier.
    grounding_model: str | None = None
    grounding_provider: str | None = None
    grounding_input_tokens: int | None = None
    grounding_output_tokens: int | None = None
    trust_report: dict | None = None           # full TrustReport (§5e), or None
    deterministic_passed: bool | None = None   # None = no checks declared; else all-checks-passed
    # Resume-only (SPEC-durable-runs.md): True when this step/group actually had
    # in-flight work at resume time (a fresh execution, an escalate-on-resume, or a
    # HITL re-arm) — as opposed to being fully persisted already, or not yet reached.
    # Lets the resume loop consume its single on_interrupted policy application at
    # the right step even when a group's completeness can't be known ahead of
    # executing it (a fan-out's branch count depends on re-resolving `over`).
    resume_had_gap: bool = False


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
        gateway_rest_url: str | None = None,
    ):
        self._executor_classes = executors
        self._executor_instances: dict[str, BaseExecutor] = {}
        self._session_factory = session_factory
        self._notifiers: dict[str, TelegramNotifier] = notifiers or {}
        self._artifact_store = artifact_store
        self._pipeline_registry = pipeline_registry
        self._gateway_rest_url = gateway_rest_url

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

    async def execute_candidate(self, step: StepConfig, ctx: dict) -> LLMOutput:
        """Single bare primary-executor call — no verifier, no grounding, no
        loop_until refinement. Retry/timeout semantics mirror _run_step_impl's
        primary-call loop exactly (same executor instances, via _get_executor,
        so replay never spins a parallel execution path), minus everything a
        production step does around that one call.

        Used only by replay.py (SPEC-replay-shadow-eval.md §2 "the candidate
        runs bare — cheaper, and isolates the variable under test"). Raises the
        last error on exhausted retries rather than returning a failed
        LLMOutput, since replay counts an execution failure as unreplayable
        rather than as a graded (and auto-labelled) sample.
        """
        max_attempts = step.retry.attempts if step.retry else 1
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                executor = self._get_executor(step.executor)
                coro = executor.execute(step, ctx)
                if step.timeout_seconds:
                    return await asyncio.wait_for(coro, timeout=step.timeout_seconds)
                return await coro
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    await asyncio.sleep(_compute_backoff(step.retry, attempt))
        assert last_exc is not None
        raise last_exc

    async def run_deterministic_checks(self, step: StepConfig, ctx: dict) -> list[dict]:
        """Public entry point for replay.py — same evaluation as production
        (_run_deterministic_checks), just without a run_log to append to."""
        return await self._run_deterministic_checks(step=step, ctx=ctx, run_log=[])

    def _build_bucket_reset(
        self,
        step_name: str,
        agent: str | None,
        model: str | None,
        provider: str | None,
        current_prompt_hash: str | None,
        current_agent_version: str | None,
    ) -> dict | None:
        """When the current (step, agent, model, provider, prompt_hash, agent_version)
        bucket isn't validated, check whether a DIFFERENT prompt_hash/agent_version of
        the same (step, agent, model, provider) combo was validated — i.e. this isn't
        a step with no history, it's a step whose history just reset (SPEC-prompt-
        versioning.md §4h). Reads CalibrationCache's already-loaded bucket dict via
        previous_versions_for — no extra query."""
        if self._calibration_cache is None:
            return None
        candidates = [
            b for b in self._calibration_cache.previous_versions_for(step_name, agent, model, provider)
            if (b.prompt_hash, b.agent_version) != (current_prompt_hash, current_agent_version)
            and any(bin_.validated for bin_ in b.bins)
        ]
        if not candidates:
            return None
        # Most recently active previous version, if more than one prior version qualifies.
        previous = max(candidates, key=lambda b: b.last_seen_at or datetime.min)

        prompt_changed = previous.prompt_hash != current_prompt_hash
        agent_changed = previous.agent_version != current_agent_version
        reason = (
            "both_changed" if prompt_changed and agent_changed
            else "prompt_changed" if prompt_changed
            else "agent_changed"
        )
        return {
            "reason": reason,
            "previous_version_last_seen": (
                previous.last_seen_at.isoformat() if previous.last_seen_at else None
            ),
            "previous_validated_n": previous.total_n,
        }

    @staticmethod
    def _extract_usage(raw_response: dict) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) from a gateway raw_response, or (0, 0)."""
        usage = ((raw_response or {}).get("meta") or {}).get("agentMeta", {}).get("usage") or {}
        return (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))

    @staticmethod
    def _cost_for(
        provider: str | None, model: str | None,
        input_tokens: int | None, output_tokens: int | None,
    ) -> float | None:
        """Price a raw (provider, model, tokens) tuple against the loaded pricing
        table. None when unpriced (no rate match) or no token data at all —
        shared by the LLMOutput-based helpers below and by grounding, which has
        no LLMOutput of its own by the time cost is computed."""
        return pricing.step_cost(
            pricing.resolve_rate(pricing.get_table(), provider, model),
            input_tokens or None, output_tokens or None,
        )

    @classmethod
    def _output_cost(cls, output: LLMOutput) -> float | None:
        """Price a single LLMOutput against the loaded pricing table. None when
        unpriced (no rate match) or when it reported no token usage at all."""
        in_tok, out_tok = cls._extract_usage(output.raw_response)
        return cls._cost_for(output.provider, output.model, in_tok, out_tok)

    @staticmethod
    def _approx_cost_for(
        provider: str | None, model: str | None,
        input_tokens: int | None, output_tokens: int | None,
    ) -> float | None:
        """Best-effort approximation from OpenRouter's live catalog (SPEC-live-
        pricing.md) — only ever used when _cost_for already returned None."""
        return live_pricing.approx_step_cost(
            live_pricing.resolve_approx_rate(live_pricing.get_catalog(), provider, model),
            input_tokens or None, output_tokens or None,
        )

    @classmethod
    def _output_approx_cost(cls, output: LLMOutput) -> float | None:
        in_tok, out_tok = cls._extract_usage(output.raw_response)
        return cls._approx_cost_for(output.provider, output.model, in_tok, out_tok)

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
            elif t == "tool_denied":
                # Gateway tool_policy blocked this call before it ran — no
                # tool_result follows it, so without this branch the judge sees
                # a TOOL CALL with no matching result and no explanation why.
                reason = str(ev.get("reason", ""))
                if len(reason) > max_chars:
                    reason = reason[:max_chars] + "…"
                lines.append(f"TOOL DENIED ({ev.get('name', '')}): {reason}")
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
        resume: bool = False,
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
                "vectorstep.pipeline.name": pipeline.name,
                "vectorstep.run.id": run_id,
                "vectorstep.source": normalised.source,
                **({"vectorstep.team": normalised.team} if normalised.team else {}),
                **({"vectorstep.parent_run_id": parent_run_id} if parent_run_id else {}),
            },
        ) as run_span:
            result = await self._run_pipeline_body(
                pipeline, normalised, run_id, run_log,
                from_step=from_step,
                initial_step_outputs=initial_step_outputs,
                parent_run_id=parent_run_id,
                resume=resume,
            )
            run_span.set_attribute("vectorstep.run.status", result.status)
            if result.abort_reason:
                run_span.set_attribute("vectorstep.run.abort_reason", result.abort_reason)
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
        resume: bool = False,
    ) -> PipelineRunResult:
        logger.info("Pipeline run started: id=%s pipeline=%s", run_id, pipeline.name)
        _log_event(run_log, "info", "run_started",
                   f"Pipeline started: {pipeline.name} (source: {normalised.source})")

        if from_step:
            n = len(initial_step_outputs or {})
            logger.info("Re-run from step '%s': %d prior step output(s) pre-loaded", from_step, n)
            _log_event(run_log, "info", "rerun_started",
                       f"Re-run from step: {from_step} ({n} prior step output(s) replayed)")

        persisted_branches: dict[str, dict[str, LLMOutput]] = {}
        plain_persisted_names: set[str] = set()
        pending_by_step: dict[str, PendingApproval] = {}

        if resume:
            # The run row already exists and is still 'running' — it never left that
            # state across the outage, so re-inserting would either collide on the
            # primary key or (worse) be mistaken for a dedup rejection. Load what
            # already completed instead (SPEC-durable-runs.md).
            persisted_rows = await self._db_load_run_steps(run_id)
            (
                step_outputs, persisted_branches, plain_persisted_names,
                accumulated_tokens, accumulated_cost,
            ) = self._reconstruct_resume_state(pipeline, persisted_rows)
            for pending in await get_pending_approvals_for_run(run_id):
                pending_by_step[pending.step_name] = pending

            n_skipped = len(plain_persisted_names) + sum(len(v) for v in persisted_branches.values())
            on_interrupted = pipeline.durable.on_interrupted if pipeline.durable else "rerun"
            logger.warning(
                "Resuming run after restart: id=%s pipeline=%s (%d step(s)/branch(es) "
                "already persisted, on_interrupted=%s)",
                run_id, pipeline.name, n_skipped, on_interrupted,
            )
            _log_event(
                run_log, "warn", "run_resumed",
                f"Run resumed after restart — {n_skipped} step(s)/branch(es) already "
                "completed before the crash were loaded from the database, not "
                f"re-executed (on_interrupted={on_interrupted}).",
                steps_skipped=n_skipped, on_interrupted=on_interrupted,
            )
            await mark_run_resumed(run_id)
        else:
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
            step_outputs = dict(initial_step_outputs or {})
            accumulated_tokens = 0
            accumulated_cost = 0.0

        result = PipelineRunResult(
            run_id=run_id,
            pipeline_name=pipeline.name,
            status="completed",
        )

        skipping = (not resume) and from_step is not None
        # Consumed the first time resume finds a step/group that genuinely had
        # in-flight work — see StepResult.resume_had_gap. Only that one step/group
        # gets durable.on_interrupted applied; everything after it executes exactly
        # like a fresh run, since by construction nothing beyond it could have
        # partially executed (single-writer, sequential — SPEC-durable-runs.md §2).
        resume_gap_handled = False

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

            # Whether THIS step's approximate (OpenRouter) cost should fill the gap in
            # the budget.max_usd accumulator when its real cost is None — a step-level
            # override (StepConfig only; groups have no such override) wins over the
            # pipeline's own default. Off by default: an estimate shouldn't be able to
            # abort a run unless explicitly opted into (SPEC-live-pricing.md).
            step_include_approx_cost = (
                step.include_approx_cost if isinstance(step, StepConfig) and step.include_approx_cost is not None
                else bool(pipeline.budget and pipeline.budget.include_approx_cost)
            )

            # A plain step with its own terminal row is done — nothing left to decide
            # about it, its output is already in step_outputs.
            if resume and isinstance(step, StepConfig) and step_name in plain_persisted_names:
                continue

            # Skip steps before from_step when replaying a manual rerun (never set
            # together with resume=True — see main.py's /runs/{id}/rerun vs. the
            # startup resume scheduler).
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

            # Only the single step/group immediately after the last fully-persisted
            # unit is a genuine resume gap — durable.on_interrupted only ever applies
            # there. A group's completeness can't always be known ahead of executing
            # it (fan-out's branch count depends on re-resolving `over`), so this is
            # offered speculatively and only "spent" (resume_gap_handled=True) once
            # the call actually reports in-flight work via resume_had_gap.
            is_gap_candidate = resume and not resume_gap_handled
            gap_on_interrupted = (
                (pipeline.durable.on_interrupted if pipeline.durable else "rerun")
                if is_gap_candidate else None
            )

            if isinstance(step, ParallelGroupConfig):
                step_result = await self._run_parallel_group(
                    group=step.parallel,
                    index=index,
                    pipeline=pipeline,
                    normalised=normalised,
                    run_id=run_id,
                    step_outputs=step_outputs,
                    run_log=run_log,
                    completed_branches=persisted_branches.get(step_name) if resume else None,
                    on_interrupted=gap_on_interrupted,
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
                    completed_branches=persisted_branches.get(step_name) if resume else None,
                    on_interrupted=gap_on_interrupted,
                )
                for branch_name, branch_output in step_result.branch_outputs.items():
                    if branch_output:
                        step_outputs[branch_name] = branch_output
            else:
                pending = pending_by_step.get(step_name) if is_gap_candidate else None
                if is_gap_candidate and pending is not None and step.executor == "human":
                    # HITL steps resume as *waiting*, not re-executed, regardless of
                    # on_interrupted — the approval request was already delivered to
                    # Telegram/Slack/Teams before the crash (SPEC-durable-runs.md §2).
                    step_result = await self._run_step(
                        step=step, index=index * 1000, pipeline=pipeline, normalised=normalised,
                        run_id=run_id, step_outputs=step_outputs, run_log=run_log,
                        resume_pending_token=pending.token,
                    )
                    step_result.resume_had_gap = True
                elif is_gap_candidate and gap_on_interrupted == "escalate":
                    _log_event(
                        run_log, "warn", "resume_step_escalated",
                        f"Step escalated on resume (durable.on_interrupted=escalate): "
                        f"{step.name} — it was in flight when the process died and was "
                        "not re-executed",
                        step=step.name,
                    )
                    step_result = StepResult(
                        step_name=step.name,
                        step_index=index * 1000,
                        status="escalated",
                        output=LLMOutput(
                            confidence=0.0,
                            summary=(
                                f"Step '{step.name}' escalated on resume "
                                "(durable.on_interrupted=escalate) — it was in flight "
                                "when the process died"
                            ),
                            next_step_context="", raw_response={},
                        ),
                        verifier_output=None,
                        effective_confidence=0.0,
                        duration_ms=0,
                        resume_had_gap=True,
                    )
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
                    if is_gap_candidate:
                        step_result.resume_had_gap = True
                await self._db_save_step(run_id, step, step_result)
                if step_result.output:
                    step_outputs[step.name] = step_result.output

            if is_gap_candidate and step_result.resume_had_gap:
                resume_gap_handled = True

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
                        _cost_contribution = step_result.cost
                        if _cost_contribution is None and step_include_approx_cost:
                            _cost_contribution = step_result.approx_cost
                        accumulated_cost += _cost_contribution or 0
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

            # Budget guardrail — check after registering the completed step's output.
            # NULL-cost steps (unpriced, or no token data) contribute 0 to the cost
            # accumulator, mirroring how other-executor steps already contribute 0 to
            # the token accumulator — unless this step opted into approximate cost
            # (step_include_approx_cost), in which case the OpenRouter estimate fills
            # the gap instead of 0.
            accumulated_tokens += step_result.total_tokens
            _cost_contribution = step_result.cost
            if _cost_contribution is None and step_include_approx_cost:
                _cost_contribution = step_result.approx_cost
            accumulated_cost += _cost_contribution or 0
            budget = pipeline.budget
            tokens_exceeded = bool(budget and budget.max_tokens and accumulated_tokens > budget.max_tokens)
            # Checked in this fixed order (tokens, then cost) rather than by which ceiling
            # was configured first — if both trip on the same step, the token message wins
            # and is what's logged; not a meaningful ordering choice, just a deterministic one.
            cost_exceeded = bool(budget and budget.max_usd and accumulated_cost > budget.max_usd)
            if tokens_exceeded or cost_exceeded:
                if tokens_exceeded:
                    msg = (
                        f"Token budget exceeded: {accumulated_tokens:,} tokens used "
                        f"(limit: {budget.max_tokens:,})"
                    )
                else:
                    currency = (pricing.get_table().currency if pricing.get_table() else None) or "USD"
                    msg = (
                        f"Cost budget exceeded: {accumulated_cost:,.2f} {currency} used "
                        f"(limit: {budget.max_usd:,.2f} {currency})"
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

    def _inject_vectorstep_context(self, ctx: dict, normalised: "NormalisedContext") -> None:
        """Inject internal runner references so executor: pipeline can call sub-pipelines."""
        if self._pipeline_registry is not None:
            ctx["_vectorstep_runner"] = self
            ctx["_vectorstep_normalised"] = normalised
            ctx["_vectorstep_registry"] = self._pipeline_registry

    @staticmethod
    def _set_step_span_attributes(
        span, result: "StepResult", prompt_template: str | None = None,
    ) -> None:
        span.set_attribute("vectorstep.step.status", result.status)
        if result.effective_confidence is not None:
            span.set_attribute("vectorstep.confidence.effective", result.effective_confidence)
        if result.output:
            span.set_attribute("vectorstep.confidence.primary", result.output.confidence)
            if result.output.model:
                span.set_attribute("vectorstep.model", result.output.model)
            if result.output.provider:
                span.set_attribute("vectorstep.provider", result.output.provider)
            if result.output.agent_version:
                span.set_attribute("vectorstep.agent_version", result.output.agent_version)
        if result.verifier_output:
            span.set_attribute("vectorstep.confidence.verifier", result.verifier_output.confidence)
        # prompt_template is only passed where a single template unambiguously applies
        # to this span (a plain step, or a fan-out group — all its branches share one
        # template). A parallel group's branches each have their own distinct template,
        # so no single hash would be meaningful there — omitted deliberately, not a gap.
        if prompt_template is not None:
            h = prompt_hash(prompt_template)
            if h is not None:
                span.set_attribute("vectorstep.prompt_hash", h)
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
        resume_pending_token: str | None = None,
    ) -> StepResult:
        agent = step.executor_config.get("agent", "")
        with tracer.start_as_current_span(
            step.name,
            attributes={
                "vectorstep.span.kind": "step",
                "vectorstep.executor": step.executor,
                "vectorstep.agent": agent,
            },
        ) as span:
            result = await self._run_step_impl(
                step, index, pipeline, normalised, run_id, step_outputs, run_log,
                resume_pending_token=resume_pending_token,
            )
            self._set_step_span_attributes(span, result, prompt_template=step.prompt_template)
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
        resume_pending_token: str | None = None,
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
            self._inject_vectorstep_context(ctx, normalised)
            if resume_pending_token:
                # HumanExecutor reads this to re-arm the same token's wait instead of
                # sending a new approval request (SPEC-durable-runs.md §2) — the
                # message was already delivered to Telegram/Slack/Teams pre-crash.
                ctx["_resume_pending_token"] = resume_pending_token
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
                    prompt_hash=prompt_hash(step.prompt_template),
                    agent_version=primary_output.agent_version,
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
                bucket_reset = self._build_bucket_reset(
                    step.name, agent_key, primary_output.model, primary_output.provider,
                    prompt_hash(step.prompt_template), primary_output.agent_version,
                )
                if bucket_reset is not None:
                    calibration_report["bucket_reset"] = bucket_reset
                if step.calibration.on_uncalibrated == "escalate":
                    combined_trust = 0.0

        grounding_model: str | None = None
        grounding_provider: str | None = None
        grounding_input_tokens: int | None = None
        grounding_output_tokens: int | None = None
        _grounding_ran = False  # a real judge call happened — vs. no_trace/error, which didn't
        if step.grounding is not None:
            grounding_score, grounding_report, grounding_tokens = await self._run_grounding(
                step=step, ctx=ctx, primary_output=primary_output, run_log=run_log,
            )
            if step.grounding.enforce and grounding_score is not None:
                combined_trust = min(combined_trust, grounding_score)
                gate_policy = "trust_vector"
            if grounding_report.get("computed"):
                _grounding_ran = True
                grounding_model = grounding_report.get("model")
                grounding_provider = grounding_report.get("provider")
                grounding_input_tokens = grounding_report.get("input_tokens") or None
                grounding_output_tokens = grounding_report.get("output_tokens") or None

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
            or verifier_output is not None
        ):
            trust_report = self._build_trust_report(
                primary_confidence=primary_output.confidence,
                effective_confidence=effective_confidence,
                verifier_confidence=verifier_output.confidence if verifier_output else None,
                verifier_mode=step.verifier.mode if step.verifier and verifier_output else None,
                verifier_combination_strategy=(
                    step.verifier.combination_strategy if step.verifier and verifier_output else None
                ),
                verifier_veto_floor=(
                    step.verifier.veto_floor
                    if step.verifier and verifier_output and step.verifier.combination_strategy == "veto"
                    else None
                ),
                grounding_score=grounding_score,
                grounding_report=grounding_report,
                deterministic_results=deterministic_results,
                calibration_report=calibration_report,
                combined_trust=combined_trust,
                gate_policy=gate_policy,
                confidence_threshold=step.confidence_threshold,
                on_low_confidence=step.on_low_confidence,
            )

        _in_tok, _out_tok = self._extract_usage(primary_output.raw_response)
        if verifier_output:
            _vi, _vo = self._extract_usage(verifier_output.raw_response)
            _in_tok += _vi
            _out_tok += _vo
        _step_tokens = _in_tok + _out_tok + grounding_tokens

        # Computed here (not only at _db_save_step time) so it's available to the
        # budget.max_usd accumulator regardless of whether a session_factory is
        # configured — same reasoning as _step_tokens above. Every component that
        # actually ran (primary always; verifier/grounding only if they ran)
        # contributes to the sum — see the honest-NULL-propagation comment below.
        _real_components = [self._output_cost(primary_output)]
        if verifier_output is not None:
            _real_components.append(self._output_cost(verifier_output))
        if _grounding_ran:
            _real_components.append(
                self._cost_for(grounding_provider, grounding_model, grounding_input_tokens, grounding_output_tokens)
            )
        # Honest NULL propagation: a component that ran but couldn't be priced makes
        # the whole step's cost unknown rather than a silently undercounted partial
        # sum (SPEC-cost-accounting.md §2).
        _step_cost_val = None if any(c is None for c in _real_components) else sum(_real_components)

        # Approximate (OpenRouter) cost only for steps the real pricing table
        # couldn't price — never computed (or shown) alongside a real cost. Unlike
        # the real-cost combination above, this doesn't track which component used
        # which source: it's already labeled an approximation, so summing
        # per-component approx-or-real is unnecessary precision for what's meant
        # to be a rough estimate.
        _step_approx_cost_val: float | None = None
        if _step_cost_val is None:
            _approx_components = [self._output_approx_cost(primary_output)]
            if verifier_output is not None:
                _approx_components.append(self._output_approx_cost(verifier_output))
            if _grounding_ran:
                _approx_components.append(self._approx_cost_for(
                    grounding_provider, grounding_model, grounding_input_tokens, grounding_output_tokens,
                ))
            _step_approx_cost_val = (
                None if any(c is None for c in _approx_components) else sum(_approx_components)
            )

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
                    cost=_step_cost_val,
                    approx_cost=_step_approx_cost_val,
                    grounding_score=grounding_score,
                    grounding_model=grounding_model,
                    grounding_provider=grounding_provider,
                    grounding_input_tokens=grounding_input_tokens,
                    grounding_output_tokens=grounding_output_tokens,
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
                    cost=_step_cost_val,
                    approx_cost=_step_approx_cost_val,
                    grounding_score=grounding_score,
                    grounding_model=grounding_model,
                    grounding_provider=grounding_provider,
                    grounding_input_tokens=grounding_input_tokens,
                    grounding_output_tokens=grounding_output_tokens,
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
                cost=_step_cost_val,
                approx_cost=_step_approx_cost_val,
                grounding_score=grounding_score,
                grounding_model=grounding_model,
                grounding_provider=grounding_provider,
                grounding_input_tokens=grounding_input_tokens,
                grounding_output_tokens=grounding_output_tokens,
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
            cost=_step_cost_val,
            approx_cost=_step_approx_cost_val,
            grounding_score=grounding_score,
            grounding_model=grounding_model,
            grounding_provider=grounding_provider,
            grounding_input_tokens=grounding_input_tokens,
            grounding_output_tokens=grounding_output_tokens,
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
        completed_branches: "dict[str, LLMOutput] | None" = None,
        on_interrupted: str | None = None,
    ) -> StepResult:
        with tracer.start_as_current_span(
            group.name,
            attributes={
                "vectorstep.span.kind": "parallel_group",
                "vectorstep.join_strategy": group.join,
                "vectorstep.branch_count": len(group.steps),
            },
        ) as span:
            result = await self._run_parallel_group_impl(
                group, index, pipeline, normalised, run_id, step_outputs, run_log,
                completed_branches=completed_branches, on_interrupted=on_interrupted,
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
        completed_branches: "dict[str, LLMOutput] | None" = None,
        on_interrupted: str | None = None,
    ) -> StepResult:
        start_ms = int(time.time() * 1000)
        completed_branches = completed_branches or {}
        # Only the branches NOT already persisted from before a restart get a fresh
        # coroutine (SPEC-durable-runs.md) — every other branch is already saved
        # (_db_save_branch, below, is only ever called for branches actually run
        # here), so a resumed run never re-fires a side-effecting branch that
        # completed pre-crash. When nothing was persisted (completed_branches empty,
        # the ordinary case), this is identical to a fresh run.
        missing = [b for b in group.steps if b.name not in completed_branches]

        ctx = await build_context(
            pipeline, normalised, run_id, group.name, step_outputs,
            artifact_store=self._artifact_store,
        )
        self._inject_vectorstep_context(ctx, normalised)

        if on_interrupted == "escalate" and missing:
            _log_event(
                run_log, "warn", "resume_step_escalated",
                f"Parallel group escalated on resume (durable.on_interrupted=escalate): "
                f"{group.name} — {len(missing)}/{len(group.steps)} branch(es) were in "
                "flight when the process died and were not re-executed",
                group=group.name,
            )
            confidences = [o.confidence for o in completed_branches.values()]
            weights = [b.weight for b in group.steps if b.name in completed_branches]
            effective_confidence = (
                self._join_confidences(group.join, confidences, weights) if confidences else 0.0
            )
            _group_tokens = sum(
                sum(self._extract_usage(o.raw_response)) for o in completed_branches.values()
            )
            _group_cost = sum((self._output_cost(o) or 0) for o in completed_branches.values())
            return StepResult(
                step_name=group.name,
                step_index=index,
                status="escalated",
                output=LLMOutput(
                    confidence=effective_confidence,
                    summary=(
                        f"Parallel group '{group.name}' escalated on resume — "
                        f"{len(completed_branches)}/{len(group.steps)} branch(es) had "
                        "completed before the restart"
                    ),
                    next_step_context="", raw_response={},
                ),
                verifier_output=None,
                effective_confidence=effective_confidence,
                duration_ms=int(time.time() * 1000) - start_ms,
                branch_outputs=dict(completed_branches),
                total_tokens=_group_tokens,
                cost=_group_cost,
                resume_had_gap=True,
            )

        if completed_branches:
            logger.info(
                "Executing parallel group '%s' — %d branch(es) to run, %d already "
                "persisted from before a restart",
                group.name, len(missing), len(completed_branches),
            )
        else:
            logger.info(
                "Executing parallel group '%s' — %d branch(es)", group.name, len(group.steps)
            )
        _log_event(run_log, "info", "parallel_group_started",
                   f"Parallel group started: {group.name} ({len(missing)} branch(es) to run)",
                   group=group.name)

        branch_coros = [
            self._run_parallel_branch(branch, ctx, run_log, run_id) for branch in missing
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

        branch_outputs: dict[str, LLMOutput] = dict(completed_branches)

        for i, (branch, raw) in enumerate(zip(missing, raw_results)):
            verifier_output: LLMOutput | None = None
            primary_confidence: float | None = None
            if isinstance(raw, Exception):
                msg = f"Executor error: {type(raw).__name__}: {raw}"
                logger.error("Branch '%s' raised unhandled exception: %s", branch.name, raw)
                _log_event(run_log, "error", "branch_failed",
                           f"Branch failed: {branch.name} — {msg}",
                           branch=branch.name, group=group.name)
                output = LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True)
            else:
                output, verifier_output, primary_confidence = raw
                if not getattr(output, "failed", False):
                    _log_event(run_log, "info", "branch_completed",
                               f"Branch completed: {branch.name} — confidence {output.confidence:.0%}",
                               branch=branch.name, group=group.name)

            branch_outputs[branch.name] = output

            # Original config index (not position within `missing`) — keeps step_index
            # consistent with a fresh run's ordering regardless of which branches were
            # already persisted.
            branch_config_index = next(j for j, b in enumerate(group.steps) if b.name == branch.name)
            await self._db_save_branch(
                run_id, group.name, branch, index, branch_config_index, output,
                verifier_output, primary_confidence,
            )

        # Confidence/weight are recomputed from the FULL branch set (persisted +
        # freshly executed) in group.steps order, not execution order — join
        # strategies like weighted_average are order-sensitive to steps declaration,
        # matching a fresh run's ordering exactly.
        confidences = [branch_outputs[b.name].confidence for b in group.steps if b.name in branch_outputs]
        weights = [b.weight for b in group.steps if b.name in branch_outputs]
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
        # Budget-accumulator use only (each branch's own cost is already persisted by
        # _db_save_branch above) — an unpriced branch contributes 0 here, same as an
        # unpriced branch contributes 0 tokens to _group_tokens. _group_approx_cost is
        # the same sum with an OpenRouter approximation filling in per-branch where the
        # real cost is None — only used if the pipeline opts in via
        # budget.include_approx_cost (groups have no per-step override).
        _group_cost = 0.0
        _group_approx_cost = 0.0
        for o in branch_outputs.values():
            _branch_real = self._output_cost(o)
            _group_cost += _branch_real or 0
            _group_approx_cost += (_branch_real if _branch_real is not None else self._output_approx_cost(o)) or 0

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
                    cost=_group_cost,
                    approx_cost=_group_approx_cost,
                    resume_had_gap=bool(missing),
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
            cost=_group_cost,
            approx_cost=_group_approx_cost,
            resume_had_gap=bool(missing),
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
        completed_branches: "dict[str, LLMOutput] | None" = None,
        on_interrupted: str | None = None,
    ) -> StepResult:
        with tracer.start_as_current_span(
            fan_out.name,
            attributes={
                "vectorstep.span.kind": "fan_out",
                "vectorstep.join_strategy": fan_out.join,
            },
        ) as span:
            result = await self._run_fan_out_impl(
                fan_out, index, pipeline, normalised, run_id, step_outputs, run_log,
                completed_branches=completed_branches, on_interrupted=on_interrupted,
            )
            self._set_step_span_attributes(span, result, prompt_template=fan_out.prompt_template)
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
        completed_branches: "dict[str, LLMOutput] | None" = None,
        on_interrupted: str | None = None,
    ) -> StepResult:
        import ast
        from jinja2 import Environment

        start_ms = int(time.time() * 1000)
        completed_branches = completed_branches or {}

        base_ctx = await build_context(
            pipeline, normalised, run_id, fan_out.name, step_outputs,
            artifact_store=self._artifact_store,
        )
        self._inject_vectorstep_context(base_ctx, normalised)

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

        # Only items not already persisted from before a restart get re-executed
        # (SPEC-durable-runs.md) — the branch name convention (f"{fan_out.name}/{i}")
        # is deterministic given `items`, itself deterministic given the same
        # reconstructed step_outputs `over` was rendered against, so this lines up
        # with whatever the crash actually left persisted.
        missing_indices = [i for i in range(total) if f"{fan_out.name}/{i}" not in completed_branches]

        if on_interrupted == "escalate" and missing_indices:
            _log_event(
                run_log, "warn", "resume_step_escalated",
                f"Fan-out escalated on resume (durable.on_interrupted=escalate): "
                f"{fan_out.name} — {len(missing_indices)}/{total} branch(es) were in "
                "flight when the process died and were not re-executed",
                step=fan_out.name,
            )
            confidences = [o.confidence for o in completed_branches.values()]
            effective_confidence = (
                self._join_confidences(fan_out.join, confidences, [1.0] * len(confidences))
                if confidences else 0.0
            )
            _fan_out_tokens = sum(
                sum(self._extract_usage(o.raw_response)) for o in completed_branches.values()
            )
            _fan_out_cost = sum((self._output_cost(o) or 0) for o in completed_branches.values())
            return StepResult(
                step_name=fan_out.name,
                step_index=index,
                status="escalated",
                output=LLMOutput(
                    confidence=effective_confidence,
                    summary=(
                        f"Fan-out '{fan_out.name}' escalated on resume — "
                        f"{len(completed_branches)}/{total} branch(es) had completed "
                        "before the restart"
                    ),
                    next_step_context="", raw_response={},
                ),
                verifier_output=None,
                effective_confidence=effective_confidence,
                duration_ms=int(time.time() * 1000) - start_ms,
                branch_outputs=dict(completed_branches),
                total_tokens=_fan_out_tokens,
                cost=_fan_out_cost,
                resume_had_gap=True,
            )

        logger.info(
            "Fan-out '%s': %d item(s), %d to run", fan_out.name, total, len(missing_indices)
        )
        _log_event(run_log, "info", "fan_out_started",
                   f"Fan-out started: {fan_out.name} ({len(missing_indices)} of {total} "
                   "item(s) to run)", step=fan_out.name)

        branch_coros = []
        for i in missing_indices:
            item = items[i]
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

        branch_outputs: dict[str, LLMOutput] = dict(completed_branches)

        for i, raw in zip(missing_indices, raw_results):
            branch_name = f"{fan_out.name}/{i}"
            verifier_output: LLMOutput | None = None
            primary_confidence: float | None = None
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
                output, verifier_output, primary_confidence = raw
                if not getattr(output, "failed", False):
                    _log_event(run_log, "info", "fan_out_branch_completed",
                               f"Fan-out branch completed: {branch_name} — "
                               f"confidence {output.confidence:.0%}",
                               step=fan_out.name, branch=branch_name)

            branch_outputs[branch_name] = output

            # DB branch name is str(i) so stored step_name = "{fan_out.name}/{i}"
            db_branch = ParallelStepConfig(
                name=str(i),
                executor=fan_out.executor,
                executor_config=fan_out.executor_config,
                prompt_template=fan_out.prompt_template,
                verifier=fan_out.verifier,
            )
            await self._db_save_branch(
                run_id, fan_out.name, db_branch, index, i, output,
                verifier_output, primary_confidence,
            )

        # Recomputed from the FULL branch set (persisted + freshly executed) in item
        # order, not execution order — matches a fresh run's ordering exactly.
        confidences = [
            branch_outputs[f"{fan_out.name}/{i}"].confidence
            for i in range(total) if f"{fan_out.name}/{i}" in branch_outputs
        ]
        weights = [1.0] * len(confidences)
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
        _fan_out_cost = 0.0
        _fan_out_approx_cost = 0.0
        for o in branch_outputs.values():
            _branch_real = self._output_cost(o)
            _fan_out_cost += _branch_real or 0
            _fan_out_approx_cost += (_branch_real if _branch_real is not None else self._output_approx_cost(o)) or 0

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
                    cost=_fan_out_cost,
                    approx_cost=_fan_out_approx_cost,
                    resume_had_gap=bool(missing_indices),
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
            resume_had_gap=bool(missing_indices),
            total_tokens=_fan_out_tokens,
            cost=_fan_out_cost,
            approx_cost=_fan_out_approx_cost,
        )

    async def _run_parallel_branch(
        self,
        branch: ParallelStepConfig,
        ctx: dict,
        run_log: list,
        run_id: str,
    ) -> tuple[LLMOutput, LLMOutput | None, float]:
        # TODO(grounding phase): branches — grounding is not yet wired into fan-out/
        # parallel branches, only sequential StepConfig steps (see SPEC-grounding-shadow.md).
        agent = branch.executor_config.get("agent", "")
        with tracer.start_as_current_span(
            branch.name,
            attributes={
                "vectorstep.span.kind": "branch",
                "vectorstep.executor": branch.executor,
                "vectorstep.agent": agent,
            },
        ) as span:
            output, verifier_output, primary_confidence = await self._run_parallel_branch_impl(
                branch, ctx, run_log, run_id
            )
            span.set_attribute("vectorstep.confidence", output.confidence)
            if output.model:
                span.set_attribute("vectorstep.model", output.model)
            if output.provider:
                span.set_attribute("vectorstep.provider", output.provider)
            if getattr(output, "failed", False):
                span.set_status(Status(StatusCode.ERROR))
            return output, verifier_output, primary_confidence

    async def _run_parallel_branch_impl(
        self,
        branch: ParallelStepConfig,
        ctx: dict,
        run_log: list,
        run_id: str,
    ) -> tuple[LLMOutput, LLMOutput | None, float]:
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
            return LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True), None, 0.0
        except Exception as exc:
            msg = f"Executor error: {type(exc).__name__}: {exc}"
            logger.error("Branch '%s' %s", branch.name, msg)
            return LLMOutput(confidence=0.0, summary=msg, next_step_context="", raw_response={}, failed=True), None, 0.0

        verifier_output: LLMOutput | None = None
        primary_confidence = output.confidence
        if branch.verifier and self._should_verify(branch.verifier, output.confidence):
            verifier_output = await self._run_verifier(branch_step, ctx, output, run_log)
            if verifier_output:
                adjusted = self._combine_confidence(
                    branch_step, primary_confidence, verifier_output.confidence
                )
                output = output.model_copy(update={"confidence": adjusted})
                logger.info(
                    "Branch '%s' verifier: primary=%.2f verifier=%.2f effective=%.2f",
                    branch.name, primary_confidence, verifier_output.confidence, adjusted,
                )
                _log_event(run_log, "info", "verifier_ran",
                           f"Verifier ran: {branch.name} — primary {primary_confidence:.0%} / "
                           f"verifier {verifier_output.confidence:.0%} → effective {adjusted:.0%}",
                           branch=branch.name)

        logger.debug(
            "Branch '%s' confidence=%.2f summary=%s",
            branch.name, output.confidence, output.summary,
        )
        return output, verifier_output, primary_confidence

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
            attributes={"vectorstep.span.kind": "verifier", "vectorstep.verifier.mode": verifier.mode},
        ) as span:
            output = await self._run_verifier_impl(step, ctx, primary_output, run_log)
            if output is not None:
                span.set_attribute("vectorstep.confidence", output.confidence)
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
            # Without the trace, a critic can only judge the primary's ACCOUNT of its
            # work (its self-reported summary/reasoning), never the work itself — it
            # can't tell "claims a ticket was created" apart from "actually created one".
            # Same transcript-building grounding already uses, so a claim invisible to
            # one is invisible to the other for the same reason (see grounding.max_trace_chars).
            trace = (primary_output.raw_response or {}).get("trace") or []
            transcript = self._format_trace_for_grounding(trace, max_chars=verifier.max_trace_chars)
            verifier_ctx = {
                **ctx,
                "primary_prompt": primary_prompt,
                "primary_response": json.dumps(
                    primary_output.model_dump(exclude={"raw_response"}), indent=2
                ),
                "agent_trace": transcript,
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
        transcript = self._format_trace_for_grounding(trace, max_chars=grounding.max_trace_chars)
        if not transcript:
            # No evidence trail (non-gateway step, or a trace with no tool activity):
            # "nothing to check" is null, not zero.
            return None, {"computed": False, "reason": "no_trace", "agent": grounding.agent, "enforce": grounding.enforce}, 0

        grounding_ctx = {
            **ctx,
            # The original task the primary agent was given — same rendering the critic
            # verifier already shares (see _run_verifier_impl). Without this, the judge
            # has no way to tell "restates a fact it was given as input" (e.g. alert
            # severity, service name) apart from "claims something it needed to discover"
            # — it would mark input facts unsupported for lack of a matching tool result,
            # which is a false "unsupported" verdict, not a real gap in the evidence.
            "primary_prompt": Environment().from_string(step.prompt_template).render(**ctx),
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
            attributes={"vectorstep.span.kind": "grounding", "vectorstep.agent": grounding.agent},
        ) as span:
            try:
                executor = self._get_executor(grounding.executor)
                coro = executor.execute(grounding_step, grounding_ctx)
                out = await asyncio.wait_for(coro, timeout=grounding.timeout_seconds)
            except Exception as exc:
                # asyncio.TimeoutError (raised by the wait_for above, not by the
                # executor) carries no message of its own — str(exc) is '', which
                # renders as a blank, useless "failed: " line. Give it one.
                error_message = (
                    f"grounding call timed out after {grounding.timeout_seconds}s"
                    if isinstance(exc, asyncio.TimeoutError) else str(exc)
                )
                logger.warning(
                    "Grounding pass for step '%s' failed: %s — recording G=null",
                    step.name, error_message,
                )
                _log_event(run_log, "warn", "grounding_failed",
                           f"Grounding failed for {step.name}: {error_message}", step=step.name)
                span.set_status(Status(StatusCode.ERROR))
                error_report = {"computed": False, "reason": "error", "error": error_message,
                                 "agent": grounding.agent, "enforce": grounding.enforce}
                if isinstance(exc, LLMParseError):
                    # The judge's own JSON failed to parse — persist the full,
                    # untruncated text it actually returned (not just the message's
                    # 500-char snippet) so a human can tell "truncated mid-output"
                    # apart from "malformed from the start".
                    error_report["raw_output"] = exc.raw_text
                return None, error_report, 0

            g = max(0.0, min(1.0, float(out.confidence)))
            span.set_attribute("vectorstep.grounding.score", g)
            claims = (out.reasoning or {}).get("claims") if out.reasoning else None
            _gi, _go = self._extract_usage(out.raw_response)
            _log_event(run_log, "info", "grounding_ran",
                       f"Grounding (shadow): {step.name} — G {g:.0%} vs self-report "
                       f"{primary_output.confidence:.0%}", step=step.name)
            report = {
                "computed": True,
                "agent": grounding.agent,
                "model": out.model,
                "provider": out.provider,
                "input_tokens": _gi,
                "output_tokens": _go,
                "enforce": grounding.enforce,
                "score": g,
                "summary": out.summary,
                "claims": claims if isinstance(claims, list) else [],
                "prompt": (out.raw_response or {}).get("prompt"),
                "raw_output": (out.raw_response or {}).get("response_text"),
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
        # `finalize` shell-quotes every interpolated {{ }} value (shlex.quote) so a
        # step's output — agent-generated, not operator-controlled — can't break out
        # of the command the pipeline author wrote via embedded shell metacharacters
        # (;, |, $(...), quotes). The command template itself is trusted git-controlled
        # YAML; a step's output value is data and must be treated as such, never as
        # shell syntax, even though the command *as authored* still runs unsandboxed
        # (see README's "Unsandboxed by design" note — that's about the operator's own
        # command, not about giving arbitrary injected values a free pass).
        env = Environment(undefined=Undefined, finalize=lambda v: shlex.quote(str(v)))
        command = env.from_string(check.run).render(**ctx)

        proc = await asyncio.create_subprocess_shell(
            command,
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
        headers = {
            k: self._resolve_env(env.from_string(v).render(**ctx))
            for k, v in check.headers.items()
        }
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
        verifier_combination_strategy: str | None,
        verifier_veto_floor: float | None,
        grounding_score: float | None,
        grounding_report: dict | None,
        deterministic_results: list[dict] | None,
        calibration_report: dict | None,
        combined_trust: float,
        gate_policy: str,
        confidence_threshold: float,
        on_low_confidence: str,
    ) -> dict:
        deterministic_passed = (
            all(r["passed"] for r in deterministic_results) if deterministic_results else None
        )
        return {
            "version": 5,   # bumped from 4 — gate.confidence_threshold/on_low_confidence are new
            "mode": "enforced" if gate_policy == "trust_vector" else "shadow",
            "signals": {
                "S": primary_confidence,
                "S_after_V": effective_confidence,
                "V": verifier_confidence,
                "V_mode": verifier_mode,   # "critic" | "independent" | null (no verifier ran)
                "V_combination_strategy": verifier_combination_strategy,   # NEW — "minimum" | "veto" | null
                "V_veto_floor": verifier_veto_floor,   # NEW — only set when strategy is "veto"; null otherwise
                "G": grounding_score,
                "C": None,                      # consistency — still a later phase
                "D": deterministic_passed,       # NEW — bool, or null if no checks declared
            },
            "combined_trust": combined_trust,    # NEW — what the gate actually compared
            "grounding": grounding_report,
            "deterministic_checks": deterministic_results,   # NEW — full per-check detail, or null
            "calibration": calibration_report,   # NEW — see calibration.py, or null if not enforced
            "gate": {
                "policy": gate_policy,      # "legacy_confidence" | "trust_vector"
                "confidence_threshold": confidence_threshold,   # NEW — what combined_trust was compared against
                "on_low_confidence": on_low_confidence,          # NEW — "escalate" | "abort" | "proceed"
            },
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
                config_fingerprint=pipeline_config_fingerprint(pipeline),
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
            if result.verifier_output is None:
                _verifier_in_tok, _verifier_out_tok = None, None
            else:
                _vi, _vo = self._extract_usage(result.verifier_output.raw_response)
                _verifier_in_tok, _verifier_out_tok = _vi or None, _vo or None

            # result.cost was already computed in _run_step_impl (so it's available to
            # the budget.max_usd accumulator regardless of whether this method's
            # session_factory guard above returns early) — persisted here, not
            # recomputed, so the two can never disagree.
            # The gateway executor stashes the rendered prompt text in raw_response —
            # use that when present (the actual instructions the agent was given,
            # needed to judge whether a claim like "the agent didn't check X" is a real
            # gap or the prompt never asked for X). Other executors don't set this key,
            # so this falls back to the executor_config dump they've always gotten.
            _rendered_prompt = (result.output.raw_response or {}).get("prompt") if result.output else None
            _prompt_hash = prompt_hash(step.prompt_template)
            _agent_key = f"{step.executor}:{_agent}" if _agent else None
            _agent_version = result.output.agent_version if result.output else None
            session.add(PipelineStep(
                run_id=run_id,
                step_name=result.step_name,
                step_index=result.step_index,
                executor=step.executor,
                agent=_agent_key,
                model=result.output.model if result.output else None,
                provider=result.output.provider if result.output else None,
                prompt_hash=_prompt_hash,
                agent_version=_agent_version,
                prompt=_rendered_prompt if _rendered_prompt else json.dumps(step.executor_config),
                raw_output=json.dumps(result.output.raw_response) if result.output else None,
                parsed_output=result.output.model_dump_json(exclude={"raw_response"}) if result.output else None,
                verifier_output=result.verifier_output.model_dump_json(exclude={"raw_response"}) if result.verifier_output else None,
                verifier_mode=step.verifier.mode if step.verifier and result.verifier_output else None,
                verifier_agent=(
                    f"{step.verifier.executor}:{step.verifier.executor_config.get('agent')}"
                    if step.verifier and result.verifier_output and step.verifier.executor_config.get("agent")
                    else None
                ),
                verifier_model=result.verifier_output.model if result.verifier_output else None,
                verifier_provider=result.verifier_output.provider if result.verifier_output else None,
                verifier_prompt=(
                    (result.verifier_output.raw_response or {}).get("prompt")
                    if result.verifier_output else None
                ),
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
                verifier_input_tokens=_verifier_in_tok,
                verifier_output_tokens=_verifier_out_tok,
                grounding_model=result.grounding_model,
                grounding_provider=result.grounding_provider,
                grounding_input_tokens=result.grounding_input_tokens,
                grounding_output_tokens=result.grounding_output_tokens,
                cost=result.cost,
            ))
            if _prompt_hash is not None:
                await record_prompt_version(
                    session, hash_=_prompt_hash, step_name=result.step_name,
                    template=step.prompt_template,
                )
            if self._gateway_rest_url and _agent_version is not None and _agent_key is not None:
                await record_agent_version(
                    session, self._gateway_rest_url,
                    agent_version=_agent_version, agent=_agent_key,
                )
            await session.commit()

    async def _db_save_branch(
        self,
        run_id: str,
        group_name: str,
        branch: ParallelStepConfig,
        group_index: int,
        branch_index: int,
        output: LLMOutput,
        verifier_output: LLMOutput | None = None,
        primary_confidence: float | None = None,
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
        if verifier_output is None:
            _verifier_in_tok, _verifier_out_tok = None, None
        else:
            _vi, _vo = self._extract_usage(verifier_output.raw_response)
            _verifier_in_tok, _verifier_out_tok = _vi or None, _vo or None
        _branch_cost = self._output_cost(output)
        if verifier_output is not None:
            _branch_cost = (_branch_cost or 0) + (self._output_cost(verifier_output) or 0)
        _branch_step_name = f"{group_name}/{branch.name}"
        _prompt_hash = prompt_hash(branch.prompt_template)
        _agent_key = f"{branch.executor}:{_agent}" if _agent else None
        async with self._session_factory() as session:
            session.add(PipelineStep(
                run_id=run_id,
                step_name=_branch_step_name,
                # Encode group position + branch position so DB ordering mirrors execution order.
                # Branches share the same group_index prefix and sort together.
                step_index=group_index * 1000 + branch_index,
                executor=branch.executor,
                agent=_agent_key,
                model=output.model,
                provider=output.provider,
                prompt_hash=_prompt_hash,
                agent_version=output.agent_version,
                prompt=json.dumps(branch.executor_config),
                raw_output=json.dumps(output.raw_response),
                parsed_output=output.model_dump_json(exclude={"raw_response"}),
                verifier_output=verifier_output.model_dump_json(exclude={"raw_response"}) if verifier_output else None,
                verifier_mode=branch.verifier.mode if branch.verifier and verifier_output else None,
                verifier_agent=(
                    f"{branch.verifier.executor}:{branch.verifier.executor_config.get('agent')}"
                    if branch.verifier and verifier_output and branch.verifier.executor_config.get("agent")
                    else None
                ),
                verifier_model=verifier_output.model if verifier_output else None,
                verifier_provider=verifier_output.provider if verifier_output else None,
                verifier_prompt=(
                    (verifier_output.raw_response or {}).get("prompt") if verifier_output else None
                ),
                status="failed" if branch_failed else "completed",
                primary_confidence=None if branch_failed else (primary_confidence if primary_confidence is not None else output.confidence),
                verifier_confidence=verifier_output.confidence if verifier_output else None,
                effective_confidence=None if branch_failed else output.confidence,
                duration_ms=None,
                executed_at=utc_now(),
                artifacts=_artifact_refs,
                agent_trace=_trace,
                input_tokens=_in_tok or None,
                output_tokens=_out_tok or None,
                verifier_input_tokens=_verifier_in_tok,
                verifier_output_tokens=_verifier_out_tok,
                cost=_branch_cost,
            ))
            if _prompt_hash is not None:
                # Registry step_name uses the collapsed group name (not the full
                # "group/branch" runtime name), matching calibration.py's own
                # step_name.split("/", 1)[0] collapse — so GET /steps/{name}/versions
                # and GET /steps/{name}/calibration key on the same step_name universe.
                await record_prompt_version(
                    session, hash_=_prompt_hash, step_name=group_name,
                    template=branch.prompt_template,
                )
            if self._gateway_rest_url and output.agent_version is not None and _agent_key is not None:
                await record_agent_version(
                    session, self._gateway_rest_url,
                    agent_version=output.agent_version, agent=_agent_key,
                )
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

    async def _db_load_run_steps(self, run_id: str) -> list[PipelineStep]:
        """All persisted steps for a run, in execution order — the resume path's only
        source of truth for what already completed (SPEC-durable-runs.md). A run still
        in status='running' can only have 'completed' step rows: any other status
        would have already ended the run via the abort/escalate/fail/stop break in the
        main loop, flipping it out of 'running' — so resume never has to reason about
        a persisted row in some other state.
        """
        if not self._session_factory:
            return []
        from sqlalchemy import select
        async with self._session_factory() as session:
            result = await session.execute(
                select(PipelineStep)
                .where(PipelineStep.run_id == run_id)
                .order_by(PipelineStep.step_index)
            )
            return list(result.scalars().all())

    def _reconstruct_resume_state(
        self, pipeline: PipelineConfig, persisted: list[PipelineStep],
    ) -> tuple[dict[str, LLMOutput], dict[str, dict[str, LLMOutput]], set[str], int, float]:
        """Rebuild everything the step loop needs to pick up where it left off:
          - step_outputs: keyed exactly as the live loop would key them (bare branch
            name for a parallel group's branches, matching _run_parallel_group_impl;
            the full "group/branch" string for a fan-out's, matching _run_fan_out_impl
            — the two differ today, see runner.py's own branch-registration loops in
            _run_pipeline_body, and reconstruction must mirror that exactly or a
            downstream {{steps.*}} reference would resolve differently after a resume
            than it would have live).
          - persisted_branches: {group_or_fan_out_name: {branch_key: output}} — what
            _run_parallel_group/_run_fan_out use to skip already-done branches.
          - plain_persisted_names: step.name values with their own terminal row —
            these are skipped outright, matching a completed step's persisted output
            already being in step_outputs.
          - accumulated_tokens / accumulated_cost: seeded from every persisted row
            (plain steps and branches alike) so the budget guardrail counts
            pre-restart usage, which the from_step-based manual rerun path (main.py's
            /runs/{id}/rerun) does NOT do today — a resume must, or a run that already
            spent most of its budget.max_tokens before the crash would get a fresh
            budget on top of it.
        """
        group_names = {
            step.parallel.name for step in pipeline.steps if isinstance(step, ParallelGroupConfig)
        }
        fan_out_names = {
            step.fan_out.name for step in pipeline.steps if isinstance(step, FanOutGroupConfig)
        }

        step_outputs: dict[str, LLMOutput] = {}
        persisted_branches: dict[str, dict[str, LLMOutput]] = {}
        plain_persisted_names: set[str] = set()
        accumulated_tokens = 0
        accumulated_cost = 0.0

        for row in persisted:
            if not row.parsed_output:
                continue
            output = LLMOutput.model_validate(json.loads(row.parsed_output))
            # Branch rows have no verifier/grounding columns (neither is wired into
            # parallel/fan-out branches — see _run_parallel_branch_impl's TODO), so
            # these are always 0 there; for a plain step's row they mirror exactly
            # what _step_tokens summed live in _run_step_impl (primary + verifier +
            # grounding), which input_tokens/output_tokens alone would undercount.
            accumulated_tokens += (
                (row.input_tokens or 0) + (row.output_tokens or 0)
                + (row.verifier_input_tokens or 0) + (row.verifier_output_tokens or 0)
                + (row.grounding_input_tokens or 0) + (row.grounding_output_tokens or 0)
            )
            accumulated_cost += row.cost or 0

            if "/" in row.step_name:
                prefix, suffix = row.step_name.split("/", 1)
                persisted_branches.setdefault(prefix, {})[
                    suffix if prefix in group_names else row.step_name
                ] = output
                if prefix in group_names:
                    step_outputs[suffix] = output
                elif prefix in fan_out_names:
                    step_outputs[row.step_name] = output
            else:
                plain_persisted_names.add(row.step_name)
                step_outputs[row.step_name] = output

        return step_outputs, persisted_branches, plain_persisted_names, accumulated_tokens, accumulated_cost

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
