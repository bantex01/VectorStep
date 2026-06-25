import logging
import uuid

from jinja2 import Environment

from .base import BaseExecutor
from ..models.llm import LLMOutput
from ..models.pipeline import StepConfig

logger = logging.getLogger(__name__)


class PipelineExecutor(BaseExecutor):
    """Executor that calls another named pipeline as a sub-pipeline.

    Requires pipeline_registry to be set on the PipelineRunner — the runner
    injects _pork_runner, _pork_normalised, and _pork_registry into the Jinja2
    context before calling execute().

    executor_config keys:
      pipeline  (required) — name of the pipeline to call
      context   (optional) — dict of Jinja2 templates that override fields on
                             the sub-pipeline's NormalisedContext. Labels and
                             metadata are merged with the parent's, not replaced.
    """

    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        runner = context.get("_pork_runner")
        normalised = context.get("_pork_normalised")
        registry = context.get("_pork_registry")
        parent_run_id = context.get("pipeline_run_id")

        if runner is None or normalised is None or registry is None:
            raise RuntimeError(
                "executor: pipeline requires pipeline_registry to be configured on "
                "PipelineRunner — set it via the pipeline_registry constructor arg"
            )

        pipeline_name = step.executor_config.get("pipeline")
        if not pipeline_name:
            raise ValueError(
                "executor_config.pipeline is required when using executor: pipeline"
            )

        sub_pipeline = registry.get(pipeline_name)
        if sub_pipeline is None:
            raise ValueError(
                f"Sub-pipeline '{pipeline_name}' not found in registry. "
                f"Available: {sorted(registry.keys())}"
            )

        # Build the sub-pipeline's NormalisedContext from the parent's.
        # fingerprint=None bypasses dedup; source marks the run's origin.
        # team (like labels/metadata/severity/summary) is inherited from the
        # parent unmodified here — overridable below via context: {team: ...}.
        sub_normalised = normalised.model_copy(update={
            "pipeline": pipeline_name,
            "source": "sub-pipeline",
            "fingerprint": None,
        })

        # Apply optional context: overrides (Jinja2-rendered, merged for dicts).
        context_overrides = step.executor_config.get("context") or {}
        if context_overrides:
            env = Environment()
            updates: dict = {}
            for key, template in context_overrides.items():
                if key in ("labels", "metadata") and isinstance(template, dict):
                    rendered = {
                        k: env.from_string(str(v)).render(**context) if isinstance(v, str) else v
                        for k, v in template.items()
                    }
                    parent_val = dict(getattr(normalised, key, {}) or {})
                    parent_val.update(rendered)
                    updates[key] = parent_val
                elif isinstance(template, str):
                    updates[key] = env.from_string(template).render(**context)
                else:
                    updates[key] = template
            sub_normalised = sub_normalised.model_copy(update=updates)

        sub_run_id = str(uuid.uuid4())
        logger.info(
            "Sub-pipeline '%s': spawning run_id=%s (parent=%s)",
            pipeline_name, sub_run_id, parent_run_id,
        )

        result = await runner.run(
            pipeline=sub_pipeline,
            normalised=sub_normalised,
            run_id=sub_run_id,
            parent_run_id=parent_run_id,
        )

        logger.info(
            "Sub-pipeline '%s' run_id=%s finished: status=%s",
            pipeline_name, sub_run_id, result.status,
        )

        if result.final_output is not None:
            return result.final_output.model_copy(update={
                "sub_run_id": sub_run_id,
                "sub_pipeline_status": result.status,
            })

        # Sub-pipeline had no step output (empty pipeline or all steps skipped).
        confidence = 1.0 if result.status == "completed" else 0.0
        return LLMOutput(
            confidence=confidence,
            summary=f"Sub-pipeline '{pipeline_name}' {result.status}",
            next_step_context=result.abort_reason or "",
            raw_response={},
            sub_run_id=sub_run_id,
            sub_pipeline_status=result.status,
        )
