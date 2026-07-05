import logging
from typing import TYPE_CHECKING, Any

from ..models.context import NormalisedContext
from ..models.llm import LLMOutput
from ..models.pipeline import PipelineConfig

if TYPE_CHECKING:
    from ..artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)


async def build_context(
    pipeline: PipelineConfig,
    normalised: NormalisedContext,
    pipeline_run_id: str,
    current_step: str,
    step_outputs: dict[str, LLMOutput],
    artifact_store: "ArtifactStore | None" = None,
) -> dict[str, Any]:
    """Build the Jinja2 template context dict for a pipeline step.

    The returned dict contains:
    - All fields named in context_template.include, resolved from NormalisedContext
    - pipeline_run_id, pipeline_name, current_step
    - steps.<name>.<field> references for all previously completed steps
    - A top-level `labels` dict for convenience (always present)
    - A top-level `team` field (owning team resolved from the auth token, or None)
    - An `artifacts` dict when artifact_store is provided, giving downstream steps
      access to artifact content via {{artifacts.step_name.key}}
    """
    ctx: dict[str, Any] = {
        "pipeline_run_id": pipeline_run_id,
        "pipeline_name": pipeline.name,
        "current_step": current_step,
        "labels": dict(normalised.labels),
        "team": normalised.team,
        **pipeline.vars,
    }

    # Resolve context_template.include fields from NormalisedContext
    for field_path in pipeline.context_template.include:
        value = _resolve_field(normalised, field_path)
        if value is not None:
            # Store under the leaf key (e.g. "labels.service" → ctx["service"])
            # and also under the full dotted path for explicit references
            leaf = field_path.split(".")[-1]
            ctx[leaf] = value
            if "." in field_path:
                ctx[field_path] = value

    # Inject prior step outputs under a nested `steps` key.
    # Jinja2 dot notation cannot traverse hyphenated keys (hyphens are subtraction),
    # so each step is stored under both its original name and a normalised version
    # with hyphens replaced by underscores.
    # Use {{steps.first_line_triage.field}} or {{steps['first-line-triage'].field}}.
    steps_ctx: dict[str, dict] = {}
    for step_name, output in step_outputs.items():
        dumped = output.model_dump(exclude={"raw_response"})
        steps_ctx[step_name] = dumped
        norm_name = step_name.replace("-", "_")
        if norm_name != step_name:
            steps_ctx[norm_name] = dumped

    ctx["steps"] = steps_ctx

    # Resolve artifact content for any prior steps that produced artifacts.
    # Content is loaded from the store and injected as {{artifacts.step_name.key}}.
    if artifact_store is not None:
        artifacts_ctx: dict[str, dict] = {}
        for step_name, output in step_outputs.items():
            step_artifacts = output.model_extra.get("artifacts") if output.model_extra else None
            if not isinstance(step_artifacts, dict) or not step_artifacts:
                continue
            resolved: dict[str, str] = {}
            for key, ref in step_artifacts.items():
                if not isinstance(ref, str):
                    continue
                try:
                    resolved[key] = await artifact_store.read(ref)
                except Exception as exc:
                    logger.warning("Could not read artifact %s/%s: %s", step_name, key, exc)
                    resolved[key] = f"[artifact unavailable: {key}]"
            if resolved:
                artifacts_ctx[step_name] = resolved
                norm_name = step_name.replace("-", "_")
                if norm_name != step_name:
                    artifacts_ctx[norm_name] = resolved

        ctx["artifacts"] = artifacts_ctx

    return ctx


def _resolve_field(normalised: NormalisedContext, field_path: str) -> Any:
    """Resolve a dotted field path against a NormalisedContext.

    Supports:
    - Top-level fields: "severity", "summary", "source", "pipeline"
    - Nested label access: "labels.service", "labels.environment"
    - Nested metadata access: "metadata.some_key"

    Returns None if the path cannot be resolved.
    """
    parts = field_path.split(".", 1)
    root = parts[0]

    top_level = {
        "severity": normalised.severity,
        "summary": normalised.summary,
        "source": normalised.source,
        "pipeline": normalised.pipeline,
        "received_at": normalised.received_at.isoformat(),
    }

    if root in top_level:
        if len(parts) == 1:
            return top_level[root]
        # Scalar top-level field with a sub-path — not resolvable
        logger.warning("Cannot resolve sub-path on scalar field: %s", field_path)
        return None

    if root == "labels":
        if len(parts) == 1:
            return dict(normalised.labels)
        return normalised.labels.get(parts[1])

    if root == "metadata":
        if len(parts) == 1:
            return dict(normalised.metadata)
        return normalised.metadata.get(parts[1])

    logger.warning("Unknown field path in context_template.include: %s", field_path)
    return None
