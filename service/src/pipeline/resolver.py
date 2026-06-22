import logging
import re
from typing import Any

from ..models.context import NormalisedContext
from ..models.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

_COMPARISON_OPS = {"gt", "gte", "lt", "lte"}


def resolve_pipeline(
    context: NormalisedContext,
    pipelines: list[PipelineConfig],
) -> PipelineConfig | None:
    """Match a NormalisedContext to the first pipeline whose trigger conditions all pass.

    If the context carries an explicit pipeline name (set by the source parser),
    that name is matched directly against loaded configs — trigger.match is skipped.

    Otherwise each pipeline's trigger.match block is evaluated in list order.
    All conditions must pass (AND logic). First match wins.

    Match keys are checked against:
      1. Top-level NormalisedContext fields (severity, source, summary, pipeline)
      2. context.labels (e.g. environment, service)
    """
    # Explicit pipeline name set by parser — bypass match logic
    if context.pipeline:
        for pipeline in pipelines:
            if pipeline.name == context.pipeline:
                logger.info(
                    "Pipeline resolved by explicit name: %s", pipeline.name
                )
                return pipeline
        logger.warning(
            "Explicit pipeline name '%s' not found in loaded configs", context.pipeline
        )
        return None

    # Match against trigger.match conditions
    for pipeline in pipelines:
        if _matches(context, pipeline):
            logger.info(
                "Pipeline resolved by trigger match: %s", pipeline.name
            )
            return pipeline

    logger.warning("No pipeline matched for context: source=%s severity=%s labels=%s",
                   context.source, context.severity, context.labels)
    return None


def _matches(context: NormalisedContext, pipeline: PipelineConfig) -> bool:
    """Return True if all trigger.match conditions are satisfied."""
    conditions = pipeline.trigger.match
    if not conditions:
        # Empty match block — matches everything; useful as a catch-all
        return True

    top_level = {
        "source": context.source,
        "severity": context.severity,
        "summary": context.summary,
        "pipeline": context.pipeline,
    }

    for key, expected in conditions.items():
        if key in top_level:
            actual = top_level[key]
        elif key in context.labels:
            actual = context.labels[key]
        else:
            # Condition references a field that doesn't exist on the context
            return False

        if not _evaluate_condition(actual, expected):
            return False

    return True


def _evaluate_condition(actual: Any, expected: Any) -> bool:
    """Evaluate a single match condition.

    `expected` is either a plain scalar (exact equality, the original behaviour)
    or a single-key operator dict, e.g. {"in": [...]}, {"regex": "..."}.
    """
    if not isinstance(expected, dict):
        return actual == expected

    if len(expected) != 1:
        logger.warning("Match operator dict must have exactly one key: %s", expected)
        return False

    op, value = next(iter(expected.items()))

    if op == "eq":
        return actual == value
    if op == "ne":
        return actual != value
    if op == "in":
        return actual in value
    if op == "not_in":
        return actual not in value
    if op == "regex":
        return actual is not None and re.search(value, str(actual)) is not None
    if op in _COMPARISON_OPS:
        return _compare(op, actual, value)

    logger.warning("Unknown match operator: %s", op)
    return False


def _compare(op: str, actual: Any, value: Any) -> bool:
    a, b = _as_number(actual), _as_number(value)
    if a is None or b is None:
        return False
    if op == "gt":
        return a > b
    if op == "gte":
        return a >= b
    if op == "lt":
        return a < b
    return a <= b  # lte


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
