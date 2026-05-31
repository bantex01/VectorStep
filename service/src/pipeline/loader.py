import logging
from pathlib import Path

import yaml

from ..models.pipeline import LibraryStepConfig, PipelineConfig

logger = logging.getLogger(__name__)


def load_step_library(steps_dir: str | Path) -> dict[str, dict]:
    """Load all YAML step definitions from the library directory.

    Returns a mapping of step name → raw dict. Each file is validated against
    LibraryStepConfig so errors surface at startup rather than at run time.
    The raw dict (not the model) is stored so description/tags are available
    to the UI, and merging into pipeline steps is a plain dict operation.
    """
    steps_dir = Path(steps_dir)
    if not steps_dir.is_dir():
        logger.debug("Step library directory not found: %s — library disabled", steps_dir)
        return {}

    library: dict[str, dict] = {}
    for path in sorted(steps_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text())
            if not isinstance(raw, dict):
                logger.warning("Step library file %s is not a YAML mapping — skipped", path.name)
                continue
            step = LibraryStepConfig.model_validate(raw)
            library[step.name] = raw
            logger.info("Loaded library step: %s (from %s)", step.name, path.name)
        except Exception as exc:
            logger.error("Failed to load step library file %s: %s", path.name, exc)
            raise

    logger.info("Loaded %d library step(s) from %s", len(library), steps_dir)
    return library


def _resolve_step_references(raw_steps: list, library: dict[str, dict]) -> list:
    """Resolve 'use:' references in a step list against the step library.

    For each step dict containing a 'use:' key:
      - The named library step is used as the base config.
      - Any other fields defined locally override the library values.
      - executor_config is deep-merged so individual keys (e.g. model) can be
        added or overridden without replacing the whole block.
      - description and tags are stripped — they are library-only metadata.

    Parallel group inner steps are resolved recursively. Non-'use:' steps
    pass through unchanged.
    """
    resolved = []
    for step in raw_steps:
        if not isinstance(step, dict):
            resolved.append(step)
            continue

        # Parallel group — resolve inner steps recursively
        if "parallel" in step and isinstance(step["parallel"], dict):
            inner = step["parallel"]
            if "steps" in inner:
                inner = {**inner, "steps": _resolve_step_references(inner["steps"], library)}
            resolved.append({"parallel": inner})
            continue

        use = step.get("use")
        if not use:
            resolved.append(step)
            continue

        lib_step = library.get(use)
        if lib_step is None:
            available = sorted(library.keys())
            raise ValueError(
                f"Step references unknown library step 'use: {use}'. "
                f"Available: {available or '(library is empty)'}"
            )

        # Start from library base, stripping UI-only metadata fields
        merged = {k: v for k, v in lib_step.items() if k not in ("description", "tags")}

        # Apply local overrides
        for key, value in step.items():
            if key == "use":
                continue
            if (key == "executor_config"
                    and key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)):
                # Deep merge executor_config so callers can add model/thinking_level
                # without repeating the full agent/session_key block
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        resolved.append(merged)

    return resolved


def load_pipelines(
    config_dir: str | Path,
    step_library: dict[str, dict] | None = None,
) -> list[PipelineConfig]:
    """Load and validate all YAML pipeline configs from a directory.

    If step_library is provided, 'use:' references in step lists are resolved
    against it before Pydantic validation.

    Files are returned in filesystem order (alphabetical). Callers that need
    priority ordering should name files accordingly (e.g. 10-critical.yaml,
    20-warning.yaml) or sort the returned list themselves.
    """
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise ValueError(f"Pipeline config directory not found: {config_dir}")

    pipelines: list[PipelineConfig] = []
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text())
            if step_library and isinstance(raw.get("steps"), list):
                raw["steps"] = _resolve_step_references(raw["steps"], step_library)
            pipeline = PipelineConfig.model_validate(raw)
            pipelines.append(pipeline)
            logger.info("Loaded pipeline config: %s (from %s)", pipeline.name, path.name)
        except Exception as exc:
            logger.error("Failed to load pipeline config %s: %s", path.name, exc)
            raise

    logger.info("Loaded %d pipeline config(s) from %s", len(pipelines), config_dir)
    return pipelines
