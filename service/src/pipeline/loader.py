import logging
from pathlib import Path

import yaml

from ..models.pipeline import LibraryStepConfig, PipelineConfig, StepConfig

logger = logging.getLogger(__name__)


def _warn_correlated_critic_on_gated_steps(pipeline: PipelineConfig) -> None:
    """Advisory only — CONFIDENCE-REDESIGN.md §2.3: a 'critic' verifier's agreement
    correlates with the primary's own errors and carries little signal, so it's a weak
    corroboration source for a step that already opted into the harder trust-vector gate
    (grounding.enforce or deterministic_checks:). This never blocks a load or changes any
    gate — it's a one-time, load-time nudge, logged once per step, exactly like the
    'Loaded pipeline config' line beside it."""
    for step in pipeline.steps:
        if not isinstance(step, StepConfig):
            continue  # parallel/fan-out groups don't carry a `grounding`/`deterministic_checks` shape today
        is_gated = (step.grounding is not None and step.grounding.enforce) or bool(step.deterministic_checks)
        if is_gated and step.verifier is not None and step.verifier.mode == "critic":
            logger.info(
                "Pipeline '%s' step '%s' uses verifier mode=critic on a step with an "
                "enforced trust gate (grounding.enforce or deterministic_checks) — a "
                "critic's agreement correlates with the primary's own errors, so it's a "
                "weaker corroboration signal here. Consider mode: independent for gates "
                "that authorise side effects (CONFIDENCE-REDESIGN.md §2.3).",
                pipeline.name, step.name,
            )


def load_step_library_from_raw(
    raw_by_filename: dict[str, str],
    log_success: bool = True,
) -> dict[str, dict]:
    """Core of load_step_library, decoupled from the filesystem.

    Takes {filename: yaml_text} instead of scanning a directory, so a caller
    that wants to validate a *candidate* directory state — the real files
    plus one new/changed entry, without writing it to disk first — can build
    that dict and pass it straight through (see main.py's atomic-validated-
    write path, SPEC-pork-service-mcp.md §2.9/§5c). Raises on the first
    invalid file, same as load_step_library.
    """
    library: dict[str, dict] = {}
    for filename in sorted(raw_by_filename):
        content = raw_by_filename[filename]
        try:
            raw = yaml.safe_load(content)
            if not isinstance(raw, dict):
                logger.warning("Step library file %s is not a YAML mapping — skipped", filename)
                continue
            step = LibraryStepConfig.model_validate(raw)
            library[step.name] = raw
            if log_success:
                logger.info("Loaded library step: %s (from %s)", step.name, filename)
        except Exception as exc:
            logger.error("Failed to load step library file %s: %s", filename, exc)
            raise
    return library


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

    raw_by_filename = {path.name: path.read_text() for path in steps_dir.glob("*.yaml")}
    library = load_step_library_from_raw(raw_by_filename)
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


# Public name for callers outside this module (e.g. config_writer.py's
# validate-only dry run, which needs to resolve 'use:' refs the same way
# load_pipelines_from_raw does without going through the full loader).
resolve_step_references = _resolve_step_references


def load_pipelines_from_raw(
    raw_by_filename: dict[str, str],
    step_library: dict[str, dict] | None = None,
    log_success: bool = True,
) -> list[PipelineConfig]:
    """Core of load_pipelines, decoupled from the filesystem — see
    load_step_library_from_raw's docstring for why (candidate-directory
    validation ahead of an atomic write, without touching disk first).
    Raises on the first invalid file, same as load_pipelines."""
    pipelines: list[PipelineConfig] = []
    for filename in sorted(raw_by_filename):
        content = raw_by_filename[filename]
        try:
            raw = yaml.safe_load(content)
            if step_library and isinstance(raw.get("steps"), list):
                raw["steps"] = _resolve_step_references(raw["steps"], step_library)
            pipeline = PipelineConfig.model_validate(raw)
            _warn_correlated_critic_on_gated_steps(pipeline)
            pipelines.append(pipeline)
            if log_success:
                logger.info("Loaded pipeline config: %s (from %s)", pipeline.name, filename)
        except Exception as exc:
            logger.error("Failed to load pipeline config %s: %s", filename, exc)
            raise
    return pipelines


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

    raw_by_filename = {path.name: path.read_text() for path in config_dir.glob("*.yaml")}
    pipelines = load_pipelines_from_raw(raw_by_filename, step_library=step_library)
    logger.info("Loaded %d pipeline config(s) from %s", len(pipelines), config_dir)
    return pipelines
