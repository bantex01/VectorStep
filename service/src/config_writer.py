"""Atomic validated-write path for pipeline/step-library YAML configs
(SPEC-vectorstep-service-mcp.md §2.9/§5c). This is the ONE place that writes to
service/pipelines or service/steps for the create/update/delete JSON
endpoints — a single tested write path shared by any future caller (the MCP,
or a future UI editor).

Every write validates a *candidate* view of the directory — the real files
plus the one new/changed/removed entry, built in memory via
pipeline.loader's *_from_raw functions — before anything touches disk. Only
if that candidate reload succeeds does the real file get written (via a
temp-file + os.replace, so a crash mid-write can't leave a half-written
file). This is what catches a step-library change that would break some
*other* pipeline's 'use:' resolution, per §9's rollback test requirement.
"""

import os
import uuid
from dataclasses import dataclass, field

import yaml
from pydantic import ValidationError

from .models.pipeline import LibraryStepConfig, PipelineConfig
from .pipeline.loader import load_pipelines_from_raw, load_step_library_from_raw, resolve_step_references


@dataclass
class WriteResult:
    ok: bool
    config: dict | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_detail: dict = field(default_factory=dict)


def _read_all_yaml(dir_path: str) -> dict[str, str]:
    if not os.path.isdir(dir_path):
        return {}
    result = {}
    for fname in os.listdir(dir_path):
        if fname.endswith(".yaml"):
            with open(os.path.join(dir_path, fname)) as f:
                result[fname] = f.read()
    return result


def _parse_named_yaml(yaml_text: str, kind: str) -> tuple[dict | None, WriteResult | None]:
    """Parse yaml_text and require a 'name:' field. Returns (parsed, None) on
    success, or (None, WriteResult) with a validation error on failure."""
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return None, WriteResult(ok=False, error_type="validation", error_message=f"Invalid YAML: {exc}")
    if not isinstance(parsed, dict) or not parsed.get("name"):
        return None, WriteResult(
            ok=False, error_type="validation",
            error_message=f"{kind} YAML must be a mapping with a 'name' field",
        )
    return parsed, None


def _atomic_write(dir_path: str, final_name: str, content: str) -> None:
    tmp_path = os.path.join(dir_path, f".{final_name}.tmp-{uuid.uuid4().hex}")
    with open(tmp_path, "w") as f:
        f.write(content)
    os.replace(tmp_path, os.path.join(dir_path, final_name))


def _validation_errors_from_pydantic(exc: ValidationError) -> list[dict]:
    return [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]


# ── Pipelines ──────────────────────────────────────────────────────────────

def write_pipeline_yaml(
    pipeline_dir: str,
    yaml_text: str,
    step_library: dict[str, dict],
    *,
    url_name: str | None,
    overwrite: bool,
) -> WriteResult:
    """Validate and atomically write a pipeline YAML. url_name is the path
    {name} for an update (PUT) — must match the YAML's own name: field — or
    None for a create (POST), where the name is derived from the YAML."""
    parsed, err = _parse_named_yaml(yaml_text, "Pipeline")
    if err:
        return err
    name = parsed["name"]

    if url_name is not None:
        if not os.path.exists(os.path.join(pipeline_dir, f"{url_name}.yaml")):
            return WriteResult(ok=False, error_type="not_found", error_message=f"Pipeline '{url_name}' not found")
        if url_name != name:
            return WriteResult(
                ok=False, error_type="validation",
                error_message=f"URL name '{url_name}' does not match the YAML's name '{name}'",
            )

    filename = f"{name}.yaml"
    if url_name is None and os.path.exists(os.path.join(pipeline_dir, filename)) and not overwrite:
        return WriteResult(ok=False, error_type="collision", error_message=f"Pipeline '{name}' already exists")

    candidate = _read_all_yaml(pipeline_dir)
    candidate[filename] = yaml_text
    try:
        load_pipelines_from_raw(candidate, step_library=step_library, log_success=False)
    except ValidationError as exc:
        return WriteResult(ok=False, error_type="validation", error_message=str(exc),
                            error_detail={"errors": _validation_errors_from_pydantic(exc)})
    except Exception as exc:
        return WriteResult(ok=False, error_type="reload_failed", error_message=f"Reload failed: {exc}")

    _atomic_write(pipeline_dir, filename, yaml_text)
    return WriteResult(ok=True, config=parsed)


def delete_pipeline_yaml(pipeline_dir: str, name: str) -> WriteResult:
    """Delete a pipeline's YAML, returning the prior content for audit. No
    candidate-reload check is needed: pipelines don't reference each other,
    so removing one can't break another's resolution."""
    path = os.path.join(pipeline_dir, f"{name}.yaml")
    if not os.path.exists(path):
        return WriteResult(ok=False, error_type="not_found", error_message=f"Pipeline '{name}' not found")
    with open(path) as f:
        prior_yaml = f.read()
    os.remove(path)
    return WriteResult(ok=True, config={"yaml": prior_yaml})


# ── Step library ───────────────────────────────────────────────────────────

def write_step_yaml(
    step_library_dir: str,
    pipeline_dir: str,
    yaml_text: str,
    *,
    url_name: str | None,
    overwrite: bool,
) -> WriteResult:
    """Validate and atomically write a step-library YAML. Also re-validates
    every pipeline against the candidate step library, since a step change
    can break another pipeline's 'use:' resolution — this is the cross-file
    risk §2.9/§9 call out explicitly."""
    parsed, err = _parse_named_yaml(yaml_text, "Step")
    if err:
        return err
    name = parsed["name"]

    if url_name is not None:
        if not os.path.exists(os.path.join(step_library_dir, f"{url_name}.yaml")):
            return WriteResult(ok=False, error_type="not_found", error_message=f"Step '{url_name}' not found")
        if url_name != name:
            return WriteResult(
                ok=False, error_type="validation",
                error_message=f"URL name '{url_name}' does not match the YAML's name '{name}'",
            )

    filename = f"{name}.yaml"
    if url_name is None and os.path.exists(os.path.join(step_library_dir, filename)) and not overwrite:
        return WriteResult(ok=False, error_type="collision", error_message=f"Step '{name}' already exists")

    candidate_steps = _read_all_yaml(step_library_dir)
    candidate_steps[filename] = yaml_text
    try:
        candidate_library = load_step_library_from_raw(candidate_steps, log_success=False)
    except ValidationError as exc:
        return WriteResult(ok=False, error_type="validation", error_message=str(exc),
                            error_detail={"errors": _validation_errors_from_pydantic(exc)})
    except Exception as exc:
        return WriteResult(ok=False, error_type="reload_failed", error_message=f"Reload failed: {exc}")

    candidate_pipelines = _read_all_yaml(pipeline_dir)
    try:
        load_pipelines_from_raw(candidate_pipelines, step_library=candidate_library, log_success=False)
    except Exception as exc:
        return WriteResult(
            ok=False, error_type="reload_failed",
            error_message=f"Step '{name}' would break pipeline resolution: {exc}",
        )

    _atomic_write(step_library_dir, filename, yaml_text)
    return WriteResult(ok=True, config=parsed)


def delete_step_yaml(step_library_dir: str, pipeline_dir: str, name: str) -> WriteResult:
    """Delete a step-library YAML, returning the prior content for audit.
    Refuses (reload_failed) if any pipeline still references this step via
    'use:' — removing it would break that pipeline's resolution on the next
    reload."""
    path = os.path.join(step_library_dir, f"{name}.yaml")
    if not os.path.exists(path):
        return WriteResult(ok=False, error_type="not_found", error_message=f"Step '{name}' not found")
    with open(path) as f:
        prior_yaml = f.read()

    candidate_steps = _read_all_yaml(step_library_dir)
    del candidate_steps[f"{name}.yaml"]
    candidate_library = load_step_library_from_raw(candidate_steps, log_success=False)

    candidate_pipelines = _read_all_yaml(pipeline_dir)
    try:
        load_pipelines_from_raw(candidate_pipelines, step_library=candidate_library, log_success=False)
    except Exception as exc:
        return WriteResult(
            ok=False, error_type="reload_failed",
            error_message=f"Cannot delete step '{name}': still referenced by a pipeline ({exc})",
        )

    os.remove(path)
    return WriteResult(ok=True, config={"yaml": prior_yaml})


# ── Validate-only (dry run, no write) ────────────────────────────────────────

def validate_pipeline_yaml(yaml_text: str, step_library: dict[str, dict]) -> tuple[bool, list[dict]]:
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return False, [{"loc": [], "msg": f"Invalid YAML: {exc}"}]
    if not isinstance(parsed, dict):
        return False, [{"loc": [], "msg": "YAML must be a mapping"}]
    try:
        if step_library and isinstance(parsed.get("steps"), list):
            parsed = {**parsed, "steps": resolve_step_references(parsed["steps"], step_library)}
        PipelineConfig.model_validate(parsed)
        return True, []
    except ValidationError as exc:
        return False, _validation_errors_from_pydantic(exc)
    except Exception as exc:
        return False, [{"loc": [], "msg": str(exc)}]


def validate_step_yaml(yaml_text: str) -> tuple[bool, list[dict]]:
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return False, [{"loc": [], "msg": f"Invalid YAML: {exc}"}]
    if not isinstance(parsed, dict):
        return False, [{"loc": [], "msg": "YAML must be a mapping"}]
    try:
        LibraryStepConfig.model_validate(parsed)
        return True, []
    except ValidationError as exc:
        return False, _validation_errors_from_pydantic(exc)
