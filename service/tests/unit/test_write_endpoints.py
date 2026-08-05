"""Tests for the write/validate/delete JSON endpoints added for the VectorStep
Service MCP (SPEC-vectorstep-service-mcp.md §5c): create/update/delete for
pipelines and step-library entries, the /validate dry-run endpoints, and the
atomic validated-rollback write path (§2.9) — including the cross-file
rollback scenario where deleting a still-referenced step must leave the live
config untouched."""
import os

import httpx
import pytest

import src.main as main
from src.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _pipeline_and_step_dirs(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "pipelines"
    step_dir = tmp_path / "steps"
    pipeline_dir.mkdir()
    step_dir.mkdir()
    monkeypatch.setattr(main, "_pipeline_dir", str(pipeline_dir))
    monkeypatch.setattr(main, "_step_library_dir", str(step_dir))
    monkeypatch.setattr(main, "_pipelines", [])
    monkeypatch.setattr(main, "_step_library", {})
    return pipeline_dir, step_dir


_VALID_PIPELINE_YAML = """\
name: my-pipeline
trigger:
  match: {}
steps:
  - name: s
    executor: openclaw
"""

_VALID_STEP_YAML = """\
name: my-step
executor: openclaw
"""


# ---------------------------------------------------------------------------
# POST /pipelines
# ---------------------------------------------------------------------------

async def test_create_pipeline_writes_file_and_reloads(client, _pipeline_and_step_dirs):
    pipeline_dir, _ = _pipeline_and_step_dirs

    resp = await client.post("/pipelines", json={"yaml": _VALID_PIPELINE_YAML})

    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["name"] == "my-pipeline"
    assert body["committed"] is False
    assert (pipeline_dir / "my-pipeline.yaml").read_text() == _VALID_PIPELINE_YAML
    assert any(p.name == "my-pipeline" for p in main._pipelines)


async def test_create_pipeline_collision_without_overwrite_returns_409(client):
    await client.post("/pipelines", json={"yaml": _VALID_PIPELINE_YAML})

    resp = await client.post("/pipelines", json={"yaml": _VALID_PIPELINE_YAML})

    assert resp.status_code == 409


async def test_create_pipeline_with_overwrite_succeeds(client):
    await client.post("/pipelines", json={"yaml": _VALID_PIPELINE_YAML})

    changed = _VALID_PIPELINE_YAML.replace("name: s", "name: s2")
    resp = await client.post("/pipelines", json={"yaml": changed, "overwrite": True})

    assert resp.status_code == 200


async def test_create_pipeline_invalid_yaml_returns_400(client):
    resp = await client.post("/pipelines", json={"yaml": "not: valid: yaml: at: all:"})

    assert resp.status_code == 400


async def test_create_pipeline_missing_name_returns_400(client):
    resp = await client.post("/pipelines", json={"yaml": "trigger:\n  match: {}\nsteps: []\n"})

    assert resp.status_code == 400


async def test_create_pipeline_missing_required_field_returns_400(client):
    # No 'steps:' — required by PipelineConfig.
    resp = await client.post("/pipelines", json={"yaml": "name: p\ntrigger:\n  match: {}\n"})

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# PUT /pipelines/{name}
# ---------------------------------------------------------------------------

async def test_update_pipeline_success(client, _pipeline_and_step_dirs):
    pipeline_dir, _ = _pipeline_and_step_dirs
    await client.post("/pipelines", json={"yaml": _VALID_PIPELINE_YAML})

    updated = _VALID_PIPELINE_YAML.replace("description", "description") + "  # comment\n"
    updated = _VALID_PIPELINE_YAML.replace("name: s", "name: renamed-step")
    resp = await client.put("/pipelines/my-pipeline", json={"yaml": updated})

    assert resp.status_code == 200
    assert "renamed-step" in (pipeline_dir / "my-pipeline.yaml").read_text()


async def test_update_pipeline_unknown_returns_404(client):
    resp = await client.put("/pipelines/does-not-exist", json={"yaml": _VALID_PIPELINE_YAML})

    assert resp.status_code == 404


async def test_update_pipeline_name_mismatch_returns_400(client):
    # The URL name must refer to an existing pipeline (a rename target that
    # doesn't exist yet is a 404, not a mismatch) — so create it first under
    # its real name, then PUT to that same URL with YAML claiming a
    # different name.
    await client.post("/pipelines", json={"yaml": _VALID_PIPELINE_YAML})
    renamed = _VALID_PIPELINE_YAML.replace("name: my-pipeline", "name: a-different-name")

    resp = await client.put("/pipelines/my-pipeline", json={"yaml": renamed})

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /pipelines/{name}
# ---------------------------------------------------------------------------

async def test_delete_pipeline_returns_prior_yaml_and_removes_file(client, _pipeline_and_step_dirs):
    pipeline_dir, _ = _pipeline_and_step_dirs
    await client.post("/pipelines", json={"yaml": _VALID_PIPELINE_YAML})

    resp = await client.delete("/pipelines/my-pipeline")

    assert resp.status_code == 200
    body = resp.json()
    assert body["yaml"] == _VALID_PIPELINE_YAML
    assert not (pipeline_dir / "my-pipeline.yaml").exists()
    assert not any(p.name == "my-pipeline" for p in main._pipelines)


async def test_delete_pipeline_unknown_returns_404(client):
    resp = await client.delete("/pipelines/does-not-exist")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /pipelines/validate — dry run, no write
# ---------------------------------------------------------------------------

async def test_validate_pipeline_valid_yaml_writes_nothing(client, _pipeline_and_step_dirs):
    pipeline_dir, _ = _pipeline_and_step_dirs

    resp = await client.post("/pipelines/validate", json={"yaml": _VALID_PIPELINE_YAML})

    assert resp.status_code == 200
    assert resp.json() == {"valid": True, "errors": []}
    assert list(pipeline_dir.iterdir()) == []


async def test_validate_pipeline_invalid_yaml_returns_structured_errors(client):
    resp = await client.post("/pipelines/validate", json={"yaml": "name: p\n"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert len(body["errors"]) > 0
    assert all({"loc", "msg"} <= set(e) for e in body["errors"])


# ---------------------------------------------------------------------------
# Steps — create/update/delete/validate
# ---------------------------------------------------------------------------

async def test_create_step_writes_to_gitignored_dir(client, _pipeline_and_step_dirs):
    _, step_dir = _pipeline_and_step_dirs

    resp = await client.post("/steps", json={"yaml": _VALID_STEP_YAML})

    assert resp.status_code == 200
    body = resp.json()
    assert "gitignored" in body["note"]
    assert (step_dir / "my-step.yaml").read_text() == _VALID_STEP_YAML
    assert "my-step" in main._step_library


async def test_create_step_collision_without_overwrite_returns_409(client):
    await client.post("/steps", json={"yaml": _VALID_STEP_YAML})

    resp = await client.post("/steps", json={"yaml": _VALID_STEP_YAML})

    assert resp.status_code == 409


async def test_update_step_unknown_returns_404(client):
    resp = await client.put("/steps/does-not-exist", json={"yaml": _VALID_STEP_YAML})

    assert resp.status_code == 404


async def test_delete_step_returns_prior_yaml(client, _pipeline_and_step_dirs):
    _, step_dir = _pipeline_and_step_dirs
    await client.post("/steps", json={"yaml": _VALID_STEP_YAML})

    resp = await client.delete("/steps/my-step")

    assert resp.status_code == 200
    assert resp.json()["yaml"] == _VALID_STEP_YAML
    assert not (step_dir / "my-step.yaml").exists()


async def test_validate_step_invalid_returns_structured_errors(client):
    resp = await client.post("/steps/validate", json={"yaml": "name: s\n"})  # missing required executor

    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert len(body["errors"]) > 0


# ---------------------------------------------------------------------------
# Rollback guarantee (§2.9/§9): deleting a step still referenced by a
# pipeline must fail with reload_failed and leave the live config untouched.
# ---------------------------------------------------------------------------

async def test_delete_step_still_in_use_by_pipeline_is_refused(client, _pipeline_and_step_dirs):
    pipeline_dir, step_dir = _pipeline_and_step_dirs

    await client.post("/steps", json={"yaml": _VALID_STEP_YAML})
    consumer_yaml = (
        "name: consumer\n"
        "trigger:\n  match: {}\n"
        "steps:\n  - use: my-step\n"
    )
    resp = await client.post("/pipelines", json={"yaml": consumer_yaml})
    assert resp.status_code == 200, resp.text

    resp = await client.delete("/steps/my-step")

    assert resp.status_code == 500
    assert resp.json()["detail"]["type"] == "reload_failed"
    # The file must still be on disk and the live registries untouched.
    assert (step_dir / "my-step.yaml").exists()
    assert "my-step" in main._step_library
    assert any(p.name == "consumer" for p in main._pipelines)


async def test_update_pipeline_breaking_own_resolution_leaves_other_pipelines_untouched(client, _pipeline_and_step_dirs):
    """A candidate write that fails its own resolution (references a
    nonexistent library step) must be rejected without touching the live
    registry — including any other, unrelated pipeline already loaded."""
    # A non-empty step library is required for 'use:' resolution to actually
    # run (see loader.py's `if step_library and ...` guard) — otherwise a
    # dangling 'use:' just falls through to a generic Pydantic "field
    # required" error (a validation 400, not the reload_failed path this
    # test targets).
    await client.post("/steps", json={"yaml": _VALID_STEP_YAML})
    await client.post("/pipelines", json={"yaml": _VALID_PIPELINE_YAML})

    broken_yaml = (
        "name: my-pipeline\n"
        "trigger:\n  match: {}\n"
        "steps:\n  - use: does-not-exist\n"
    )
    resp = await client.put("/pipelines/my-pipeline", json={"yaml": broken_yaml})

    assert resp.status_code == 500
    assert resp.json()["detail"]["type"] == "reload_failed"
    # Live config still has the original, working version.
    live = next(p.name for p in main._pipelines if p.name == "my-pipeline")
    assert live == "my-pipeline"
    assert main._pipelines[0].steps[0].name == "s"


# ---------------------------------------------------------------------------
# Secrets: ${VAR} placeholders preserved verbatim (§8)
# ---------------------------------------------------------------------------

async def test_secret_placeholder_preserved_verbatim(client, _pipeline_and_step_dirs):
    pipeline_dir, _ = _pipeline_and_step_dirs
    yaml_with_secret = (
        "name: with-secret\n"
        "trigger:\n  match: {}\n"
        "steps:\n"
        "  - name: s\n"
        "    executor: webhook\n"
        "    executor_config:\n"
        "      url: '${WEBHOOK_URL}'\n"
        "      token: '${API_TOKEN}'\n"
    )

    resp = await client.post("/pipelines", json={"yaml": yaml_with_secret})

    assert resp.status_code == 200
    on_disk = (pipeline_dir / "with-secret.yaml").read_text()
    assert "${WEBHOOK_URL}" in on_disk
    assert "${API_TOKEN}" in on_disk
