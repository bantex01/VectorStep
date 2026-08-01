import pytest
import yaml

from src.pipeline.loader import (
    _resolve_step_references,
    load_pipelines,
    load_pipelines_from_raw,
    load_step_library,
)

LIBRARY = {
    "sre-investigation": {
        "name": "sre-investigation",
        "description": "Grafana RED metrics investigation",
        "tags": ["investigation", "grafana"],
        "executor": "openclaw",
        "executor_config": {
            "agent": "sre-investigation",
            "session_key": "agent:sre-investigation:{{pipeline_run_id}}:{{current_step}}",
        },
        "confidence_threshold": 0.60,
        "prompt_template": "default prompt",
    }
}

LIBRARY_WITH_READINESS = {
    **LIBRARY,
    "step-with-readiness": {
        "name": "step-with-readiness",
        "executor": "gateway",
        "prompt_template": "x",
        "readiness": {
            "operational": {"min_runs": 20},
            "accuracy": {"min_accuracy": 0.9, "min_marked": 10},
        },
    },
}


# ----------------------------------------------------------------------
# _resolve_step_references
# ----------------------------------------------------------------------

def test_full_inheritance_with_no_overrides():
    resolved = _resolve_step_references([{"use": "sre-investigation"}], LIBRARY)

    step = resolved[0]
    assert step["name"] == "sre-investigation"
    assert step["confidence_threshold"] == 0.60
    assert step["executor"] == "openclaw"
    assert "use" not in step


def test_description_and_tags_are_stripped():
    resolved = _resolve_step_references([{"use": "sre-investigation"}], LIBRARY)

    step = resolved[0]
    assert "description" not in step
    assert "tags" not in step


def test_top_level_local_override_wins():
    resolved = _resolve_step_references(
        [{"use": "sre-investigation", "confidence_threshold": 0.80}], LIBRARY
    )

    step = resolved[0]
    assert step["confidence_threshold"] == 0.80
    # untouched fields still inherited from the library
    assert step["executor"] == "openclaw"
    assert step["prompt_template"] == "default prompt"


def test_executor_config_deep_merge_adds_new_key():
    resolved = _resolve_step_references(
        [{"use": "sre-investigation", "executor_config": {"model": "anthropic/claude-opus-4-8"}}],
        LIBRARY,
    )

    cfg = resolved[0]["executor_config"]
    assert cfg["model"] == "anthropic/claude-opus-4-8"
    # library keys preserved alongside the new key
    assert cfg["agent"] == "sre-investigation"
    assert cfg["session_key"] == "agent:sre-investigation:{{pipeline_run_id}}:{{current_step}}"


def test_executor_config_deep_merge_overrides_existing_key():
    resolved = _resolve_step_references(
        [{"use": "sre-investigation", "executor_config": {"agent": "overridden-agent"}}],
        LIBRARY,
    )

    cfg = resolved[0]["executor_config"]
    assert cfg["agent"] == "overridden-agent"
    # other library keys untouched
    assert cfg["session_key"] == "agent:sre-investigation:{{pipeline_run_id}}:{{current_step}}"


def test_unknown_library_step_raises_value_error():
    with pytest.raises(ValueError, match="unknown library step"):
        _resolve_step_references([{"use": "does-not-exist"}], LIBRARY)


def test_non_use_step_passes_through_unchanged():
    step = {"name": "inline-step", "executor": "webhook", "prompt_template": "hi"}

    resolved = _resolve_step_references([step], LIBRARY)

    assert resolved[0] == step


def test_parallel_group_inner_steps_resolved_recursively():
    steps = [
        {
            "parallel": {
                "name": "context-gathering",
                "join": "all_must_pass",
                "steps": [
                    {"use": "sre-investigation"},
                    {"name": "inline-branch", "executor": "gateway", "prompt_template": "hi"},
                ],
            }
        }
    ]

    resolved = _resolve_step_references(steps, LIBRARY)

    inner_steps = resolved[0]["parallel"]["steps"]
    assert inner_steps[0]["name"] == "sre-investigation"
    assert inner_steps[0]["confidence_threshold"] == 0.60
    assert inner_steps[1]["name"] == "inline-branch"


# ----------------------------------------------------------------------
# load_step_library
# ----------------------------------------------------------------------

def test_load_step_library_missing_directory_returns_empty(tmp_path):
    result = load_step_library(tmp_path / "does-not-exist")

    assert result == {}


def test_load_step_library_loads_yaml_files(tmp_path):
    (tmp_path / "my-step.yaml").write_text(
        yaml.dump(
            {
                "name": "my-step",
                "executor": "openclaw",
                "executor_config": {"agent": "test-agent"},
            }
        )
    )

    library = load_step_library(tmp_path)

    assert "my-step" in library
    assert library["my-step"]["executor"] == "openclaw"


# ----------------------------------------------------------------------
# load_pipelines
# ----------------------------------------------------------------------

def test_load_pipelines_resolves_use_references(tmp_path):
    (tmp_path / "p.yaml").write_text(
        yaml.dump(
            {
                "name": "test-pipeline",
                "trigger": {"match": {}},
                "steps": [{"use": "sre-investigation"}],
            }
        )
    )

    pipelines = load_pipelines(tmp_path, step_library=LIBRARY)

    assert len(pipelines) == 1
    assert pipelines[0].name == "test-pipeline"
    assert pipelines[0].steps[0].confidence_threshold == 0.60


def test_load_pipelines_missing_directory_raises(tmp_path):
    with pytest.raises(ValueError):
        load_pipelines(tmp_path / "missing")


# ----------------------------------------------------------------------
# _warn_correlated_critic_on_gated_steps (SPEC-verifier-semantics.md §6)
# ----------------------------------------------------------------------

def _write_pipeline(tmp_path, name: str, step: dict):
    (tmp_path / f"{name}.yaml").write_text(
        yaml.dump({
            "name": name,
            "trigger": {"match": {}},
            "steps": [step],
        })
    )


def _gated_step(mode: str) -> dict:
    return {
        "name": "investigate",
        "executor": "openclaw",
        "verifier": {"executor": "openclaw", "mode": mode},
        "deterministic_checks": [{"type": "shell", "name": "check", "run": "true"}],
    }


def test_critic_mode_on_gated_step_logs_advisory_nudge(tmp_path, caplog):
    _write_pipeline(tmp_path, "p", _gated_step("critic"))

    with caplog.at_level("INFO"):
        load_pipelines(tmp_path)

    assert "weaker corroboration signal" in caplog.text


def test_legacy_reviewer_alias_on_gated_step_logs_advisory_nudge(tmp_path, caplog):
    _write_pipeline(tmp_path, "p", _gated_step("reviewer"))

    with caplog.at_level("INFO"):
        load_pipelines(tmp_path)

    assert "weaker corroboration signal" in caplog.text


def test_independent_mode_on_gated_step_does_not_log_advisory_nudge(tmp_path, caplog):
    _write_pipeline(tmp_path, "p", _gated_step("independent"))

    with caplog.at_level("INFO"):
        load_pipelines(tmp_path)

    assert "weaker corroboration signal" not in caplog.text


def test_critic_mode_without_gate_does_not_log_advisory_nudge(tmp_path, caplog):
    step = {
        "name": "investigate",
        "executor": "openclaw",
        "verifier": {"executor": "openclaw", "mode": "critic"},
    }
    _write_pipeline(tmp_path, "p", step)

    with caplog.at_level("INFO"):
        load_pipelines(tmp_path)

    assert "weaker corroboration signal" not in caplog.text


# ----------------------------------------------------------------------
# readiness: — loader-level merge behaviour (SPEC-readiness-criteria.md §14)
# ----------------------------------------------------------------------

def test_library_readiness_with_local_override_replaces_whole_block():
    resolved = _resolve_step_references(
        [{"use": "step-with-readiness", "readiness": {"operational": {"min_runs": 5}}}],
        LIBRARY_WITH_READINESS,
    )
    step = resolved[0]
    # Whole-value replace, not a field merge — the library's accuracy tier is gone.
    assert step["readiness"] == {"operational": {"min_runs": 5}}


def test_library_step_readiness_wins_per_tier_conflict_with_pipeline_level():
    """Documented wart (§5): after 'use:' flattening, a library-provided readiness
    block is indistinguishable from a locally-written one, so when both the
    pipeline-level default and the library step configure the SAME tier, the
    library step's value wins — not a bug to fix here, but behaviour that must
    not silently drift into something different-but-also-undocumented."""
    from src.readiness import step_specs

    raw = {
        "name": "p",
        "trigger": {"match": {}},
        "readiness": {"operational": {"min_runs": 999}},   # pipeline-level house standard
        "steps": [{"use": "step-with-readiness"}],           # library step ALSO sets operational
    }
    pipelines = load_pipelines_from_raw({"p.yaml": yaml.dump(raw)}, step_library=LIBRARY_WITH_READINESS)
    spec = step_specs(pipelines[0])[0]

    assert spec.readiness.operational.min_runs == 20   # the library's value, not the pipeline's 999
    assert spec.readiness_source["operational"] == "step"
    # The accuracy tier, only set by the library step, still comes through untouched.
    assert spec.readiness.accuracy.min_accuracy == 0.9


def test_parallel_sibling_key_survives_resolution_with_nonempty_library():
    """Regression test for the §10a bug: `resolved.append({"parallel": inner})`
    used to silently discard every sibling key of `parallel:` — but only when a
    non-empty step library triggers _resolve_step_references at all."""
    steps = [
        {
            "parallel": {
                "name": "cross-checks",
                "readiness": {"operational": {"min_runs": 50}},
                "steps": [{"use": "sre-investigation"}],
            }
        }
    ]
    resolved = _resolve_step_references(steps, LIBRARY)
    assert resolved[0]["parallel"]["readiness"] == {"operational": {"min_runs": 50}}


# ----------------------------------------------------------------------
# _warn_readiness_misconfiguration (SPEC-readiness-criteria.md §10b)
# ----------------------------------------------------------------------

def test_warns_when_judgment_tier_on_non_llm_executor(tmp_path, caplog):
    step = {
        "name": "notify-oncall", "executor": "notify",
        "readiness": {"accuracy": {"min_accuracy": 0.9, "min_marked": 5}},
    }
    _write_pipeline(tmp_path, "p", step)

    with caplog.at_level("WARNING"):
        load_pipelines(tmp_path)

    assert "never writes effective_confidence" in caplog.text


def test_no_warning_for_operational_only_on_non_llm_executor(tmp_path, caplog):
    step = {
        "name": "notify-oncall", "executor": "notify",
        "readiness": {"operational": {"min_runs": 5}},
    }
    _write_pipeline(tmp_path, "p", step)

    with caplog.at_level("WARNING"):
        load_pipelines(tmp_path)

    assert "never writes effective_confidence" not in caplog.text


def test_warns_for_readiness_on_sub_pipeline_step(tmp_path, caplog):
    step = {
        "name": "child", "executor": "pipeline", "executor_config": {"pipeline_name": "other"},
        "readiness": {"operational": {"min_runs": 5}},
    }
    _write_pipeline(tmp_path, "p", step)

    with caplog.at_level("WARNING"):
        load_pipelines(tmp_path)

    assert "sub-pipeline step never writes" in caplog.text


def test_warns_for_wide_bin_width(tmp_path, caplog):
    step = {
        "name": "investigate", "executor": "gateway", "prompt_template": "x",
        "readiness": {"calibration": {"bin_width": 0.5}},
    }
    _write_pipeline(tmp_path, "p", step)

    with caplog.at_level("WARNING"):
        load_pipelines(tmp_path)

    assert "quantise the predicted score" in caplog.text


def test_warns_when_max_divergence_smaller_than_half_bin_width(tmp_path, caplog):
    step = {
        "name": "investigate", "executor": "gateway", "prompt_template": "x",
        "readiness": {"calibration": {"bin_width": 0.2, "max_divergence": 0.05}},
    }
    _write_pipeline(tmp_path, "p", step)

    with caplog.at_level("WARNING"):
        load_pipelines(tmp_path)

    assert "quantisation error" in caplog.text


def test_warns_on_duplicate_collapsed_bucket_names(tmp_path, caplog):
    raw = {
        "name": "p",
        "trigger": {"match": {}},
        "steps": [
            {"name": "x", "executor": "gateway", "prompt_template": "a"},
            {"parallel": {"name": "x", "steps": [
                {"name": "branch", "executor": "gateway", "prompt_template": "b"},
            ]}},
        ],
    }
    (tmp_path / "p.yaml").write_text(yaml.dump(raw))

    with caplog.at_level("WARNING"):
        load_pipelines(tmp_path)

    assert "collapse to the same" in caplog.text
