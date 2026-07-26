"""Tests for scripts/backfill_prompt_versions.py's pure-logic helpers
(SPEC-prompt-versioning.md §4i) — resolving a runtime PipelineStep.step_name back
to the template that currently produces it, across plain steps, parallel branches,
and fan-out branches (whose runtime names don't match their config name 1:1)."""
from scripts.backfill_prompt_versions import _build_template_index, _resolve_template
from src.models.pipeline import (
    FanOutConfig,
    FanOutGroupConfig,
    ParallelGroupConfig,
    ParallelGroupInner,
    ParallelStepConfig,
    PipelineConfig,
    StepConfig,
    TriggerConfig,
)


def _pipeline(name: str, steps: list) -> PipelineConfig:
    return PipelineConfig(name=name, trigger=TriggerConfig(), steps=steps)


class TestBuildTemplateIndex:
    def test_plain_step_indexed_by_name(self):
        step = StepConfig(name="investigate", executor="gateway", prompt_template="Investigate this.")
        by_name, fanout_by_group = _build_template_index([_pipeline("p", [step])])

        assert by_name["investigate"] == "Investigate this."
        assert fanout_by_group == {}

    def test_parallel_branch_indexed_by_group_slash_branch(self):
        group = ParallelGroupConfig(parallel=ParallelGroupInner(
            name="context-gather",
            steps=[
                ParallelStepConfig(name="jira-history", executor="gateway", prompt_template="Search Jira."),
                ParallelStepConfig(name="quick-assessment", executor="gateway", prompt_template="Assess quickly."),
            ],
        ))
        by_name, fanout_by_group = _build_template_index([_pipeline("p", [group])])

        assert by_name["context-gather/jira-history"] == "Search Jira."
        assert by_name["context-gather/quick-assessment"] == "Assess quickly."

    def test_fan_out_indexed_by_group_name_not_branch(self):
        fan_out = FanOutGroupConfig(fan_out=FanOutConfig(
            name="per-service", over="services", executor="gateway",
            prompt_template="Investigate {{item}}.",
        ))
        by_name, fanout_by_group = _build_template_index([_pipeline("p", [fan_out])])

        assert fanout_by_group["per-service"] == "Investigate {{item}}."
        assert by_name == {}

    def test_first_pipeline_wins_on_name_collision_across_pipelines(self):
        step_a = StepConfig(name="notify", executor="gateway", prompt_template="From pipeline A.")
        step_b = StepConfig(name="notify", executor="gateway", prompt_template="From pipeline B.")
        by_name, _ = _build_template_index([_pipeline("a", [step_a]), _pipeline("b", [step_b])])

        assert by_name["notify"] == "From pipeline A."


class TestResolveTemplate:
    def test_exact_match(self):
        by_name = {"investigate": "Investigate this."}
        assert _resolve_template("investigate", by_name, {}) == "Investigate this."

    def test_parallel_branch_exact_match(self):
        by_name = {"context-gather/jira-history": "Search Jira."}
        assert _resolve_template("context-gather/jira-history", by_name, {}) == "Search Jira."

    def test_fan_out_branch_resolves_via_group_prefix(self):
        fanout_by_group = {"per-service": "Investigate {{item}}."}
        assert _resolve_template("per-service/0", {}, fanout_by_group) == "Investigate {{item}}."
        assert _resolve_template("per-service/17", {}, fanout_by_group) == "Investigate {{item}}."

    def test_unknown_step_name_returns_none(self):
        assert _resolve_template("deleted-step", {"investigate": "t"}, {}) is None

    def test_unknown_group_prefix_returns_none(self):
        assert _resolve_template("deleted-group/0", {}, {"per-service": "t"}) is None
