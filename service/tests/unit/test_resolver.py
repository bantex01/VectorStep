from src.models.context import NormalisedContext
from src.models.pipeline import PipelineConfig, StepConfig, TriggerConfig
from src.pipeline.resolver import resolve_pipeline


def _pipeline(name: str, match: dict) -> PipelineConfig:
    return PipelineConfig(
        name=name,
        trigger=TriggerConfig(match=match),
        steps=[StepConfig(name="step1", executor="openclaw")],
    )


def _context(**overrides) -> NormalisedContext:
    defaults = dict(source="alertmanager", pipeline="", severity=None, labels={}, summary=None)
    defaults.update(overrides)
    return NormalisedContext(**defaults)


def test_explicit_pipeline_name_resolves_directly():
    pipelines = [_pipeline("a", {"severity": "critical"}), _pipeline("b", {})]
    ctx = _context(pipeline="b", severity="critical")

    result = resolve_pipeline(ctx, pipelines)

    assert result is not None
    assert result.name == "b"


def test_explicit_pipeline_name_not_found_returns_none():
    pipelines = [_pipeline("a", {})]
    ctx = _context(pipeline="missing")

    assert resolve_pipeline(ctx, pipelines) is None


def test_match_on_top_level_field():
    pipelines = [_pipeline("critical-path", {"severity": "critical"})]
    ctx = _context(severity="critical")

    result = resolve_pipeline(ctx, pipelines)

    assert result is not None
    assert result.name == "critical-path"


def test_match_on_label_field():
    pipelines = [_pipeline("prod-only", {"environment": "prod"})]
    ctx = _context(labels={"environment": "prod"})

    result = resolve_pipeline(ctx, pipelines)

    assert result is not None
    assert result.name == "prod-only"


def test_all_conditions_must_match_and_logic():
    pipelines = [_pipeline("prod-critical", {"severity": "critical", "environment": "prod"})]
    ctx = _context(severity="critical", labels={"environment": "staging"})

    assert resolve_pipeline(ctx, pipelines) is None


def test_match_key_referencing_unknown_field_does_not_match():
    pipelines = [_pipeline("p", {"nonexistent_field": "value"})]
    ctx = _context(severity="critical")

    assert resolve_pipeline(ctx, pipelines) is None


def test_empty_match_block_is_catch_all():
    pipelines = [
        _pipeline("specific", {"severity": "critical"}),
        _pipeline("catchall", {}),
    ]
    ctx = _context(severity="warning")

    result = resolve_pipeline(ctx, pipelines)

    assert result is not None
    assert result.name == "catchall"


def test_first_match_wins():
    pipelines = [
        _pipeline("first", {"severity": "critical"}),
        _pipeline("second", {"severity": "critical"}),
    ]
    ctx = _context(severity="critical")

    result = resolve_pipeline(ctx, pipelines)

    assert result is not None
    assert result.name == "first"


def test_no_pipeline_matches_returns_none():
    pipelines = [_pipeline("p", {"severity": "critical"})]
    ctx = _context(severity="warning")

    assert resolve_pipeline(ctx, pipelines) is None
