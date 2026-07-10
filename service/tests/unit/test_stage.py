import pytest
from pydantic import ValidationError

from src.models.pipeline import PipelineConfig, TriggerConfig


def _pipeline(**overrides):
    return PipelineConfig(name="p", trigger=TriggerConfig(), steps=[], **overrides)


def test_stage_defaults_to_testing_when_omitted():
    pipeline = _pipeline()
    assert pipeline.stage == "testing"


def test_stage_explicit_production():
    pipeline = _pipeline(stage="production")
    assert pipeline.stage == "production"


def test_stage_explicit_testing():
    pipeline = _pipeline(stage="testing")
    assert pipeline.stage == "testing"


def test_stage_rejects_bogus_value():
    with pytest.raises(ValidationError):
        _pipeline(stage="staging")
