from src.models.pipeline import StepConfig, VerifierConfig, VerifierTriggerConfig
from src.pipeline.runner import PipelineRunner


def _runner() -> PipelineRunner:
    return PipelineRunner(executors={})


def _step_with_verifier(combination_strategy: str = "minimum", veto_floor: float = 0.60) -> StepConfig:
    return StepConfig(
        name="step",
        executor="openclaw",
        verifier=VerifierConfig(
            executor="openclaw",
            combination_strategy=combination_strategy,
            veto_floor=veto_floor,
        ),
    )


# ----------------------------------------------------------------------
# _combine_confidence
# ----------------------------------------------------------------------

def test_minimum_strategy_returns_lower_of_the_two():
    step = _step_with_verifier(combination_strategy="minimum")
    runner = _runner()

    assert runner._combine_confidence(step, 0.9, 0.6) == 0.6
    assert runner._combine_confidence(step, 0.5, 0.8) == 0.5


def test_veto_below_floor_overrides_with_verifier_score():
    step = _step_with_verifier(combination_strategy="veto", veto_floor=0.6)
    runner = _runner()

    assert runner._combine_confidence(step, 0.9, 0.5) == 0.5


def test_veto_at_or_above_floor_keeps_primary_score():
    step = _step_with_verifier(combination_strategy="veto", veto_floor=0.6)
    runner = _runner()

    # verifier == floor — not vetoed (strictly less-than)
    assert runner._combine_confidence(step, 0.9, 0.6) == 0.9
    # verifier above floor — primary passes through unchanged
    assert runner._combine_confidence(step, 0.9, 0.95) == 0.9


# ----------------------------------------------------------------------
# _should_verify
# ----------------------------------------------------------------------

def test_always_true_verifies_regardless_of_confidence():
    verifier = VerifierConfig(executor="openclaw", trigger=VerifierTriggerConfig(always=True))
    runner = _runner()

    assert runner._should_verify(verifier, 0.99) is True
    assert runner._should_verify(verifier, 0.0) is True


def test_band_based_fires_inside_the_band():
    verifier = VerifierConfig(
        executor="openclaw",
        trigger=VerifierTriggerConfig(confidence_below=0.95, confidence_above=0.50),
    )
    runner = _runner()

    assert runner._should_verify(verifier, 0.70) is True


def test_band_based_skips_when_primary_clearly_confident():
    verifier = VerifierConfig(
        executor="openclaw",
        trigger=VerifierTriggerConfig(confidence_below=0.95, confidence_above=0.50),
    )
    runner = _runner()

    assert runner._should_verify(verifier, 0.96) is False


def test_band_based_skips_when_primary_clearly_failing():
    verifier = VerifierConfig(
        executor="openclaw",
        trigger=VerifierTriggerConfig(confidence_below=0.95, confidence_above=0.50),
    )
    runner = _runner()

    assert runner._should_verify(verifier, 0.40) is False


def test_band_boundaries_are_exclusive():
    verifier = VerifierConfig(
        executor="openclaw",
        trigger=VerifierTriggerConfig(confidence_below=0.95, confidence_above=0.50),
    )
    runner = _runner()

    assert runner._should_verify(verifier, 0.95) is False
    assert runner._should_verify(verifier, 0.50) is False


# ----------------------------------------------------------------------
# _join_confidences
# ----------------------------------------------------------------------

def test_all_must_pass_returns_minimum():
    runner = _runner()

    assert runner._join_confidences("all_must_pass", [0.9, 0.5, 0.7], [1.0, 1.0, 1.0]) == 0.5


def test_any_must_pass_returns_maximum():
    runner = _runner()

    assert runner._join_confidences("any_must_pass", [0.9, 0.5, 0.7], [1.0, 1.0, 1.0]) == 0.9


def test_weighted_average_combines_by_weight():
    runner = _runner()

    result = runner._join_confidences("weighted_average", [1.0, 0.0], [1.0, 3.0])

    assert result == 0.25


def test_empty_confidences_returns_zero():
    runner = _runner()

    assert runner._join_confidences("all_must_pass", [], []) == 0.0


def test_weighted_average_zero_total_weight_returns_zero():
    runner = _runner()

    assert runner._join_confidences("weighted_average", [0.5, 0.9], [0.0, 0.0]) == 0.0
