import src.main as main
from src.models.pipeline import DedupConfig, PipelineConfig, StepConfig, TriggerConfig


def _pipeline(dedup: DedupConfig | None = None) -> PipelineConfig:
    return PipelineConfig(
        name="p",
        trigger=TriggerConfig(match={}, dedup=dedup),
        steps=[StepConfig(name="s", executor="openclaw")],
    )


def test_no_override_falls_back_to_service_defaults(monkeypatch):
    monkeypatch.setattr(main, "_dedup_enabled", True)
    monkeypatch.setattr(main, "_dedup_window_seconds", 300)

    enabled, window_seconds = main._resolve_dedup_settings(_pipeline(dedup=None))

    assert (enabled, window_seconds) == (True, 300)


def test_pipeline_can_widen_window(monkeypatch):
    monkeypatch.setattr(main, "_dedup_enabled", True)
    monkeypatch.setattr(main, "_dedup_window_seconds", 300)

    enabled, window_seconds = main._resolve_dedup_settings(
        _pipeline(dedup=DedupConfig(window_seconds=600))
    )

    assert (enabled, window_seconds) == (True, 600)


def test_pipeline_can_disable_dedup(monkeypatch):
    monkeypatch.setattr(main, "_dedup_enabled", True)
    monkeypatch.setattr(main, "_dedup_window_seconds", 300)

    enabled, window_seconds = main._resolve_dedup_settings(
        _pipeline(dedup=DedupConfig(enabled=False))
    )

    assert (enabled, window_seconds) == (False, 300)


def test_partial_override_only_overrides_set_fields(monkeypatch):
    monkeypatch.setattr(main, "_dedup_enabled", False)
    monkeypatch.setattr(main, "_dedup_window_seconds", 120)

    enabled, window_seconds = main._resolve_dedup_settings(
        _pipeline(dedup=DedupConfig(window_seconds=900))
    )

    # enabled falls back to the service default (False); window_seconds is overridden
    assert (enabled, window_seconds) == (False, 900)
