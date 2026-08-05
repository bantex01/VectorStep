"""Tests for the log notification channel."""
import logging
import pytest

from src.notifications.log import LogNotifier
from src.models.pipeline import NotificationConfig


def _notif(template: str, config: dict | None = None) -> NotificationConfig:
    return NotificationConfig(channel="log", template=template, config=config or {})


# ---------------------------------------------------------------------------
# Default behaviour
# ---------------------------------------------------------------------------

async def test_logs_at_warning_by_default(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.WARNING, logger="vectorstep.notifications"):
        await notifier.send(_notif("pipeline escalated"), context={})
    assert "pipeline escalated" in caplog.text


async def test_default_logger_name(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.WARNING, logger="vectorstep.notifications"):
        await notifier.send(_notif("hello"), context={})
    assert any(r.name == "vectorstep.notifications" for r in caplog.records)


# ---------------------------------------------------------------------------
# Level config
# ---------------------------------------------------------------------------

async def test_level_info(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.DEBUG, logger="vectorstep.notifications"):
        await notifier.send(_notif("info msg", {"level": "info"}), context={})
    record = next(r for r in caplog.records if "info msg" in r.message)
    assert record.levelno == logging.INFO


async def test_level_error(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.DEBUG, logger="vectorstep.notifications"):
        await notifier.send(_notif("error msg", {"level": "error"}), context={})
    record = next(r for r in caplog.records if "error msg" in r.message)
    assert record.levelno == logging.ERROR


async def test_level_debug(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.DEBUG, logger="vectorstep.notifications"):
        await notifier.send(_notif("debug msg", {"level": "debug"}), context={})
    record = next(r for r in caplog.records if "debug msg" in r.message)
    assert record.levelno == logging.DEBUG


async def test_level_warn_alias(caplog):
    """'warn' is accepted as an alias for 'warning'."""
    notifier = LogNotifier()
    with caplog.at_level(logging.DEBUG, logger="vectorstep.notifications"):
        await notifier.send(_notif("warn msg", {"level": "warn"}), context={})
    record = next(r for r in caplog.records if "warn msg" in r.message)
    assert record.levelno == logging.WARNING


async def test_unknown_level_falls_back_to_warning(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.DEBUG, logger="vectorstep.notifications"):
        await notifier.send(_notif("unknown level msg", {"level": "critical_bad"}), context={})
    record = next(r for r in caplog.records if "unknown level msg" in r.message)
    assert record.levelno == logging.WARNING


# ---------------------------------------------------------------------------
# Custom logger name
# ---------------------------------------------------------------------------

async def test_custom_logger_name(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.WARNING, logger="my.custom.logger"):
        await notifier.send(
            _notif("custom logger msg", {"logger": "my.custom.logger"}), context={}
        )
    assert any(r.name == "my.custom.logger" for r in caplog.records)


# ---------------------------------------------------------------------------
# Jinja2 template rendering
# ---------------------------------------------------------------------------

async def test_template_rendered_with_context(caplog):
    notifier = LogNotifier()
    ctx = {"pipeline_name": "alert-triage", "step_summary": "confidence too low"}
    with caplog.at_level(logging.WARNING, logger="vectorstep.notifications"):
        await notifier.send(
            _notif("Pipeline {{pipeline_name}} escalated: {{step_summary}}"),
            context=ctx,
        )
    assert "alert-triage" in caplog.text
    assert "confidence too low" in caplog.text


async def test_template_strips_leading_trailing_whitespace(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.WARNING, logger="vectorstep.notifications"):
        await notifier.send(_notif("  trimmed  \n"), context={})
    record = next(r for r in caplog.records if "trimmed" in r.message)
    assert record.message == "trimmed"


async def test_undefined_template_var_renders_empty(caplog):
    """Undefined Jinja2 variables render to '' (Undefined), not an exception."""
    notifier = LogNotifier()
    with caplog.at_level(logging.WARNING, logger="vectorstep.notifications"):
        await notifier.send(_notif("val={{missing_var}}"), context={})
    assert "val=" in caplog.text  # rendered without error


# ---------------------------------------------------------------------------
# Integration with runner._dispatch_notification
# ---------------------------------------------------------------------------

async def test_dispatch_notification_routes_to_log_notifier(caplog):
    """Runner._dispatch_notification delivers to LogNotifier when channel=log."""
    from src.pipeline.runner import PipelineRunner
    from src.models.pipeline import PipelineConfig, TriggerConfig, NotificationConfig

    notifier = LogNotifier()
    runner = PipelineRunner(
        executors={},
        session_factory=None,
        notifiers={"log": notifier},
    )
    pipeline = PipelineConfig(
        name="test-pipeline",
        trigger=TriggerConfig(),
        steps=[],
        notifications={
            "escalate": [
                NotificationConfig(
                    channel="log",
                    template="Escalated: {{pipeline_name}}",
                    config={"level": "error"},
                )
            ]
        },
    )

    with caplog.at_level(logging.DEBUG, logger="vectorstep.notifications"):
        await runner._dispatch_notification(
            pipeline=pipeline,
            action="escalate",
            context={"pipeline_name": "test-pipeline"},
            run_log=[],
        )

    assert "Escalated: test-pipeline" in caplog.text
    record = next(r for r in caplog.records if "Escalated" in r.message)
    assert record.levelno == logging.ERROR


async def test_dispatch_notification_production_uses_configured_channel():
    """stage=production sends to the configured channel and logs notification_sent."""
    from src.pipeline.runner import PipelineRunner
    from src.models.pipeline import PipelineConfig, TriggerConfig, NotificationConfig

    notifier = LogNotifier()
    runner = PipelineRunner(
        executors={}, session_factory=None, notifiers={"log": notifier},
    )
    pipeline = PipelineConfig(
        name="test-pipeline",
        trigger=TriggerConfig(),
        steps=[],
        stage="production",
        notifications={"escalate": [NotificationConfig(channel="log", template="hi")]},
    )
    run_log: list = []
    await runner._dispatch_notification(
        pipeline=pipeline, action="escalate", context={}, run_log=run_log,
    )
    events = [e["event"] for e in run_log]
    assert "notification_sent" in events
    assert "notification_suppressed_testing" not in events


async def test_dispatch_notification_testing_routes_to_log_and_suppresses(caplog):
    """stage=testing (the default) mutes a telegram notification to the log channel."""
    from src.pipeline.runner import PipelineRunner
    from src.models.pipeline import PipelineConfig, TriggerConfig, NotificationConfig

    notifier = LogNotifier()
    runner = PipelineRunner(
        executors={}, session_factory=None, notifiers={"log": notifier},
    )
    pipeline = PipelineConfig(
        name="test-pipeline",
        trigger=TriggerConfig(),
        steps=[],
        notifications={
            "escalate": [
                NotificationConfig(channel="telegram", template="Escalated: {{pipeline_name}}")
            ]
        },
    )
    run_log: list = []
    with caplog.at_level(logging.DEBUG, logger="vectorstep.notifications"):
        await runner._dispatch_notification(
            pipeline=pipeline,
            action="escalate",
            context={"pipeline_name": "test-pipeline"},
            run_log=run_log,
        )

    assert "Escalated: test-pipeline" in caplog.text
    events = [e["event"] for e in run_log]
    assert "notification_suppressed_testing" in events
    assert "notification_sent" not in events


async def test_dispatch_notification_testing_no_log_notifier_registered():
    """If no 'log' notifier is registered, testing-mode gating skips cleanly (warns, doesn't raise)."""
    from src.pipeline.runner import PipelineRunner
    from src.models.pipeline import PipelineConfig, TriggerConfig, NotificationConfig

    runner = PipelineRunner(executors={}, session_factory=None, notifiers={})
    pipeline = PipelineConfig(
        name="test-pipeline",
        trigger=TriggerConfig(),
        steps=[],
        notifications={"escalate": [NotificationConfig(channel="telegram", template="hi")]},
    )
    run_log: list = []
    await runner._dispatch_notification(
        pipeline=pipeline, action="escalate", context={}, run_log=run_log,
    )
    assert run_log == []
