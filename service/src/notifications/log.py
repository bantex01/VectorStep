import logging

from jinja2 import Environment, Undefined

from ..models.pipeline import NotificationConfig

_jinja_env = Environment(undefined=Undefined)

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LogNotifier:
    """Emits pipeline state-transition notifications to the application logger.

    Always available — no external service or config.yaml block required.
    Use it as a zero-dependency fallback when Telegram / webhook aren't
    configured, or as a lightweight audit trail that lands in whatever log
    aggregation stack (stdout, file, Loki, CloudWatch, etc.) the service
    already ships to.

    Per-notification config keys (under notification.config):
        level   — log level: debug | info | warning | error | critical (default: warning)
        logger  — logger name (default: vectorstep.notifications)
    """

    async def send(self, notification: NotificationConfig, context: dict) -> None:
        cfg = notification.config
        level_str = str(cfg.get("level", "warning")).lower()
        level = _LEVELS.get(level_str, logging.WARNING)
        logger_name = cfg.get("logger", "vectorstep.notifications")

        log = logging.getLogger(logger_name)
        message = _jinja_env.from_string(notification.template).render(**context).strip()
        log.log(level, "%s", message)
