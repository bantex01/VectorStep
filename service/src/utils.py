from datetime import datetime, timezone


def utc_now() -> datetime:
    """Naive UTC timestamp — deprecation-safe replacement for datetime.utcnow().

    Returns the same naive-UTC value datetime.utcnow() did, so it remains
    comparable with existing stored timestamps and ISO strings.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
