"""
In-process pub/sub bus for live pipeline run events.

The runner publishes log events as they happen; SSE endpoint subscribers
receive them in real-time. Each subscriber gets its own asyncio.Queue so
multiple concurrent tail connections to the same run work independently.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

# run_id -> list of subscriber queues
_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)


def subscribe(run_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _subscribers[run_id].append(q)
    return q


def unsubscribe(run_id: str, q: asyncio.Queue) -> None:
    try:
        _subscribers[run_id].remove(q)
    except ValueError:
        pass
    if not _subscribers.get(run_id):
        _subscribers.pop(run_id, None)


def publish(run_id: str, event: dict) -> None:
    for q in list(_subscribers.get(run_id, [])):
        q.put_nowait(event)


def publish_complete(run_id: str, status: str) -> None:
    """Push a completion sentinel then clean up all subscriber queues for this run."""
    sentinel = {"type": "run_complete", "status": status}
    for q in list(_subscribers.get(run_id, [])):
        q.put_nowait(sentinel)
    _subscribers.pop(run_id, None)
