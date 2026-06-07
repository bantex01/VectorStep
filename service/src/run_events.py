"""
In-process pub/sub bus for live pipeline run events.

The runner publishes log events as they happen; SSE endpoint subscribers
receive them in real-time. Each subscriber gets its own asyncio.Queue so
multiple concurrent tail connections to the same run work independently.

History is kept in memory for the lifetime of each run so that late-connecting
SSE clients receive a full replay of everything that happened before they
connected, then transition seamlessly into live streaming.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

# run_id -> list of subscriber queues
_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

# run_id -> ordered list of all events published so far
_history: dict[str, list[dict]] = {}


def subscribe(run_id: str) -> tuple[asyncio.Queue, list[dict]]:
    """Subscribe to a run and atomically snapshot its current history.

    Because asyncio is cooperative and both operations are synchronous, no
    events can be published between the subscribe and the snapshot. Replaying
    the returned snapshot then draining the queue is guaranteed duplicate-free.

    Returns (queue, history_snapshot).
    """
    q: asyncio.Queue = asyncio.Queue()
    _subscribers[run_id].append(q)
    snapshot = list(_history.get(run_id, []))
    return q, snapshot


def unsubscribe(run_id: str, q: asyncio.Queue) -> None:
    lst = _subscribers.get(run_id)
    if lst:
        try:
            lst.remove(q)
        except ValueError:
            pass
        if not lst:
            _subscribers.pop(run_id, None)


def publish(run_id: str, event: dict) -> None:
    _history.setdefault(run_id, []).append(event)
    for q in list(_subscribers.get(run_id, [])):
        q.put_nowait(event)


def publish_complete(run_id: str, status: str) -> None:
    """Push a completion sentinel then clean up all state for this run."""
    sentinel = {"type": "run_complete", "status": status}
    for q in list(_subscribers.get(run_id, [])):
        q.put_nowait(sentinel)
    _subscribers.pop(run_id, None)
    _history.pop(run_id, None)
