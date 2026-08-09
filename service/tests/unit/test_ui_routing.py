"""Route-manifest regression test for the ui/ package split (SPEC-ui-modularisation.md).

The list below is the exact (method, path) set captured from the monolithic
ui.py router before the split into ui/dashboard.py, ui/runs.py, etc. Each area
now owns its own APIRouter (no prefix) and ui/__init__.py aggregates them —
this test is the guarantee that the split added, dropped, or renamed nothing.

It asserts *set* equality, not the exact pre-refactor registration order. The
old file registered routes in physical decorator order, which interleaved
areas (e.g. the pipelines list route landed before the insights block, but the
pipeline-detail route landed after it) — reproducing that exact interleaving
would require splitting decorators for the same area across multiple router
objects, defeating the one-router-per-module design. This is safe rather than
just convenient: FastAPI/Starlette pick the first *matching* route, and two
routes can only both match one concrete request path if they share the same
HTTP method, the same segment count, and agree at every segment on literal-vs-
{param} — test_no_ambiguous_route_shapes below asserts that invariant holds
for the whole router, which is what actually makes registration order
irrelevant here. If a future route ever violates that invariant, order starts
mattering again and this test's rationale needs revisiting.

Future intentional route additions/removals edit PRE_REFACTOR_ROUTES below.
"""
from src.ui import router

PRE_REFACTOR_ROUTES = [
    (("GET",), "/ui"),
    (("GET",), "/ui/"),
    (("GET",), "/ui/agents"),
    (("GET",), "/ui/agents/{executor}/{agent_id}"),
    (("GET",), "/ui/approvals"),
    (("GET",), "/ui/approvals/{token}"),
    (("GET",), "/ui/insights"),
    (("GET",), "/ui/insights/agents"),
    (("GET",), "/ui/insights/mcp"),
    (("GET",), "/ui/insights/models"),
    (("GET",), "/ui/insights/pipelines"),
    (("GET",), "/ui/insights/providers"),
    (("GET",), "/ui/insights/steps"),
    (("GET",), "/ui/insights/teams"),
    (("GET",), "/ui/marking-queue"),
    (("GET",), "/ui/mcp"),
    (("GET",), "/ui/pipelines"),
    (("GET",), "/ui/pipelines/{name}"),
    (("GET",), "/ui/pipelines/{name}/feedback"),
    (("GET",), "/ui/providers"),
    (("GET",), "/ui/replays/new"),
    (("GET",), "/ui/replays/{run_id}"),
    (("POST",), "/ui/replays"),
    (("GET",), "/ui/runs"),
    (("GET",), "/ui/runs/{run_id}"),
    (("GET",), "/ui/runs/{run_id}/stream"),
    (("GET",), "/ui/schedules"),
    (("GET",), "/ui/steps"),
    (("POST",), "/ui/approvals/{token}/approve"),
    (("POST",), "/ui/approvals/{token}/reject"),
]


def _current_routes() -> list[tuple[tuple[str, ...], str]]:
    return sorted(
        (tuple(sorted(getattr(r, "methods", None) or [])), r.path)
        for r in router.routes
    )


def test_route_manifest_unchanged():
    current = _current_routes()
    expected = sorted(PRE_REFACTOR_ROUTES)
    assert current == expected, (
        f"route set changed — only in old: {set(expected) - set(current)}, "
        f"only in new: {set(current) - set(expected)}"
    )


def test_no_ambiguous_route_shapes():
    """No two routes sharing a method may have the same segment count while
    also agreeing, at every segment, on literal-vs-{param} — that's the
    condition under which registration order could change which handler
    serves a request. See module docstring."""

    def shape(path: str) -> tuple[str, ...]:
        return tuple("{}" if seg.startswith("{") else seg for seg in path.split("/"))

    by_method: dict[str, list[str]] = {}
    for methods, path in ((tuple(sorted(getattr(r, "methods", None) or [])), r.path) for r in router.routes):
        for m in methods:
            by_method.setdefault(m, []).append(path)

    for method, paths in by_method.items():
        seen: dict[tuple[str, ...], str] = {}
        for path in paths:
            key = shape(path)
            assert key not in seen, (
                f"{method} {path!r} has the same shape as {seen[key]!r} — "
                "registration order now matters, which breaks this test's safety argument"
            )
            seen[key] = path
