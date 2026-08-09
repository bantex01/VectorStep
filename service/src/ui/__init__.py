from fastapi import APIRouter

from . import agents, approvals, dashboard, insights, insights_trust, pipelines, replays, runs, steps
from .helpers import configure, _fetch_openclaw_agents, _fetch_vectorstep_gateway_agents
from .pipelines import _read_pipeline_yaml
from .steps import _read_step_yaml

# Include order here does not reproduce ui.py's pre-refactor route registration
# order exactly (SPEC-ui-modularisation.md §2's "derive the safe order" caveat)
# because interleaved areas (e.g. pipelines list vs. pipelines detail landed on
# either side of the insights block in the old file) can't share one router per
# area without re-interleaving decorators across modules. This is safe: every
# route's path shape (segment count + which segments are literal vs {param}) is
# unique across the whole router, so no two routes can ever both match the same
# concrete request path — registration order therefore cannot change which
# handler serves a given request. tests/unit/test_ui_routing.py asserts the
# route *set* is unchanged from the pre-refactor snapshot.
# The aggregate itself carries no prefix: dashboard.router already has "/ui"
# baked into its two routes (it needs its own prefix for the bare "" root path
# to be valid at all — see ui/dashboard.py), so it's included as-is. Every other
# area router has no prefix of its own, so "/ui" is supplied here via the
# include_router prefix= kwarg instead.
router = APIRouter()
router.include_router(dashboard.router)
router.include_router(runs.router, prefix="/ui")
router.include_router(pipelines.router, prefix="/ui")
router.include_router(insights.router, prefix="/ui")
router.include_router(insights_trust.router, prefix="/ui")
router.include_router(steps.router, prefix="/ui")
router.include_router(agents.router, prefix="/ui")
router.include_router(approvals.router, prefix="/ui")
router.include_router(replays.router, prefix="/ui")

__all__ = [
    "router",
    "configure",
    "_fetch_openclaw_agents",
    "_fetch_vectorstep_gateway_agents",
    "_read_pipeline_yaml",
    "_read_step_yaml",
]
