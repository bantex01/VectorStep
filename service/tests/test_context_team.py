"""Team is resolved from the auth token (README §3b) and must reach step templates
so executors like `human` can route per-team — see pipeline/context.py build_context."""
from src.models.context import NormalisedContext
from src.models.pipeline import PipelineConfig, TriggerConfig
from src.pipeline.context import build_context


def _pipeline(name="p"):
    return PipelineConfig(name=name, trigger=TriggerConfig(), steps=[])


async def test_build_context_includes_team_when_set():
    normalised = NormalisedContext(source="generic", pipeline="p", team="team-a")
    ctx = await build_context(_pipeline(), normalised, "run-1", "step-1", {})
    assert ctx["team"] == "team-a"


async def test_build_context_team_is_none_when_unattributed():
    normalised = NormalisedContext(source="generic", pipeline="p")
    ctx = await build_context(_pipeline(), normalised, "run-1", "step-1", {})
    assert ctx["team"] is None
