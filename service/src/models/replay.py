from pydantic import BaseModel, Field


class ReplayConfig(BaseModel):
    """config.yaml `replay:` block (SPEC-replay-shadow-eval.md §2).

    safe_agents is an explicit allowlist of "executor:agent" identities the
    operator asserts are read-only. A replay request is rejected (403) unless
    BOTH the recorded step's agent and the candidate's agent appear here — a
    replay batch re-runs live tool calls, so an unlisted agent could re-fire
    side effects the operator never intended. An empty/absent list means
    replay is not configured at all: every request 403s with a pointer to
    this config key, never a silent "nothing is safe" allow-list-of-zero.
    """

    safe_agents: list[str] = Field(default_factory=list)
