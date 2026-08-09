# Releasing

One doc for all four repos (VectorStep, VectorStep-Gateway,
VectorStep-Gateway-MCP, VectorStep-Service-MCP) — they release together and
there's one operator. See each repo's own `CHANGELOG.md` for what changed.

## Tag format

`vMAJOR.MINOR.PATCH` (e.g. `v0.6.0`), one tag per repo. Pushing a tag
matching `v*` is what publishes an image — see `.github/workflows/image.yml`
in VectorStep and VectorStep-Gateway. The two MCP repos don't publish
containers; their release is just the tag + (optionally) a PyPI publish, and
their version lives in `pyproject.toml`.

## The compatibility rule

**There is no protocol version negotiation between VectorStep and the
Gateway.** Nothing in either codebase declares or checks a wire version on
the WebSocket connection between them. Until that exists, the only supported
configuration is **matching tags, deployed together** — do not run VectorStep
`v0.7.0` against Gateway `v0.5.0` (or vice versa) and expect it to be a
supported combination, even if it happens to work.

If the pair ever *does* diverge in the lab — deliberately or by accident —
and it keeps working, that's the signal real version negotiation is worth
speccing, not evidence the rule above was unnecessary.

## Release order of operations

1. Tests green on every repo being released (`tests.yml` in each).
2. Tag each repo being released: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `image.yml` builds and pushes both images (VectorStep, VectorStep-Gateway)
   to GHCR, tagged `latest` and `vX.Y.Z`.
4. Pull the new images in the lab (this spec's manifests are the input to
   that; the pull itself is out of scope here — see
   `deploy/k8s/README.md`).

## Before you tag: agent config changes reset calibration

`AgentConfig.compute_version()`
(`VectorStep-Gateway/gateway/models/agent.py`) fingerprints an agent's
config; changing an agent's prompt, model, or tools changes its version and
starts its calibration buckets over from empty. This isn't specific to
tagging a release, but a release is a natural point to have made several such
edits at once — call out in the release note anyone with accumulated
calibration data on an agent you touched, so it isn't mistaken for a
regression when their next few runs show low confidence.
