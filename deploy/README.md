# VectorStep + Gateway — evaluation deploy (docker compose)

This brings up the VectorStep + VectorStep-Gateway pair for local evaluation or
development. **It is not a production artifact** — for production use
[`k8s/`](k8s/) or [`systemd/`](systemd/) instead.

## Prerequisites

Sibling checkouts, both under the same parent directory:

```
<parent>/VectorStep
<parent>/VectorStep-Gateway
```

`docker-compose.yaml` builds `vectorstep` from `../` (this repo) and `gateway`
from `../../VectorStep-Gateway`. If you'd rather pull published images than
build locally, run `docker compose pull` instead of relying on the `build:`
blocks — the `image:` fallbacks point at `ghcr.io/bantex01/vectorstep:edge`
and `ghcr.io/bantex01/vectorstep-gateway:edge`.

## Quick start

```sh
cd deploy
cp compose.env.example .env
# edit .env: set ANTHROPIC_API_KEY, leave VECTORSTEP_GATEWAY_TOKEN blank for now
docker compose up -d
```

## First run — operator token bootstrap

VectorStep authenticates to the Gateway with a bearer token that the Gateway
generates for itself on first boot — there's no way to pre-supply it. Do this
once, after the first `docker compose up`:

1. Bring the stack up with `VECTORSTEP_GATEWAY_TOKEN` blank in `.env` (the
   `vectorstep` service will log a warning that its Gateway executor is
   unauthenticated — expected on this first pass).
2. Read the token the Gateway generated for itself:
   ```sh
   docker compose exec gateway cat /data/identity/device-auth.json
   ```
   Take the value at `.tokens.operator.token` (pipe through `jq -r
   .tokens.operator.token` if you have it installed).
3. Put that value in `.env` as `VECTORSTEP_GATEWAY_TOKEN`.
4. Recreate the `vectorstep` service so it picks up the new environment
   variable (`restart` alone does **not** re-read `.env` for an
   already-created container):
   ```sh
   docker compose up -d vectorstep
   ```

From here on, `docker compose down && docker compose up -d` reuses the same
identity and token — the bootstrap is only needed once per `gateway-data`
volume.

## Postgres (optional)

```sh
docker compose --profile postgres up -d
```
then point `deploy/compose/vectorstep-config.yaml`'s `database.url` at
`postgresql+asyncpg://vectorstep:vectorstep@postgres:5432/vectorstep`.

## Files

- `docker-compose.yaml` — the stack.
- `compose/vectorstep-config.yaml`, `compose/gateway-config.yaml` — minimal
  working configs mounted read-only into each container. Edit these directly
  for evaluation; see each repo's `samples/config.yaml.example` for the full
  annotated reference.
- `compose.env.example` — copy to `.env`, see above.
