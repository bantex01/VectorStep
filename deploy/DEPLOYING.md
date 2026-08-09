# Deploying to Kubernetes, start to finish

This stitches together `RELEASING.md` (how a tag becomes an image),
`deploy/k8s/README.md` in each repo (the `kubectl apply` order), and the
gotchas that don't fit neatly in either — into one linear path from a local
change to a running cluster. It assumes:

- `VectorStep` and `VectorStep-Gateway` checked out as sibling directories,
  both with `origin` pointed at `github.com/bantex01/...`.
- `kubectl` already configured against your target cluster.
- Docker installed if you want to build/smoke-test locally before pushing
  (optional — CI builds regardless).

## 1. (Optional) Build and smoke-test locally

```sh
cd VectorStep
docker build -f service/Dockerfile -t vectorstep:dev .
docker run --rm vectorstep:dev
# expected: exits fast with "Config file not found at '/etc/vectorstep/config.yaml' ..."
# — that's the fail-fast check working, not a bug.
```

Same idea for the Gateway from its own repo root with the root `Dockerfile`.

## 2. Push your changes

You commit and push — see each repo's own workflow for that. Pushing to
`main` runs `tests.yml` and runs `image.yml` in **build-only** mode (no
push to GHCR) — check the Actions tab to confirm both are green before you
cut a release.

## 3. Cut a release — this is what actually publishes an image

```sh
cd VectorStep
git tag v0.1.0
git push origin v0.1.0

cd ../VectorStep-Gateway
git tag v0.1.0
git push origin v0.1.0
```

Tag push triggers `image.yml` on each repo: builds `linux/amd64` +
`linux/arm64`, pushes `ghcr.io/bantex01/vectorstep:v0.1.0` (+ `:latest`) and
`ghcr.io/bantex01/vectorstep-gateway:v0.1.0` (+ `:latest`). Watch the Actions
tab — the arm64 leg runs under QEMU emulation and is genuinely slow
(multi-minute pip installs), not stuck.

**Use the same version for both repos on a given release** — see
`RELEASING.md`'s compatibility rule: there's no protocol negotiation between
VectorStep and the Gateway yet, so mismatched tags are unsupported even if
they happen to work.

## 4. Make the images pullable

New GHCR packages are **private by default**, even though the workflow that
published them used the repo's own `GITHUB_TOKEN`. Before your cluster can
pull:

- GitHub → your profile → **Packages** → `vectorstep` (and
  `vectorstep-gateway`) → **Package settings** → **Change visibility** →
  **Public**. This is the simplest option for a home-lab cluster with no
  registry credentials configured.
- Alternatively, keep them private and give the cluster an
  `imagePullSecret`: create a PAT with `read:packages` scope,
  `kubectl create secret docker-registry ghcr-pull --docker-server=ghcr.io
  --docker-username=<gh-username> --docker-password=<pat>`, then add
  `imagePullSecrets: [{name: ghcr-pull}]` to the pod spec in each
  `deployment.yaml` (not present by default — add it if you go this route).

Verify either way before moving on:

```sh
docker pull ghcr.io/bantex01/vectorstep:v0.1.0
```

## 5. Deploy the Gateway first

VectorStep's config needs the Gateway's operator token, which only exists
once the Gateway has booted once — same bootstrap dependency as the compose
path, just via `kubectl exec` instead of `docker compose exec`.

```sh
cd VectorStep-Gateway/deploy/k8s
cp configmap.example.yaml configmap.yaml
$EDITOR configmap.yaml   # fill in providers.anthropic.api_key's env var name etc.

kubectl create secret generic vectorstep-gateway-secrets \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...

kubectl apply -f pvc.yaml -f configmap.yaml -f deployment.yaml -f service.yaml
kubectl rollout status deployment/vectorstep-gateway
```

Fetch the operator token:

```sh
kubectl exec deploy/vectorstep-gateway -- cat /data/identity/device-auth.json
```

Take the value at `.tokens.operator.token`.

## 6. Deploy VectorStep

```sh
cd ../../../VectorStep/deploy/k8s
cp configmap.example.yaml configmap.yaml
$EDITOR configmap.yaml

kubectl create secret generic vectorstep-secrets \
  --from-literal=VECTORSTEP_GATEWAY_TOKEN=<token-from-step-5> \
  --from-literal=VECTORSTEP_WEBHOOK_TOKEN=$(openssl rand -hex 24)

kubectl apply -f pvc.yaml -f configmap.yaml -f deployment.yaml -f service.yaml
kubectl rollout status deployment/vectorstep
```

## 7. Verify

```sh
kubectl port-forward svc/vectorstep-gateway 18780:18780 &
curl -s localhost:18780/health

kubectl port-forward svc/vectorstep 8000:8000 &
curl -s localhost:8000/health
```

Both should report `"status": "ok"` and the version you tagged in step 3.

## 8. Upgrading later

Tag a new release (step 3), confirm both images are pullable (step 4), then
roll the images in place — no manifest edits needed unless the release
changed a config shape:

```sh
kubectl set image deployment/vectorstep-gateway vectorstep-gateway=ghcr.io/bantex01/vectorstep-gateway:vX.Y.Z
kubectl set image deployment/vectorstep vectorstep=ghcr.io/bantex01/vectorstep:vX.Y.Z
```

Gateway first, then VectorStep, for the same reason as first install — keep
them on matching tags. `deployment.yaml` ships pinned at `:latest` by
default; switch it to an explicit `vX.Y.Z` once you're past initial setup so
a re-apply of the manifest can't silently move you to a newer image than the
one you tested.
