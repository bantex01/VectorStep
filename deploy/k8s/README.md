# Kubernetes manifests — VectorStep

Plain, copy-and-adapt YAML — the same spirit as `samples/`, not a generic
chart. Assumes you already have images published to GHCR (see
`../../RELEASING.md` for how a tag becomes an image) — apply in this order:

```sh
kubectl create secret generic vectorstep-secrets \
  --from-literal=VECTORSTEP_GATEWAY_TOKEN=... \
  --from-literal=VECTORSTEP_WEBHOOK_TOKEN=...
kubectl apply -f pvc.yaml
kubectl apply -f configmap.example.yaml   # copy + edit first — see the file's header comment
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

Deploy VectorStep-Gateway (`../../VectorStep-Gateway/deploy/k8s/`) alongside
it — the config here assumes a Gateway Service named `vectorstep-gateway` is
reachable in-cluster.

**Why no Helm chart.** These manifests are the ground truth a chart would
template. Templating them before there's a second real user to justify the
abstraction is premature — this is a deliberate deferral, not an oversight.
Revisit once there's a concrete reason (a second environment, a second
operator) that a raw `kubectl apply` doesn't serve well.

**Image tags** are what a GitOps controller (Argo CD, Flux, a `kubectl set
image` job) would watch to pull new releases — see `../../RELEASING.md` for
what a tag means and the compatibility rule between VectorStep and the
Gateway. Continuous deployment itself is out of scope here; these manifests
are its input.

**Single replica.** `deployment.yaml` sets `replicas: 1` and
`strategy: Recreate` — required today regardless of database backend, since
the scheduler and dedup/event state are in-process. See the comments in
`deployment.yaml` and `configmap.example.yaml` for why that also makes the
in-process Alembic migration on boot safe with no init container.
