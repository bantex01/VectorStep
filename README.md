# VectorStep

A webhook-triggered, YAML-configured AI pipeline orchestration service, with every autonomous action gated by a trust vector built from independent signals.

## What it is

VectorStep receives webhooks from any source (Alertmanager, Grafana, Atlassian, generic JSON), normalises the payload, resolves a named pipeline config, and executes a multi-step AI pipeline using pluggable agent executor backends.

The service is designed to be:

- **Source agnostic** — any webhook source is supported via pluggable parsers
- **Executor agnostic** — AI backends are adapters behind a common interface; steps in the same pipeline can mix executors freely
- **Config driven** — all pipeline logic lives in YAML files, not code
- **Modular** — adding a new source parser or executor adapter requires no changes to core logic

Primary use case is observability automation (alert triage, Grafana investigation, bounded remediation), but the design is intentionally general purpose. Every step's result feeds a trust vector (self-report, optional verifier, optional grounding, optional deterministic checks, optional calibration) before the runner decides whether to proceed, escalate, or abort.

## Quick start

```bash
cd service
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt   # requirements.txt lives at the repo root

cp ../samples/config.yaml.example config.yaml
uvicorn src.main:app --reload --port 8000

# Verify: trigger the bundled sample webhook
curl -X POST "http://localhost:8000/webhook?source=alertmanager" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/alertmanager_critical.json
# → {"status": "accepted", "run_id": "<uuid>"}
```

This runs the engine alone, against SQLite, with no gated steps. A first real gated pipeline needs the [VectorStep Gateway](https://github.com/bantex01/VectorStep-Gateway) running too — full walkthrough (both services, first pipeline, in about ten minutes): [Quick start](https://vectorstep.io/docs/getting-started/quick-start/). Full installation reference (Postgres, testing, both services in detail): [Installation](https://vectorstep.io/docs/getting-started/installation/).

## Documentation

Full docs at [vectorstep.io](https://vectorstep.io/docs/):

| Section | Covers |
|---|---|
| [Getting Started](https://vectorstep.io/docs/getting-started/installation/) | Install, config, first pipeline |
| [Concepts](https://vectorstep.io/docs/concepts/architecture/) | Architecture, confidence & the trust vector, pipeline stages, promotion readiness |
| [Pipelines](https://vectorstep.io/docs/pipelines/schema/) | The YAML schema, step library, verifiers, grounding, calibration, flow control, scheduling |
| [Sources & Executors](https://vectorstep.io/docs/integrations/webhooks/) | Webhook intake, executor adapters, MCP servers |
| [UI & Insights](https://vectorstep.io/docs/ui/overview/) | Dashboard, run detail, insights pages, marking queue |
| [Gateway](https://vectorstep.io/docs/gateway/overview/) | The companion agent runtime |
| [Operations](https://vectorstep.io/docs/operations/deployment/) | Deployment, durability, observability, cost accounting |
| [Reference](https://vectorstep.io/docs/reference/api/) | REST API, LLMOutput contract, config reference |
| [Design & Internals](https://vectorstep.io/docs/design/decisions/) | Design decisions, extending VectorStep, project structure, testing |

## The ecosystem

| Repo | Role |
|---|---|
| **VectorStep** | The orchestration service: webhook intake, pipeline runner, trust gating, UI, analytics |
| **VectorStep-Gateway** | WebSocket gateway that runs agents: LLM providers, MCP tools, the full agentic loop |
| **VectorStep-Service-MCP** | MCP server exposing pipeline authoring, run inspection and analytics to Claude Code/Desktop |
| **VectorStep-Gateway-MCP** | MCP server for authoring and inspecting Gateway agents |

## Licence and contributions

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Open source, but not open contribution: no external code contributions are accepted. See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and [Licence & contributions](https://vectorstep.io/docs/about/licence-and-contributions/) for the full policy and why.
