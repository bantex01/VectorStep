#!/usr/bin/env bash
# Quick-start installer for the VectorStep orchestration service.
#
#   curl -sSL https://raw.githubusercontent.com/bantex01/VectorStep/main/install-service.sh | bash
#
# Safe to run more than once: an existing config.yaml is never overwritten.
#
# This installs the service alone, against SQLite — a complete, working
# install for webhook/human/notify-only pipelines, or for pointing at an
# already-running Gateway elsewhere. It does not require the VectorStep
# Gateway to be installed on this machine.
set -euo pipefail

REPO_URL="https://github.com/bantex01/VectorStep.git"
INSTALL_DIR="$HOME/.vectorstep"

log()  { printf '==> %s\n' "$1"; }
skip() { printf '==> skip: %s\n' "$1"; }
die()  { printf 'error: %s\n' "$1" >&2; exit 1; }

# --- Preflight ---------------------------------------------------------

command -v git >/dev/null 2>&1 || die "git not found on PATH. Install git and re-run."

PYTHON_BIN=""
for candidate in python3.11 python3.12 python3.13 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done
[ -n "$PYTHON_BIN" ] || die "no python3 found on PATH. Install Python 3.11+ and re-run."

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' \
  || die "found $PYTHON_BIN ($PYTHON_VERSION), but VectorStep needs Python 3.11+."

"$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || die "pip not available for $PYTHON_BIN. Install pip and re-run."

log "preflight ok (git, $PYTHON_BIN $PYTHON_VERSION, pip)"

# --- Clone ---------------------------------------------------------------

if [ -d "$INSTALL_DIR/.git" ]; then
  skip "$INSTALL_DIR already a git checkout, not re-cloning"
else
  log "cloning into $INSTALL_DIR"
  git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR/service"

# --- Virtualenv + deps -----------------------------------------------------
# venv and config live in service/; requirements.txt lives at the repo root.

if [ -d ".venv" ] && [ -x ".venv/bin/pip" ]; then
  skip ".venv already exists, not recreating"
else
  if [ -d ".venv" ]; then
    log "existing .venv has no working pip, recreating"
    rm -rf .venv
  fi

  log "creating virtualenv"
  if ! "$PYTHON_BIN" -m venv .venv; then
    # ensurepip's bootstrap install of pip into the new venv is a known
    # flaky step on some Python builds (Homebrew's python@3.12 in
    # particular) — venv creation itself succeeds, but that internal `pip
    # install --upgrade pip` subprocess fails with no useful error surfaced.
    # Recreate without the bundled bootstrap and install pip ourselves.
    log "venv creation failed bootstrapping pip (a known issue on some Python builds, e.g. Homebrew's python@3.12) — retrying without the bundled pip bootstrap"
    rm -rf .venv
    "$PYTHON_BIN" -m venv --without-pip .venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
  fi
fi

log "installing dependencies"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r ../requirements.txt

# --- Config ----------------------------------------------------------------

if [ -f "config.yaml" ]; then
  skip "config.yaml already exists, leaving it alone"
else
  log "writing config.yaml from samples/config.yaml.example (SQLite by default)"
  cp ../samples/config.yaml.example config.yaml
fi

# pipeline_config_dir/step_library_dir are gitignored, so a fresh clone lacks
# them — and the service refuses to boot if they're missing entirely (an
# empty directory is fine, a missing one isn't). Create them if absent so
# the service actually starts; a pipeline-less install is a valid starting
# state, same as an agent-less Gateway.
if [ -d "pipelines" ]; then
  skip "pipelines/ already exists"
else
  log "creating empty pipelines/"
  mkdir -p pipelines
fi

if [ -d "steps" ]; then
  skip "steps/ already exists"
else
  log "creating empty steps/"
  mkdir -p steps
fi

# --- Summary -----------------------------------------------------------

cat <<EOF

VectorStep service installed at $INSTALL_DIR

Next steps:

  1. If you're using the VectorStep Gateway for gated AI steps, paste its
     operator token into executors.gateway.token in
     $INSTALL_DIR/service/config.yaml, then restart the service.
  2. Start the service:

       cd $INSTALL_DIR/service && source .venv/bin/activate && uvicorn src.main:app --reload --port 8000

  3. Send a test webhook:

       curl -X POST "http://localhost:8000/webhook?source=alertmanager" \\
         -H "Content-Type: application/json" \\
         -d @tests/fixtures/alertmanager_critical.json

EOF
