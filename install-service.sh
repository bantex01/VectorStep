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

# macOS machines commonly have more than one Python (Homebrew, python.org,
# pyenv, system) on PATH at once, and one of them can be broken — bad
# linking against system libraries, a corrupted build — while another
# works fine. Collect every 3.11+ interpreter found, deduplicated by
# resolved path, so the venv step below can fall through to the next one
# instead of dying on whichever happens to be found first.
PYTHON_CANDIDATES=""
for name in python3 python3.13 python3.12 python3.11; do
  bin="$(command -v "$name" 2>/dev/null)" || continue
  case " $PYTHON_CANDIDATES " in
    *" $bin "*) continue ;;
  esac
  if "$bin" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; then
    PYTHON_CANDIDATES="$PYTHON_CANDIDATES $bin"
  fi
done
[ -n "$PYTHON_CANDIDATES" ] || die "no Python 3.11+ found on PATH (checked python3, python3.11, python3.12, python3.13). Install Python 3.11+ and re-run."

log "preflight ok (git; Python 3.11+ found:$PYTHON_CANDIDATES)"

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

  PYTHON_BIN=""
  for candidate in $PYTHON_CANDIDATES; do
    log "creating virtualenv with $candidate"
    if "$candidate" -m venv .venv 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi

    # ensurepip's bootstrap install of pip into the new venv is a known
    # flaky step on some Python builds — venv creation itself proceeds, but
    # the internal `pip install --upgrade pip` subprocess fails (seen on
    # both a Homebrew python@3.12 with a stale/mismatched bottle and a
    # python@3.14 with a broken pyexpat build). Retry without the bundled
    # bootstrap and install pip ourselves instead.
    log "venv creation with $candidate failed bootstrapping pip — retrying without the bundled pip bootstrap"
    rm -rf .venv
    if "$candidate" -m venv --without-pip .venv 2>/dev/null \
      && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi

    log "$candidate could not produce a working virtualenv, trying the next Python found"
    rm -rf .venv
  done

  [ -n "$PYTHON_BIN" ] || die "none of the Python interpreters on PATH ($PYTHON_CANDIDATES) could create a working virtualenv — this points to a broken local Python install rather than anything this script can fix. Try installing Python fresh from https://python.org and re-run."
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
