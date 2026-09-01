#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PHYSICS_SOURCE="$REPOSITORY_ROOT/physics/src"

cd "$REPOSITORY_ROOT/physics"
PYTHONPATH="$PHYSICS_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m unittest discover -s tests -v
PYTHONPATH="$PHYSICS_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m physics_engine.checkpoint_eval --help >/dev/null

cd "$REPOSITORY_ROOT"
PYTHONPATH="$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m harness.patchhead_robustness --help >/dev/null
"$PYTHON_BIN" -m unittest discover -s patchhead/tests -v
"$PYTHON_BIN" infer.py --help >/dev/null
"$PYTHON_BIN" patchhead/infer.py --help >/dev/null
