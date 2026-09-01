#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/DistortAware}"
DATA="${DATA:-$REPO/data/matched_refactored}"
source "$REPO/slurm/_env.sh"
# This is a networked login-node fetch; GPU jobs remain offline.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

python patchhead/fetch_sid_eval.py --output-dir "$DATA" \
  --per-class "${PER_CLASS:-100}" --seed "${SEED:-42}" "$@"
