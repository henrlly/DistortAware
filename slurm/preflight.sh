#!/bin/bash
# Run on the login node before submitting a PatchHead job.
set -euo pipefail

REPO="${REPO:-$HOME/tiktok-aigc-detect}"
DATA="${MATCHED_DATA:-$REPO/data/matched_refactored}"
TAG="${TAG:-distortion_aware}"

cd "$REPO"
test -x "$REPO/.venv/bin/python"
test -f "$DATA/train.csv"
test -f "$DATA/validation.csv"
test -f "$DATA/calibration.csv"
test -f "$DATA/wildfake_benchmark.csv"
test -f "$DATA/sid_eval_200.csv"

echo "repo=$REPO"
echo "python=$REPO/.venv/bin/python"
echo "data=$DATA"
echo "checkpoint=$REPO/patchhead/checkpoints/patchhead_${TAG}.pt"

source "$REPO/.venv/bin/activate"
python -m py_compile patchhead/manifest.py patchhead/transforms.py patchhead/metrics.py \
  patchhead/data.py patchhead/fetch.py patchhead/cli.py patchhead/model.py \
  patchhead/train.py patchhead/evaluate.py patchhead/make_report.py patchhead/predict.py
python -m pip check

if test -f "$REPO/patchhead/checkpoints/patchhead_${TAG}.pt"; then
  echo "checkpoint: present"
else
  echo "checkpoint: absent (required only for evaluation-only jobs)"
fi
echo "preflight: OK (GPU availability is checked inside the compute job)"
