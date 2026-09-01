#!/bin/bash
# Quick baseline versus distortion-aware PatchHead comparison.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=job_patchhead_quick_%j.out
#SBATCH --error=job_patchhead_quick_%j.err

# This is intentionally a GPU job: fetching remains CPU-only.
set -euo pipefail

REPO="${REPO:-$HOME/DistortAware}"
DATA="${DATA:-$REPO/data/harness_quick}"
OUT="${OUT:-$REPO/checkpoints/patchhead/quick}"
EPOCHS="${EPOCHS:-1}"
BS="${BS:-16}"

source "$REPO/slurm/_env.sh"
test -f "$DATA/train.csv"
test -f "$DATA/validation.csv"
test -f "$DATA/calibration.csv"
test -f "$DATA/test.csv"

python -m harness train-patchhead \
  --data-dir "$DATA" \
  --mode both \
  --epochs "$EPOCHS" \
  --bs "$BS" \
  --output-dir "$OUT"

echo "=== quick training complete ==="
find "$OUT" -maxdepth 2 -type f -print | sort
