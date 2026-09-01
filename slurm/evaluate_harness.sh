#!/bin/bash
# Evaluate Physics plus both PatchHead checkpoints through the harness.
# Physics itself is CPU-capable; this combined job requests a GPU for PatchHead.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=job_harness_eval_%j.out
#SBATCH --error=job_harness_eval_%j.err

set -euo pipefail

REPO="${REPO:-$HOME/tiktok-aigc-detect}"
DATA="${DATA:-$REPO/data/harness_quick}"
BASELINE="${BASELINE:-$REPO/runs/quick_training/baseline/checkpoint.pt}"
AWARE="${AWARE:-$REPO/runs/quick_training/distortion_aware/checkpoint.pt}"
OUT="${OUT:-$REPO/results/harness/quick_evaluation}"

source "$REPO/slurm/_env.sh"
test -f "$DATA/test.csv"
test -f "$BASELINE"
test -f "$AWARE"

python -m harness evaluate \
  --data-dir "$DATA" \
  --baseline-checkpoint "$BASELINE" \
  --aware-checkpoint "$AWARE" \
  --output-dir "$OUT"

echo "=== evaluation complete ==="
cat "$OUT/metrics.json"
