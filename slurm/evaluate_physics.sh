#!/bin/bash
# Evaluate the Physics sidecar independently through the harness.
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=job_physics_eval_%j.out
#SBATCH --error=job_physics_eval_%j.err
set -euo pipefail

REPO="${REPO:-$HOME/DistortAware}"
DATA="${DATA:-$REPO/data/harness_large}"
MANIFEST="${MANIFEST:-matched_test.csv}"
OUT="${OUT:-$REPO/results/parallel_evaluation/physics}"

source "$REPO/slurm/_env.sh"
if [ ! -f "$DATA/$MANIFEST" ]; then echo "missing evaluation manifest: $DATA/$MANIFEST" >&2; exit 2; fi

echo "[physics] evaluating $MANIFEST"
python -u -m harness evaluate \
  --data-dir "$DATA" \
  --manifest "$MANIFEST" \
  --models physics \
  --output-dir "$OUT"

echo "Physics evaluation complete: $OUT"
