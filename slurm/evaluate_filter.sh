#!/bin/bash
# Evaluate the independent residual/filter baseline on a shared harness manifest.
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=job_filter_eval_%j.out
#SBATCH --error=job_filter_eval_%j.err
set -euo pipefail

REPO="${REPO:-$HOME/DistortAware}"
CHECKPOINT="${CHECKPOINT:-$REPO/filter_based_approach/models/mask_classifier.pt}"
DATA="${DATA:-$REPO/data/harness_large}"
MANIFEST="${MANIFEST:-wildfake_benchmark.csv}"
OUT="${OUT:-$REPO/results/parallel_evaluation/filter}"

source "$REPO/slurm/_env.sh"
if [ ! -f "$CHECKPOINT" ]; then echo "missing filter checkpoint: $CHECKPOINT" >&2; exit 2; fi
if [ ! -f "$DATA/$MANIFEST" ]; then echo "missing evaluation manifest: $DATA/$MANIFEST" >&2; exit 2; fi

echo "[filter] evaluating $MANIFEST"
python -u -m harness evaluate \
  --data-dir "$DATA" --manifest "$MANIFEST" --models filter \
  --filter-checkpoint "$CHECKPOINT" --output-dir "$OUT"

echo "Filter evaluation complete: $OUT"
