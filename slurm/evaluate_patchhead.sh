#!/bin/bash
# Evaluate both PatchHead variants independently through the harness.
#SBATCH --partition=gpu
#SBATCH -G h100-47
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=job_patchhead_eval_%j.out
#SBATCH --error=job_patchhead_eval_%j.err
set -euo pipefail

REPO="${REPO:-$HOME/DistortAware}"
DATA="${DATA:-$REPO/data/harness_large}"
MANIFEST="${MANIFEST:-matched_test.csv}"
BASELINE="${BASELINE:-$REPO/checkpoints/patchhead/pooled/baseline/checkpoint.pt}"
AWARE="${AWARE:-$REPO/checkpoints/patchhead/pooled/distortion_aware/checkpoint.pt}"
OUT="${OUT:-$REPO/results/parallel_evaluation/patchhead}"
MODELS="${MODELS:-patchhead_baseline,patchhead_distortion_aware}"

source "$REPO/slurm/_env.sh"
if [ ! -f "$DATA/$MANIFEST" ]; then echo "missing evaluation manifest: $DATA/$MANIFEST" >&2; exit 2; fi
case ",$MODELS," in
  *,patchhead_baseline,*) if [ ! -f "$BASELINE" ]; then echo "missing baseline checkpoint: $BASELINE" >&2; exit 2; fi ;;
esac
case ",$MODELS," in
  *,patchhead_distortion_aware,*) if [ ! -f "$AWARE" ]; then echo "missing distortion-aware checkpoint: $AWARE" >&2; exit 2; fi ;;
esac

echo "[patchhead] evaluating $MANIFEST models=$MODELS"
python -u -m harness evaluate \
  --data-dir "$DATA" \
  --manifest "$MANIFEST" \
  --models "$MODELS" \
  --baseline-checkpoint "$BASELINE" \
  --aware-checkpoint "$AWARE" \
  --output-dir "$OUT"

echo "PatchHead evaluation complete: $OUT"
