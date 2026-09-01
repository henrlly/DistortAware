#!/bin/bash
# Train baseline and distortion-aware PatchHead on the full harness dataset.
#SBATCH --partition=gpu
#SBATCH -G h100-47
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=job_patchhead_train_%j.out
#SBATCH --error=job_patchhead_train_%j.err
set -euo pipefail

REPO="${REPO:-$HOME/DistortAware}"
DATA="${DATA:-$REPO/data/harness_large}"
OUT="${OUT:-$REPO/checkpoints/patchhead/pooled}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-16}"
MODE="${MODE:-both}"
INIT="${INIT:-}"

source "$REPO/slurm/_env.sh"
for manifest in train.csv validation.csv calibration.csv; do
  test -f "$DATA/$manifest"
done
test -f "$DATA/test.csv" || test -f "$DATA/matched_test.csv"

echo "[patchhead] training mode=$MODE"
command=(python -u -m harness train-patchhead --data-dir "$DATA" --mode "$MODE" \
  --epochs "$EPOCHS" --bs "$BS" --output-dir "$OUT")
if [ -n "$INIT" ]; then
  command+=(--init-checkpoint "$INIT")
fi
"${command[@]}"

echo "PatchHead training complete: $OUT"
