#!/bin/bash
# Evaluate DID on a selected harness manifest.
#SBATCH --partition=gpu
#SBATCH -G h100-47
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=job_did_manifest_eval_%j.out
#SBATCH --error=job_did_manifest_eval_%j.err
set -euo pipefail

REPO="${REPO:-$HOME/tiktok-aigc-detect}"
DATA="${DATA:-$REPO/data/harness_large}"
MANIFEST="${MANIFEST:-matched_test.csv}"
TAG="${TAG:-pooled_sd15_resnet18}"
RECON="${RECON:-sd15}"
RES="${RES:-256}"
STEPS="${STEPS:-10}"
BS="${BS:-32}"
CKPT="${CKPT:-$REPO/checkpoints/did/$TAG.pt}"
OUT="${OUT:-$REPO/results/parallel_evaluation/did}"

source "$REPO/slurm/_env.sh"
if [ ! -f "$DATA/$MANIFEST" ]; then echo "missing evaluation manifest: $DATA/$MANIFEST" >&2; exit 2; fi
if [ ! -f "$CKPT" ]; then echo "missing DID checkpoint: $CKPT" >&2; exit 2; fi

echo "[did] evaluating $MANIFEST through the independent entry point"
python -u -m harness evaluate \
  --data-dir "$DATA" --manifest "$MANIFEST" --models did \
  --did-checkpoint "$CKPT" --did-reconstructor "$RECON" \
  --did-resolution "$RES" --did-steps "$STEPS" --did-batch-size "$BS" \
  --output-dir "$OUT"
echo "DID evaluation complete: $OUT"
