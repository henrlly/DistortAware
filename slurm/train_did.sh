#!/bin/bash
# Train the DID classifier on the shared 3,000-per-class harness dataset.
# The diffusion reconstructor creates cached features; did/train.py trains the
# two ResNet heads on those features.
#SBATCH --partition=gpu
#SBATCH -G h100-47
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=job_did_train_%j.out
#SBATCH --error=job_did_train_%j.err
set -euo pipefail

REPO="${REPO:-$HOME/DistortAware}"
DATA="${DATA:-$REPO/data/harness_large}"
TAG="${TAG:-pooled_sd15_resnet18}"
RECON="${RECON:-sd15}"
RES="${RES:-256}"
STEPS="${STEPS:-10}"
BS="${BS:-32}"
EPOCHS="${EPOCHS:-14}"
BACKBONE="${BACKBONE:-resnet18}"
STAGE="${STAGE:-$REPO/runs/did_data/$TAG}"
CACHE="${CACHE:-$REPO/cache/did/$TAG}"
CKPT="${CKPT:-$REPO/checkpoints/did/$TAG.pt}"

source "$REPO/slurm/_env.sh"
test -f "$DATA/train.csv"
test -f "$DATA/test.csv" || test -f "$DATA/matched_test.csv"

echo "[did] preparing binary dataset"
python -u -m harness.did_data --data-dir "$DATA" --output-dir "$STAGE"
echo "[did] extracting clean training features"
python -u did/extract_features.py --root "$STAGE" --split train --out "$CACHE" \
  --res "$RES" --steps "$STEPS" --batch "$BS" --recon "$RECON"
echo "[did] extracting augmented training features"
python -u did/extract_features.py --root "$STAGE" --split train --out "$CACHE" \
  --res "$RES" --steps "$STEPS" --batch "$BS" --recon "$RECON" --transforms randaug1
echo "[did] extracting test features"
python -u did/extract_features.py --root "$STAGE" --split test --out "$CACHE" \
  --res "$RES" --steps "$STEPS" --batch "$BS" --recon "$RECON"
echo "[did] training classifier"
python -u did/train.py --cache "$CACHE" --epochs "$EPOCHS" --bs "$BS" \
  --train-transforms clean,randaug1 --backbone "$BACKBONE" --recon "$RECON" --out "$CKPT"

echo "DID training complete: $CKPT"
