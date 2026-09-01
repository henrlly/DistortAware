#!/bin/bash
# Evaluate a trained DID checkpoint and generate CSV, JSON, and Markdown.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=job_did_eval_%j.out
#SBATCH --error=job_did_eval_%j.err
set -euo pipefail

REPO="${REPO:-$HOME/tiktok-aigc-detect}"
DATA="${DATA:-$REPO/data/harness_large}"
TAG="${TAG:-pooled_sd15_resnet18}"
RECON="${RECON:-sd15}"
RES="${RES:-256}"
STEPS="${STEPS:-10}"
BS="${BS:-32}"
STAGE="${STAGE:-$REPO/runs/did_data/$TAG}"
CACHE="${CACHE:-$REPO/cache/did/$TAG}"
CKPT="${CKPT:-$REPO/checkpoints/did/$TAG.pt}"
OUT="${OUT:-$REPO/results/did/$TAG}"
TF="${TF:-jpeg90,jpeg70,jpeg50,jpeg30,blur0.5,blur1.0,blur2.0,resize0.5,resize0.25,noise0.02,noise0.05,noise0.10,jitter,crop80}"

source "$REPO/slurm/_env.sh"
test -f "$CKPT"
test -d "$STAGE/test"
python did/extract_features.py --root "$STAGE" --split test --out "$CACHE" \
  --res "$RES" --steps "$STEPS" --batch "$BS" --recon "$RECON" \
  --transforms "$TF"
python did/evaluate.py --cache "$CACHE" --ckpt "$CKPT" --out "$OUT"
python did/make_report.py --results "$OUT"
echo "DID report: $OUT/report.md"
