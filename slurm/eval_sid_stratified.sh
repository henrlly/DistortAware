#!/bin/bash
# Evaluation-only: assumes fetch_sid_eval.sh has already created the manifest.
# Runs the same held-out SID images with normal, top-3, and top-5 TTA.
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=job_sid_stratified_%j.out
#SBATCH --error=job_sid_stratified_%j.err
set -euo pipefail

REPO="${REPO:-$HOME/tiktok-aigc-detect}"
DATA="${DATA:-$REPO/data/matched_refactored}"
CKPT="${CKPT:-$REPO/patchhead/checkpoints/patchhead_3class_maskcrop.pt}"
OUT="${OUT:-$REPO/results/patchhead/results_3class_sid_stratified}"
source "$REPO/slurm/_env.sh"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
test -f "$CKPT"
test -f "$DATA/sid_eval_stratified.csv"

for spec in normal:0 top3:3 top5:5; do
  name="${spec%%:*}"; k="${spec##*:}"
  python patchhead/evaluate.py --manifest "$DATA/sid_eval_stratified.csv" \
    --ckpt "$CKPT" --out "${OUT}_${name}" --limit 0 --tta-top-k "$k"
done
