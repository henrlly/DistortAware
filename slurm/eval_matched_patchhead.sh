#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=job_eval_matched_%j.out
#SBATCH --error=job_eval_matched_%j.err
#
# Evaluation-only matched PatchHead job. This does NOT train a model.
# Run after the matched checkpoint and manifests already exist.
set -euo pipefail

REPO="${REPO:-$HOME/tiktok-aigc-detect}"
MATCHED_DATA="${MATCHED_DATA:-$REPO/data/matched_refactored}"
TAG="${TAG:-distortion_aware}"
CKPT="${CKPT:-$REPO/patchhead/checkpoints/patchhead_${TAG}.pt}"
WF_OUT="${WF_OUT:-$REPO/results/patchhead/results_${TAG}_wildfake_benchmark}"
SID_OUT="${SID_OUT:-$REPO/results/patchhead/results_${TAG}_sid}"

source "$REPO/slurm/_env.sh"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable on this compute node"; print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)'
python -m unittest discover -s tests -p 'test_*.py'

test -f "$CKPT"
test -f "$MATCHED_DATA/wildfake_benchmark.csv"
test -f "$MATCHED_DATA/sid_eval_200.csv"

python -c 'import sys, torch; p=sys.argv[1]; c=torch.load(p, map_location="cpu", weights_only=False); assert c.get("distortion_aware") is True, f"{p} is not distortion-aware; train the new checkpoint first"; print("verified distortion-aware checkpoint", p, flush=True)' "$CKPT"

echo "=== evaluation-only matched PatchHead ==="
echo "checkpoint=$CKPT"
echo "data=$MATCHED_DATA"
echo "device=$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"

echo "=== WildFake combined benchmark ==="
python patchhead/evaluate.py \
    --manifest "$MATCHED_DATA/wildfake_benchmark.csv" \
    --ckpt "$CKPT" --out "$WF_OUT" --limit 150 --distortion-mode predicted

echo "=== SID 200 images, normal inference ==="
python patchhead/evaluate.py \
    --manifest "$MATCHED_DATA/sid_eval_200.csv" \
    --ckpt "$CKPT" --out "${SID_OUT}_normal" --limit 0 --tta-top-k 0 \
    --distortion-mode predicted

echo "=== SID 200 images, top-3 overlapping-crop TTA ==="
python patchhead/evaluate.py \
    --manifest "$MATCHED_DATA/sid_eval_200.csv" \
    --ckpt "$CKPT" --out "${SID_OUT}_top3" --limit 0 --tta-top-k 3 \
    --distortion-mode predicted

echo "=== SID 200 images, top-5 overlapping-crop TTA ==="
python patchhead/evaluate.py \
    --manifest "$MATCHED_DATA/sid_eval_200.csv" \
    --ckpt "$CKPT" --out "${SID_OUT}_top5" --limit 0 --tta-top-k 5 \
    --distortion-mode predicted

echo "=== DONE: evaluation-only; no training was performed ==="
