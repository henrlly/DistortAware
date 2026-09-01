#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=job_matched_%j.out
#SBATCH --error=job_matched_%j.err
#
# Full PatchHead workflow: train on manifests, then evaluate held-out benchmarks.
#
# Run slurm/fetch_matched_data.sh first on the server/login node. Override
# MATCHED_DATA, TAG, or EPOCHS with --export=ALL,KEY=value.
set -euo pipefail

source "${REPO:-$HOME/DistortAware}/slurm/_env.sh"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable on this compute node"; print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)'
python -m unittest discover -s tests -p 'test_*.py'

MATCHED_DATA="${MATCHED_DATA:-$REPO/data/matched_refactored}"
TAG="${TAG:-distortion_aware}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-16}"
SIZE="${SIZE:-256}"
CKPT="$REPO/patchhead/checkpoints/patchhead_${TAG}.pt"
WF_OUT="$REPO/results/patchhead/results_${TAG}_wildfake_benchmark"
SID_OUT="$REPO/results/patchhead/results_${TAG}_sid"

test -f "$MATCHED_DATA/train.csv"
test -f "$MATCHED_DATA/validation.csv"
test -f "$MATCHED_DATA/calibration.csv"
test -f "$MATCHED_DATA/wildfake_benchmark.csv"
test -f "$MATCHED_DATA/sid_eval_200.csv"

echo "=== matched PatchHead data=$MATCHED_DATA tag=$TAG epochs=$EPOCHS ==="
python patchhead/train.py --manifest-dir "$MATCHED_DATA" --epochs "$EPOCHS" \
    --bs "$BS" --size "$SIZE" --mask-crop-prob "${MASK_CROP_PROB:-0.5}" \
    --mask-padding "${MASK_PADDING:-0.20}" --distortion-aware --out "$CKPT"

python -c 'import sys, torch; p=sys.argv[1]; c=torch.load(p, map_location="cpu", weights_only=False); assert c.get("distortion_aware") is True, f"{p} is not distortion-aware"; print("verified distortion-aware checkpoint", p, flush=True)' "$CKPT"

echo "=== WildFake combined benchmark: 1000 images ==="
python patchhead/evaluate.py --manifest "$MATCHED_DATA/wildfake_benchmark.csv" \
    --ckpt "$CKPT" --out "$WF_OUT" --limit 150 --distortion-mode predicted

echo "=== SID: exact 200-image subset, normal inference ==="
python patchhead/evaluate.py --manifest "$MATCHED_DATA/sid_eval_200.csv" \
    --ckpt "$CKPT" --out "${SID_OUT}_normal" --limit 0 --tta-top-k 0 \
    --distortion-mode predicted

echo "=== SID: exact 200-image subset, top-3 overlapping-crop TTA ==="
python patchhead/evaluate.py --manifest "$MATCHED_DATA/sid_eval_200.csv" \
    --ckpt "$CKPT" --out "${SID_OUT}_top3" --limit 0 --tta-top-k 3 \
    --distortion-mode predicted

echo "=== SID: exact 200-image subset, top-5 overlapping-crop TTA ==="
python patchhead/evaluate.py --manifest "$MATCHED_DATA/sid_eval_200.csv" \
    --ckpt "$CKPT" --out "${SID_OUT}_top5" --limit 0 --tta-top-k 5 \
    --distortion-mode predicted

echo "=== DONE ==="
cat "$WF_OUT/metrics.json"
cat "${SID_OUT}_normal/metrics.json"
cat "${SID_OUT}_top3/metrics.json"
cat "${SID_OUT}_top5/metrics.json"
