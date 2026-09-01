#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#
# Fast de-risk for the PatchHead pipeline: build the model, forward + backward a
# batch, run 1 mini training epoch on a tiny slice, eval clean only, compare.
set -e
source "${REPO:-$HOME/tiktok-aigc-detect}/slurm/_env.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch, timm
from patchhead.model import PatchHeadDetector, get_device
d = get_device(); print("device", d, "timm", timm.__version__)
m = PatchHeadDetector(lora_r=8).to(d)
print("param counts:", m.param_counts())
x = torch.rand(4, 3, 256, 256, device=d)
with torch.autocast(device_type=d, dtype=torch.bfloat16):
    il, cl, pl = m(x)
    loss = il.mean() + cl.mean() + pl.mean()
loss.backward()
g = sum(p.grad.abs().sum().item() for p in m.trainable_parameters() if p.grad is not None)
print("forward ok", il.shape, cl.shape, pl.shape, "grad-sum", g)
assert g > 0
PY

echo "--- 1-epoch tiny train ---"
python patchhead/train.py --ds wildfake --epochs 1 --bs 8 \
    --out patchhead/checkpoints/patchhead_smoke.pt

echo "--- eval clean-only + compare ---"
python patchhead/evaluate.py --ds wildfake --ckpt patchhead/checkpoints/patchhead_smoke.pt \
    --out results/patchhead/results_smoke --limit 20
python harness/did_predictions.py --cache cache/feat_wildfake_sd15_r256s10 \
    --ckpt checkpoints/did_sd15_resnet18.pt --ds wildfake --out results/patchhead/did_preds_smoke.json
python patchhead/compare.py --patchhead results/patchhead/results_smoke/preds_clean.json \
    --did results/patchhead/did_preds_smoke.json --out results/patchhead/results_compare_smoke
echo "=== SMOKE OK ==="
