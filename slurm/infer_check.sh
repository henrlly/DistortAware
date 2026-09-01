#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
set -e
source "${REPO:-$HOME/tiktok-aigc-detect}/slurm/_env.sh"
D=$TMPDIR/infer_check; rm -rf $D; mkdir -p $D
cp $(ls data/wildfake/test/real/*.png | head -6) $D/
cp $(ls data/wildfake/test/fake/*.png | head -6) $D/
python infer.py --image-dir $D --out results/did/sd15_resnet18/sample_preds.json --ckpt checkpoints/did.pt
echo "=== sample_preds.json ==="
python -c "import json;[print(f\"{d['pred']:.3f}  {'AIGC' if d['is_aigc'] else 'real'}  {d['image_path'].split('/')[-1]}\") for d in json.load(open('results/did/sd15_resnet18/sample_preds.json'))]"
