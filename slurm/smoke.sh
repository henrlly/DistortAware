#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
set -e
source "${REPO:-$HOME/tiktok-aigc-detect}/slurm/_env.sh"
RECON=${RECON:-sd15}
BACKBONE=${BACKBONE:-resnet18}
RES=${RES:-256}
C=cache/smoke_${RECON}
rm -rf "$C" checkpoints/smoke_${RECON}.pt results/did/smoke_${RECON}
COMMON="--root data/wildfake --out $C --res $RES --steps 6 --batch 8 --recon $RECON --limit 8"
python did/extract_features.py $COMMON --split train
python did/extract_features.py $COMMON --split test
python did/extract_features.py $COMMON --split train --transforms randaug1
python did/train.py --cache $C --epochs 2 --bs 8 --train-transforms clean,randaug1 \
    --backbone $BACKBONE --recon $RECON --out checkpoints/smoke_${RECON}.pt
python did/extract_features.py $COMMON --split test --transforms jpeg70,blur1.0
python did/evaluate.py --cache $C --ckpt checkpoints/smoke_${RECON}.pt --out results/did/smoke_${RECON}
python did/make_report.py --results results/did/smoke_${RECON}
echo "SMOKE OK $RECON"
