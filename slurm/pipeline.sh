#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#
# Full DID AIGC-detector pipeline for one (dataset, reconstructor, backbone) combo:
#   feature extraction (resumable, cached by DS/RECON/RES/STEPS) -> train -> eval.
#
#   DS         dataset name -> data/$DS/{train,test}/{real,fake}   (default wildfake)
#   TAG        experiment name -> checkpoints/did_$TAG.pt, results_$TAG/  (default $DS_$RECON_$BACKBONE)
#   RECON      sd15 | sana16          reconstructor family
#   BACKBONE   resnet18 | resnet50    classifier head size
#   RES        reconstruction resolution (default 256)
#   STEPS      DDIM / flow steps (default 10)
#   EPOCHS     training epochs (default 14)
#   LIMIT      per-class cap for the 14-transform test suite (default 150)
#   SKIP_EXTRACT=1   reuse an existing feature cache, go straight to train/eval
#   EVAL_CKPT=path   skip training, evaluate this checkpoint against the cache
#                    (zero-shot cross-dataset: point a wildfake ckpt at a sid cache)
set -e
source "${REPO:-$HOME/tiktok-aigc-detect}/slurm/_env.sh"

DS=${DS:-wildfake}
RECON=${RECON:-sd15}
BACKBONE=${BACKBONE:-resnet18}
RES=${RES:-256}
STEPS=${STEPS:-10}
EPOCHS=${EPOCHS:-14}
LIMIT=${LIMIT:-150}
BATCH=${BATCH:-32}
TAG=${TAG:-${DS}_${RECON}_${BACKBONE}}

CACHE=cache/feat_${DS}_${RECON}_r${RES}s${STEPS}    # shared across backbones
CKPT=checkpoints/did_$TAG.pt
OUT=results/did/$TAG

echo "=== pipeline TAG=$TAG DS=$DS RECON=$RECON BACKBONE=$BACKBONE RES=$RES STEPS=$STEPS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON="--root data/$DS --out $CACHE --res $RES --steps $STEPS --batch $BATCH --recon $RECON"

if [ "${SKIP_EXTRACT:-0}" != "1" ]; then
  echo "[1/4] clean train + test features"
  python did/extract_features.py $COMMON --split train
  python did/extract_features.py $COMMON --split test
  echo "[2/4] randaug1 train features"
  python did/extract_features.py $COMMON --split train --transforms randaug1
  echo "[3/4] transformed test suite"
  python did/extract_features.py $COMMON --split test --limit $LIMIT \
      --transforms jpeg90,jpeg70,jpeg50,jpeg30,blur0.5,blur1.0,blur2.0,resize0.5,resize0.25,noise0.02,noise0.05,noise0.10,jitter,crop80
fi

if [ -n "${EVAL_CKPT:-}" ]; then
  echo "[4/4] zero-shot evaluate $EVAL_CKPT against $CACHE"
  python did/evaluate.py --cache $CACHE --ckpt "$EVAL_CKPT" --backbone $BACKBONE --out $OUT
else
  echo "[4/4] train ($BACKBONE) + evaluate"
  python did/train.py --cache $CACHE --epochs $EPOCHS --bs 32 \
      --train-transforms clean,randaug1 --backbone $BACKBONE --recon $RECON --out $CKPT
  python did/evaluate.py --cache $CACHE --ckpt $CKPT --backbone $BACKBONE --out $OUT
fi
python did/make_report.py --results $OUT

echo "=== DONE $TAG ==="
cat $OUT/metrics.json
