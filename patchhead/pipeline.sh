#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#
# PatchHead detector (frozen DINOv3 ViT-L/16 + LoRA + spatial patch head).
# Trains on raw images -- no DID feature cache needed -- then evaluates on the
# same 14-transform suite and compares image-for-image with the DID detector.
#
#   DS         wildfake | sid_set | pooled     (default wildfake)
#   TAG        experiment name                 (default $DS)
#   EPOCHS     training epochs                 (default 10)
#   LORA_R     LoRA rank                       (default 8)
#   SIZE       model input resolution          (default 256)
#   LIMIT      per-class cap, transformed suite (default 150)
#   DID_CKPT   DID checkpoint to compare against (default checkpoints/did_${DS}_sd15_resnet18.pt)
#   DID_CACHE  DID feature cache for that ckpt  (default cache/feat_${DS}_sd15_r256s10)
#   SKIP_TRAIN=1   reuse patchhead/checkpoints/patchhead_$TAG.pt
set -e
source "${REPO:-$HOME/DistortAware}/slurm/_env.sh"

DS=${DS:-wildfake}
TAG=${TAG:-$DS}
EPOCHS=${EPOCHS:-10}
LORA_R=${LORA_R:-8}
SIZE=${SIZE:-256}
LIMIT=${LIMIT:-150}
BS=${BS:-16}
# DID detector to compare against: (ckpt, feature cache, metrics.json) per dataset
case "$DS" in
  wildfake) DID_CKPT_D=checkpoints/did_sd15_resnet18.pt;          DID_METRICS_D=results/did/sd15_resnet18/metrics.json ;;
  sid_set)  DID_CKPT_D=checkpoints/did_sid_set_sd15_resnet18.pt;  DID_METRICS_D=results/did/sid_set_sd15_resnet18/metrics.json ;;
  pooled)   DID_CKPT_D=checkpoints/did_pooled_sd15_resnet18.pt;   DID_METRICS_D=results/did/pooled_on_wildfake/metrics.json ;;
  *)        DID_CKPT_D=checkpoints/did_${DS}_sd15_resnet18.pt;    DID_METRICS_D=results/did/${DS}_sd15_resnet18/metrics.json ;;
esac
DID_CKPT=${DID_CKPT:-$DID_CKPT_D}
DID_CACHE=${DID_CACHE:-cache/feat_${DS}_sd15_r256s10}
DID_METRICS=${DID_METRICS:-$DID_METRICS_D}

CKPT=patchhead/checkpoints/patchhead_$TAG.pt
OUT=results/patchhead/$TAG
DIDP=results/patchhead/did_preds_$TAG.json
CMP=results/patchhead/compare_$TAG

echo "=== patchhead TAG=$TAG DS=$DS EPOCHS=$EPOCHS LORA_R=$LORA_R SIZE=$SIZE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import timm, torch; print('timm', timm.__version__, 'torch', torch.__version__)"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "[1/4] train"
  python patchhead/train.py --ds $DS --epochs $EPOCHS --bs $BS --lora-r $LORA_R \
      --size $SIZE --out $CKPT
fi

echo "[2/4] evaluate (clean + 14 transforms)"
python patchhead/evaluate.py --ds $DS --ckpt $CKPT --out $OUT --limit $LIMIT
python patchhead/make_report.py --results $OUT --did-metrics $DID_METRICS

if [ -f "$DID_CKPT" ] && [ -d "$DID_CACHE" ]; then
  echo "[3/4] DID per-image predictions from $DID_CKPT"
  python harness/did_predictions.py --cache $DID_CACHE --ckpt $DID_CKPT --ds $DS --out $DIDP
  echo "[4/4] compare"
  python patchhead/compare.py --patchhead $OUT/preds_clean.json --did $DIDP \
      --patchhead-metrics $OUT/metrics.json --did-metrics $DID_METRICS \
      --out $CMP
else
  echo "[3/4] SKIP compare -- DID ckpt ($DID_CKPT) or cache ($DID_CACHE) missing"
fi

echo "=== DONE $TAG ==="
cat $OUT/metrics.json
[ -f $CMP/comparison.json ] && python -c "import json;d=json.load(open('$CMP/comparison.json'));print({k:d[k] for k in ['acc_a','acc_b','both_wrong','only_a_wrong','only_b_wrong','error_phi_coefficient','oracle_upper_bound_acc']})"
