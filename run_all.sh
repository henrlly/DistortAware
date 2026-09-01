#!/usr/bin/env bash
# End-to-end reproduction of the DID AIGC detector POC (single machine).
# On the SoC SLURM cluster use `sbatch slurm/pipeline.sh` instead — see slurm/README.md.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/did:${PYTHONPATH:-}"

RECON=${RECON:-sd15}          # sd15 | sana16
BACKBONE=${BACKBONE:-resnet18} # resnet18 | resnet50
RES=${RES:-256}                # 512 for sana16
STEPS=${STEPS:-10}             # 7 for sana16
CACHE=cache/feat_${RECON}
CKPT=checkpoints/did.pt
OUT=results/did/legacy_run
TF=jpeg90,jpeg70,jpeg50,jpeg30,blur0.5,blur1.0,blur2.0,resize0.5,resize0.25,noise0.02,noise0.05,noise0.10,jitter,crop80

echo "[1/6] Sampling WildFake subset (HTTP range reads, no full download)"
python3 did/fetch_wildfake.py --out data/wildfake --train 300 --test 150

echo "[2/6] DID features: clean train + test"
python3 did/extract_features.py --root data/wildfake --split train --out "$CACHE" --res $RES --steps $STEPS --recon $RECON
python3 did/extract_features.py --root data/wildfake --split test  --out "$CACHE" --res $RES --steps $STEPS --recon $RECON

echo "[3/6] DID features: randaug1 train (robustness augmentation)"
python3 did/extract_features.py --root data/wildfake --split train --out "$CACHE" --res $RES --steps $STEPS --recon $RECON --transforms randaug1

echo "[4/6] DID features: 14-transform test suite"
python3 did/extract_features.py --root data/wildfake --split test --out "$CACHE" --res $RES --steps $STEPS --recon $RECON --limit 150 --transforms "$TF"

echo "[5/6] Train the two-head classifier"
python3 did/train.py --cache "$CACHE" --epochs 14 --bs 32 --train-transforms clean,randaug1 --backbone $BACKBONE --recon $RECON --out "$CKPT"

echo "[6/6] Evaluate + report"
python3 did/evaluate.py --cache "$CACHE" --ckpt "$CKPT" --out "$OUT"
python3 did/make_report.py --results "$OUT"

echo "Done. See $OUT/report.md, $OUT/robustness.csv, $OUT/metrics.json"
