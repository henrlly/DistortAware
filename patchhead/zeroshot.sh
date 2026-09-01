#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
# Zero-shot cross-dataset: PatchHead trained on one dataset, evaluated on the other.
# Mirrors the DID zero-shot study (results_*_zeroshot_from_*).
set -e
source "${REPO:-$HOME/DistortAware}/slurm/_env.sh"

# wildfake-trained -> SID_Set test  (compared against the SID-native DID model)
python patchhead/evaluate.py --ds sid_set --ckpt patchhead/checkpoints/patchhead_wildfake.pt \
    --out results/patchhead/results_sid_zeroshot_from_wildfake --limit 150
python patchhead/compare.py \
    --patchhead results/patchhead/results_sid_zeroshot_from_wildfake/preds_clean.json \
    --did results/patchhead/did_preds_sid_set.json \
    --name-a "PatchHead(WF->SID)" --name-b "DID(SID-native)" \
    --out results/patchhead/results_compare_sid_zeroshot_from_wildfake || true

# SID-trained -> WildFake test
python patchhead/evaluate.py --ds wildfake --ckpt patchhead/checkpoints/patchhead_sid_set.pt \
    --out results/patchhead/results_wildfake_zeroshot_from_sid --limit 150
python patchhead/compare.py \
    --patchhead results/patchhead/results_wildfake_zeroshot_from_sid/preds_clean.json \
    --did results/patchhead/did_preds_wildfake.json \
    --name-a "PatchHead(SID->WF)" --name-b "DID(WF-native)" \
    --out results/patchhead/results_compare_wildfake_zeroshot_from_sid || true

echo "=== zeroshot metrics ==="
echo "WF->SID:"; cat results/patchhead/results_sid_zeroshot_from_wildfake/metrics.json
echo "SID->WF:"; cat results/patchhead/results_wildfake_zeroshot_from_sid/metrics.json
