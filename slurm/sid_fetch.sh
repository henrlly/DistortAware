#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#
# Parse the cached SID_Set parquet shards into data/sid_set/{train,test}/{real,fake}.
# Parquet download must have happened first (slurm/dl_sid.py on the login node).
# CPU-only — parsing just needs RAM the login node doesn't allow.
set -e
source "${REPO:-$HOME/tiktok-aigc-detect}/slurm/_env.sh"
python did/fetch_sid_set.py --out data/sid_set --split validation --train 300 --test 150 "$@"
echo "=== counts ==="
for d in train/real train/fake test/real test/fake; do
  echo "$d: $(ls data/sid_set/$d 2>/dev/null | wc -l)"
done
