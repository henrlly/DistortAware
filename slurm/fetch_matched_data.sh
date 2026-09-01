#!/bin/bash
# Login-node data preparation. PatchHead owns all fetch and split logic.
set -euo pipefail

REPO="${REPO:-$HOME/tiktok-aigc-detect}"
DATA="${MATCHED_DATA:-$REPO/data/matched_refactored}"
SEED="${SEED:-42}"
BASE_QUOTA="${BASE_QUOTA:-1250}"
WILDFake_QUOTA="${WILDFake_QUOTA:-1000}"
BENCHMARK_COUNT="${BENCHMARK_COUNT:-500}"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

echo "=== PatchHead fetch ==="
echo "data=$DATA seed=$SEED base_quota=$BASE_QUOTA wildfake_quota=$WILDFake_QUOTA"
python patchhead/cli.py fetch \
  --output-dir "$DATA" \
  --base-quota "$BASE_QUOTA" \
  --wildfake-quota "$WILDFake_QUOTA" \
  --benchmark-count "$BENCHMARK_COUNT" \
  --sid-count 200 \
  --seed "$SEED"
