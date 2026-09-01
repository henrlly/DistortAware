#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=job_harness_fetch_%j.out
#SBATCH --error=job_harness_fetch_%j.err

# CPU-only, network-enabled dataset materialisation for the independent harness.
# Do not source slurm/_env.sh: fetch jobs must not inherit HF_HUB_OFFLINE=1.
#
# Submit with:
#   sbatch --export=ALL,REPO=$HOME/tiktok-aigc-detect slurm/fetch_harness_data.sh
set -euo pipefail

REPO="${REPO:-$HOME/tiktok-aigc-detect}"
DATA="${DATA:-$REPO/data/harness_large}"
SEED="${SEED:-42}"
BASE_PER_CLASS="${BASE_PER_CLASS:-3000}"
SID_PER_CLASS="${SID_PER_CLASS:-3000}"
WILDFake_PER_SOURCE="${WILDFake_PER_SOURCE:-3000}"
BENCHMARK_COUNT="${BENCHMARK_COUNT:-500}"
QUICK_PER_CLASS="${QUICK_PER_CLASS:-200}"
QUICK_DATA="${QUICK_DATA:-$REPO/data/harness_quick}"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$TMPDIR" "$HF_HOME"

FETCH_EXTRA=()
if [[ "${REFRESH:-0}" == "1" ]]; then
  FETCH_EXTRA+=(--refresh)
fi

echo "=== harness fetch ==="
echo "repo=$REPO data=$DATA quick=$QUICK_DATA"
echo "base=$BASE_PER_CLASS sid=$SID_PER_CLASS wildfake=$WILDFake_PER_SOURCE"

python - <<'PY'
import os
import urllib.request

url = os.environ.get("HARNESS_NETWORK_CHECK", "https://huggingface.co")
try:
    with urllib.request.urlopen(url, timeout=15) as response:
        print(f"network_check={response.status} {url}")
except Exception as exc:
    raise SystemExit(f"network check failed for {url}: {exc}")
PY

python -m harness fetch \
  --output-dir "$DATA" \
  --quick-output-dir "$QUICK_DATA" \
  --base-per-class "$BASE_PER_CLASS" \
  --sid-per-class "$SID_PER_CLASS" \
  --wildfake-per-source "$WILDFake_PER_SOURCE" \
  --benchmark-count "$BENCHMARK_COUNT" \
  --quick-per-class "$QUICK_PER_CLASS" \
  --seed "$SEED" \
  "${FETCH_EXTRA[@]}"

echo "=== fetch complete ==="
find "$DATA" "$QUICK_DATA" -maxdepth 1 -type f -name '*.csv' -print | sort
