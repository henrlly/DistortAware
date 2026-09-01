# Shared environment for all SLURM jobs in this repo. `source` it from job scripts.
# Login node has a 1 GB virtual-memory ulimit, so torch can only be imported on
# compute nodes. /tmp is quota-limited on every node -> keep TMPDIR in $HOME.
export REPO="${REPO:-$HOME/tiktok-aigc-detect}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HUB_OFFLINE=1            # weights are pre-fetched on the login node
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
cd "$REPO"
source "$REPO/.venv/bin/activate"
