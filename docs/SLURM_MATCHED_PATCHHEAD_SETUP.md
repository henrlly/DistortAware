
# Matched PatchHead: SLURM Setup (legacy)

> The supported workflow is now documented in [`PATCHHEAD_WORKFLOW.md`](PATCHHEAD_WORKFLOW.md).
> This older guide describes the pre-refactor workflow and is retained only for historical runs.

This workflow fetches data directly on the server. Do not copy or upload the
Windows `data/` directory.

## 1. Verify the server

```bash
ssh YOUR_USER@YOUR_CLUSTER
command -v python3 && python3 --version
command -v git && git --version
command -v gh && gh --version
command -v uv && uv --version
```

Python 3.11+, Git, GitHub CLI, and `uv` are required. If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

## 2. GitHub setup

```bash
gh auth login
gh auth status
gh repo clone henrlly/tiktok-aigc-detect "$HOME/tiktok-aigc-detect"
cd "$HOME/tiktok-aigc-detect"
git switch matched-patchhead-experiment
```

The fetch script separately clones the DINOv3 repository's
`no-tampered-training` branch into `$HOME/aicg-classifier`.

## 3. Create the virtual environment

```bash
cd "$HOME/tiktok-aigc-detect"
python3 -m venv .venv
source .venv/bin/activate
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

python -m pip install --upgrade pip
python -m pip install \
  torch torchvision diffusers transformers accelerate safetensors \
  scikit-learn timm pillow tqdm matplotlib \
  huggingface-hub pyarrow datasets kagglehub modelscope
```

Verify the environment:

```bash
python --version
python -c "import torch, timm, datasets, kagglehub, modelscope; print('environment ok')"
```

## 4. Log in to Hugging Face and Kaggle

Hugging Face:

```bash
hf auth login
```

Alternatively:

```bash
export HF_TOKEN='YOUR_HUGGINGFACE_TOKEN'
```

Kaggle:

```bash
kaggle auth login
```

Or:

```bash
export KAGGLE_USERNAME='YOUR_KAGGLE_USERNAME'
export KAGGLE_KEY='YOUR_KAGGLE_API_KEY'
```

For newer Kaggle tokens, use `KAGGLE_API_TOKEN` instead. Never commit tokens
or place them in an `sbatch` command. These credentials are needed during
fetching and do not need to be passed to the offline GPU job.

## 5. Pre-fetch DINOv3 weights

Compute nodes are offline, so download the PatchHead backbone on the login
node:

```bash
cd "$HOME/tiktok-aigc-detect"
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
export HF_HOME="$HOME/.cache/huggingface"

python -c "
from huggingface_hub import hf_hub_download
for filename in ('config.json', 'model.safetensors'):
    print(hf_hub_download('timm/vit_large_patch16_dinov3.lvd1689m', filename))
"
```

## 6. Fetch data on the server

This uses the DINOv3 repository's existing gather logic and seed 42:

```bash
cd "$HOME/tiktok-aigc-detect"
export REPO="$HOME/tiktok-aigc-detect"
export AIGC_REPO="$HOME/aicg-classifier"
export MATCHED_DATA="$REPO/data/matched_server"
export SEED=42

bash slurm/fetch_matched_data.sh
```

Expected counts:

```text
train:       4,374 records
validation:  1,313 records
calibration: 1,313 records
test:        1,750 records
SID eval:      200 records
WildFake:      500 COCO + 500 DALLE
```

## 7. Submit the long-running GPU job

`sbatch` returns immediately; SLURM continues the job in the background:

```bash
cd "$HOME/tiktok-aigc-detect"
sbatch --export=ALL,\
REPO="$HOME/tiktok-aigc-detect",\
MATCHED_DATA="$HOME/tiktok-aigc-detect/data/matched_server",\
TAG=matched_local,\
EPOCHS=10 \
slurm/matched_patchhead.sh
```

Monitor it:

```bash
squeue -u "$USER"
squeue -j JOB_ID
tail -f "job_matched_JOB_ID.out"
tail -f "job_matched_JOB_ID.err"
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS
```

## 8. Outputs

```text
patchhead/checkpoints/patchhead_matched_local.pt
results/patchhead/results_matched_local_wildfake_coco/
results/patchhead/results_matched_local_wildfake_dalle/
results/patchhead/results_matched_local_sid_top3/
```

The SID run uses top-3 TTA: the full image plus overlapping 50%-size crops on
a 3×3 grid; the strongest three crop scores are averaged with the full-image
score.

## Environment rules

`$HOME` works inside SLURM jobs and is used by `slurm/_env.sh`. Override paths
with `--export=ALL,REPO=...,MATCHED_DATA=...` when needed. User-specific paths
belong in the script body or environment file, not in `#SBATCH` lines.

The fetch step needs Hugging Face/Kaggle credentials and network access. The
GPU job sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so credentials do
not need to be forwarded to training.
