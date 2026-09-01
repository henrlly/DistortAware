# PatchHead Remote Server Runbook

## Update and prepare

```bash
cd ~/tiktok-aigc-detect
git fetch origin
git switch matched-patchhead-experiment
git pull --ff-only origin matched-patchhead-experiment
source .venv/bin/activate
python -m pip check
```

If the branch is not present locally, run `git switch --track origin/matched-patchhead-experiment`.

Verify credentials with `hf auth whoami` and `kaggle datasets list`. Run
`hf auth login` or `kaggle auth login` if needed.

## Cache model weights

On the network-enabled login node:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import hf_hub_download; [print(hf_hub_download('timm/vit_large_patch16_dinov3.lvd1689m', f)) for f in ('config.json', 'model.safetensors')]"
```

## Fetch data

```bash
cd ~/tiktok-aigc-detect
bash slurm/fetch_matched_data.sh
```

This creates `data/matched_refactored/` using the TikTok repository’s own
fetcher. It does not clone or execute the DINOv3 repository. Normal reruns use
the cache. Keep the old `data/matched_server/` directory until the new fetch
and preflight succeed.

The official benchmark is one combined manifest containing 500 COCO real and
500 DALL-E Advanced fake images. ImageNet, CelebA-HQ, AFHQ, ADM, DDIM, DDPM,
and VQDM are training/validation data only.

## Preflight

```bash
bash slurm/preflight.sh
```

Preflight checks manifests, the virtual environment, Python compilation, and
dependencies. CUDA is checked inside the compute job.

## Submit the full workflow

```bash
sbatch -G h100-47 --export=ALL slurm/matched_patchhead.sh
```

This runs training, the combined WildFake benchmark, SID normal evaluation,
and SID top-3 overlapping-crop TTA evaluation. It does not refetch data.

The first job output should contain `cuda=...` and `device=cuda`.

## Monitor SLURM

For job ID `123456`:

```bash
squeue -j 123456
tail -f job_matched_123456.out
tail -f job_matched_123456.err
sacct -j 123456 --format=JobID,State,Elapsed,MaxRSS
```

## Evaluation only

If the checkpoint already exists:

```bash
sbatch -G h100-47 --export=ALL slurm/eval_matched_patchhead.sh
```

This skips training and evaluates the existing checkpoint on the combined
WildFake benchmark and SID with normal and top-3 TTA inference.
