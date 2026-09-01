# Running this repo on the SoC SLURM cluster

This is the operational guide for training / evaluating the DID AIGC detector on
the login-node + SLURM GPU cluster. Written for a fresh agent or human.

## TL;DR

```bash
# one-time setup already done: .venv/ built, WildFake sampled into data/wildfake,
# SD-1.5 + resnet weights cached under ~/.cache. See "One-time setup" if missing.

sbatch slurm/test_gpu.sh          # 2-min sanity check (torch+cuda+SD load)
sbatch slurm/pipeline.sh          # full pipeline -> results/did/<tag>/
squeue -u $USER                   # watch it
tail -f job_<id>.out              # live log  (job_<id>.err = stderr)
scancel <id>                      # kill it
```

## Cluster facts that bit us (read this)

| Thing | Detail |
|---|---|
| **Login node has a 1 GB virtual-memory `ulimit -v`** | `import torch` **fails on the login node** (`libtorch_cuda.so: failed to map segment`). All torch work must go through `sbatch`. `pip install`, `huggingface_hub` downloads, and the WildFake HTTP sampler *do* work on the login node. |
| **`/tmp` is quota-limited (~10 MB) on every node** | Downloads/extractions die silently at ~5 MB. Every job sets `TMPDIR=$HOME/tmp`. Do the same for any ad-hoc command. |
| **`ulimit -u 64` on the login node** | `hf_xet` / `hf_transfer` spawn dozens of threads and crash with "failed to spawn thread". We download with `HF_HUB_DISABLE_XET=1` and `max_workers<=3`. |
| **Compute nodes: assume no outbound internet** | Pre-fetch all model weights on the login node into `~/.cache/huggingface`; jobs run with `HF_HUB_OFFLINE=1`. |
| No `module` system | Use `spack` (system/MPI/CUDA deps) + a pip `.venv` for the Python DL stack. |

## Dependency management

`spack` is the cluster's tool for system-level deps (compilers, CUDA, MPI) and is
used that way by the sibling `sc26/SC26-MFC` project. For **this** repo the heavy
deps are PyTorch + diffusers, whose pip wheels already bundle their own CUDA 12.4
runtime, so we use a plain venv built with the system Python 3.12:

```bash
/usr/bin/python3 -m venv .venv
export TMPDIR=$HOME/tmp
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install diffusers transformers accelerate safetensors scikit-learn \
    timm pillow tqdm matplotlib huggingface-hub pyarrow
```

No spack env is needed at runtime — the GPU nodes' NVIDIA driver + the wheel's
bundled CUDA libs are enough (`slurm/test_gpu.sh` proves it). If you ever do need
spack deps, `spack env activate <env>` before `source .venv/bin/activate`.

## SLURM crib sheet

```bash
sbatch slurm/pipeline.sh                       # submit
sbatch --export=ALL,TAG=foo,RES=192 slurm/pipeline.sh   # override knobs
squeue -u $USER                                # my queue
squeue -j <id> ; scontrol show job <id>        # detail
scancel <id>                                   # cancel
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS   # after it ends
sinfo -p gpu -o "%n %G %t"                     # what GPUs are free
```

### GPU / partition selection

- Request a specific GPU type with `#SBATCH -G <type>`: `a100-40`, `a100-80`,
  `h100-47`, `h100-96`. **Prefer `a100-40` / `h100-47`** (the MIG slices — shorter
  queue, plenty for this workload; it also happily runs on a full card if that's
  what the scheduler hands you).
- Partitions: `gpu` (3 h max, 15 min default, high priority, +200 nice for
  <=15 min jobs) and `gpu-long` (3 days, 5 h default, lower priority).
  **Prefer `gpu`** — the whole pipeline fits in well under 3 h.
- Our scripts request `-G a100-40 --partition=gpu --time=03:00:00`.

## The pipeline script

`slurm/pipeline.sh` runs, for one (reconstructor, backbone) combo:

1. DID feature extraction — clean train + clean test
2. DID feature extraction — `randaug1` train (robustness augmentation)
3. `did/train.py` — two-head classifier, threshold calibrated on a train slice
4. DID feature extraction — 14-transform test suite (`--limit 150`/class)
5. `did/evaluate.py` + `did/make_report.py`

Knobs (env vars, override with `--export=ALL,KEY=VAL`):

| var | default | meaning |
|---|---|---|
| `DS` | `wildfake` | dataset dir: `data/$DS/{train,test}/{real,fake}` (also `sid_set`) |
| `TAG` | `${DS}_${RECON}_${BACKBONE}` | names `checkpoints/did_$TAG.pt`, `results/did/$TAG/` |
| `RECON` | `sd15` | reconstructor: `sd15` or `sana16` |
| `BACKBONE` | `resnet18` | classifier head: `resnet18` or `resnet50` |
| `RES` | `256` | reconstruction resolution (`512` for sana16) |
| `STEPS` | `10` | DDIM / flow steps (`7` for sana16) |
| `EPOCHS` | `14` | training epochs |
| `LIMIT` | `150` | per-class cap for the transformed test suite |
| `SKIP_EXTRACT` | – | `1` = reuse the feature cache, go straight to train/eval |
| `EVAL_CKPT` | – | path = skip training, evaluate this ckpt (zero-shot cross-dataset) |

Feature caches are keyed `cache/feat_${DS}_${RECON}_r${RES}s${STEPS}` — shared
across backbones. Extraction is resumable (skips existing `.npz`) — re-`sbatch`
the same job if it times out.

### Experiment matrix run here

```bash
# WildFake: 2x2 reconstructor x classifier
sbatch --export=ALL,RECON=sd15,BACKBONE=resnet18   slurm/pipeline.sh
sbatch --export=ALL,RECON=sd15,BACKBONE=resnet50,SKIP_EXTRACT=1   slurm/pipeline.sh
sbatch --export=ALL,RECON=sana16,BACKBONE=resnet18,RES=512,STEPS=7 slurm/pipeline.sh
sbatch --export=ALL,RECON=sana16,BACKBONE=resnet50,RES=512,STEPS=7,SKIP_EXTRACT=1 slurm/pipeline.sh

# SID_Set: native model + zero-shot both directions
.venv/bin/python slurm/dl_sid.py 10        # login node: download 10 parquet shards
sbatch slurm/sid_fetch.sh                  # CPU job: parquet -> data/sid_set (login node OOMs on parse)
sbatch --export=ALL,DS=sid_set,RECON=sd15,BACKBONE=resnet18 slurm/pipeline.sh
sbatch --export=ALL,DS=sid_set,SKIP_EXTRACT=1,EVAL_CKPT=checkpoints/did.pt,TAG=sid_set_zeroshot_from_wildfake slurm/pipeline.sh
sbatch --export=ALL,DS=wildfake,SKIP_EXTRACT=1,EVAL_CKPT=checkpoints/did_sid_set_sd15_resnet18.pt,TAG=wildfake_zeroshot_from_sid slurm/pipeline.sh
```

## One-time setup (if `data/` or `~/.cache` is wiped)

```bash
export TMPDIR=$HOME/tmp HF_HOME=$HOME/.cache/huggingface HF_HUB_DISABLE_XET=1

# 1. WildFake subset (HTTP range reads, ~10 min, login node is fine)
.venv/bin/python did/fetch_wildfake.py --out data/wildfake --train 300 --test 150

# 2. model weights (login node, offline-cache for the jobs)
.venv/bin/python slurm/dl_sd15.py     # SD-1.5 vae/unet/text_encoder/tokenizer
.venv/bin/python slurm/dl_sana.py     # SANA-1.6B (only if running RECON=sana16)
.venv/bin/python slurm/dl_sid.py 10   # SID_Set parquet shards (only if DS=sid_set)
curl -sL --retry 10 -o ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
    https://download.pytorch.org/models/resnet18-f37072fd.pth
curl -sL --retry 10 -o ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth \
    https://download.pytorch.org/models/resnet50-0676ba61.pth
```

(The `dl_*.py` helper scripts are 3-line
`snapshot_download` calls with `allow_patterns` for the needed subfolders.)
