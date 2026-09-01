# AIGC Detection Setup and Results Report

This report records the reproducible setup, datasets, training procedures, and
evaluation commands for each method. Model results are intentionally left as
placeholders until the current H100 runs finish.

## 1. Environment and common setup

Run from the repository root on the cluster:

```bash
cd /path/to/tiktok-aigc-detect
source .venv/bin/activate
```

The fetch job is CPU-only and network-enabled. Training and learned-model
evaluation use SLURM GPUs. The current training jobs request the H100 MIG
resource `h100-47`.

Stable Diffusion weights required by DID must be downloaded on a node with
network access before DID feature extraction:

```bash
export TMPDIR="$HOME/tmp"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$TMPDIR" "$HF_HOME"
.venv/bin/python slurm/dl_sd15.py
```

Physics core dependencies are installed into the same environment:

```bash
python -m pip install -e ./physics
```

## 2. Datasets

The shared harness fetch creates two materialized datasets:

- `data/harness_large/`: full pooled dataset from SID_Set, HEMG, CIFAKE, and
  WildFake sources, using the configured 3,000-per-source quotas.
- `data/harness_quick/`: deterministic 200-per-class subset for quick checks.

Fetch and verify them:

```bash
REPO="$(pwd)"
sbatch --export=ALL,REPO="$REPO" slurm/fetch_harness_data.sh
python -m harness.verify_fetch \
  --data-dir data/harness_large \
  --quick-data-dir data/harness_quick \
  --quick-per-class 200
```

The verification command must print `FETCH OK`. It checks metadata, manifests,
labels, and referenced image and mask files.

### Recorded dataset inventory

These counts were observed for the current `harness_large` fetch. Labels are
`0 = real`, `1 = fully synthetic`, and `2 = tampered`.

| Split | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|
| Train | 10,706 | 13,001 | 1,493 |
| Validation | 2,742 | 3,113 | 445 |
| Calibration | 2,705 | 3,132 | 463 |
| Matched test | 1,847 | 1,754 | 599 |

The fetch quota is not a guarantee of equal final split sizes. The quick
dataset is expected to contain exactly 200 examples per class in each required
split.

### Planned cross-dataset benchmark: WildFake COCO/DALL·E

The fetch also materializes `data/harness_large/wildfake_benchmark.csv`, which
contains the held-out WildFake benchmark:

- 500 COCO real images;
- 500 DALL·E synthetic images.

This benchmark is separate from `matched_test.csv` and is intended to measure
cross-dataset generalisation. It must not be used for training or checkpoint
selection.

The planned benchmark evaluation will run every method on the same manifest:

| Method | Training on COCO/DALL·E? | Planned evaluation |
|---|---|---|
| DID | No | Binary real/fake prediction |
| PatchHead baseline | No | Three-class output plus binary AI decision |
| PatchHead distortion-aware | No | Three-class output plus binary AI decision |
| Physics | No | Evidence, applicability, and confidence only |
| Filter-based baseline | No | Binary real/fake prediction and mask metrics where available |

The benchmark workflow runs DID, both PatchHead checkpoints, Physics, and the
filter baseline through their independent entry points on the same materialized
COCO/DALL·E manifest and combines normalized results under:

```text
results/benchmark/wildfake_coco_dalle/
```

Independent manifest-based evaluation jobs now exist for DID, both PatchHead
variants, Physics, and the filter baseline. Every report stores the shared
manifest fingerprint so separately scheduled results can be compared safely.

## 3. DID

### Setup and data preparation

DID uses the shared three-class manifests but converts them to a binary tree:
class 0 becomes `real`; classes 1 and 2 become `fake`. The diffusion
reconstructor is pretrained; the two ResNet classifier heads are trained.

### Training

```bash
REPO="$(pwd)"
sbatch --export=ALL,REPO="$REPO" slurm/train_did.sh
```

The job prepares `runs/did_data/pooled_sd15_resnet18/`, extracts clean and
`randaug1` features, and trains for 14 epochs by default. The cache is
resumable.

Default checkpoint:

```text
checkpoints/did/pooled_sd15_resnet18.pt
```

### Test

```bash
sbatch --export=ALL,REPO="$(pwd)",TAG=pooled_sd15_resnet18 \
  slurm/evaluate_did.sh
```

Outputs:

```text
results/did/pooled_sd15_resnet18/metrics.json
results/did/pooled_sd15_resnet18/report.md
results/did/pooled_sd15_resnet18/robustness.csv
```

### DID results

| Dataset/split | Samples | Accuracy @ 0.5 | Accuracy @ calibrated threshold | Notes |
|---|---:|---:|---:|---|
| Harness large matched test | — | — | — | Pending H100 run |
| Harness quick test | — | — | — | Optional quick run |

## 4. PatchHead

### Setup and data preparation

PatchHead trains on the shared manifests directly. The DINOv3 backbone is
pretrained; LoRA adapters and detector heads are trained. Baseline and
distortion-aware variants are independent checkpoints.

### Training

```bash
REPO="$(pwd)"
sbatch --export=ALL,REPO="$REPO" slurm/train_patchhead.sh
```

Default outputs:

```text
checkpoints/patchhead/pooled/baseline/checkpoint.pt
checkpoints/patchhead/pooled/distortion_aware/checkpoint.pt
```

For a quick 200-per-class run:

```bash
sbatch --export=ALL,REPO="$(pwd)",DATA="$(pwd)/data/harness_quick" \
  slurm/train_patchhead_quick_compare.sh
```

### Test

The harness evaluates Physics and both PatchHead variants together:

```bash
sbatch --export=ALL,REPO="$(pwd)", \
  DATA="$(pwd)/data/harness_large", \
  BASELINE="$(pwd)/checkpoints/patchhead/pooled/baseline/checkpoint.pt", \
  AWARE="$(pwd)/checkpoints/patchhead/pooled/distortion_aware/checkpoint.pt", \
  OUT="$(pwd)/results/harness/pooled_evaluation" \
  slurm/evaluate_harness.sh
```

Outputs include combined and per-model CSV/Markdown reports under
`results/harness/pooled_evaluation/`.

### PatchHead results

| Dataset/split | Model | Samples | Accuracy | Balanced accuracy | ROC-AUC | Notes |
|---|---|---:|---:|---:|---:|---|
| Harness large matched test | Baseline | — | — | — | — | Pending H100 run |
| Harness large matched test | Distortion-aware | — | — | — | — | Pending H100 run |
| Harness quick test | Baseline | — | — | — | — | Optional quick run |
| Harness quick test | Distortion-aware | — | — | — | — | Optional quick run |

## 5. Physics

### Setup

Physics has no training phase and no learned checkpoint for the normal
harness path. It is an evidence sidecar that reports applicability, geometric
consistency cues, violation score, and confidence. It does not produce the
primary AIGC classifier decision.

Install its core package as described in Section 1. Optional learned proposal
backends require:

```bash
python -m pip install -e './physics[auto,eval]'
```

### Test

Physics is tested through the same harness evaluation command in the PatchHead
section. Its report fields include applicability, cue summaries, and
confidence statistics.

### Physics results

| Dataset/split | Samples | Applicable rate | Mean confidence | Mean violation score | Notes |
|---|---:|---:|---:|---:|---|
| Harness large matched test | — | — | — | — | Pending H100 harness run |
| Harness quick test | — | — | — | — | Optional quick run |

## 6. Filter-based baseline

### Setup and training

The residual/filter baseline is a separate three-class model with a mask head.
It trains directly from the SID_Set stream and does not use the pooled harness
training command:

```bash
cd filter_based_approach
uv sync
uv run python scripts/train.py \
  --train-per-class 300 \
  --validation-per-class 100
```

The default checkpoint is `filter_based_approach/models/mask_classifier.pt`.

### Test

Place the required WildFake archives at the paths documented by the baseline,
then run:

```bash
uv run python scripts/evaluate.py
```

This evaluates the SID validation stream and held-out WildFake archives.

### Filter baseline results

| Dataset/split | Samples | Binary accuracy | Balanced accuracy | Notes |
|---|---:|---:|---:|---|
| SID validation | — | — | — | Pending rerun/current checkpoint result |
| WildFake held-out | — | — | — | Pending rerun/current checkpoint result |

## 7. Final validation checklist

Before recording results, save the following for each run:

1. SLURM job ID and `sacct` status.
2. Dataset manifest counts and fetch fingerprint.
3. Training checkpoint path and training log.
4. Evaluation output directory.
5. `metrics.json`, `records.csv`, and Markdown report.
6. Hardware and key settings: GPU model, batch size, epochs, resolution,
   reconstructor, and transform list.

Use this command to verify completed jobs:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```
