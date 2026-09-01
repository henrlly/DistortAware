# Distortion-Aware AIGC Detection

## Remote GPU runbook for the repo author

This is the recommended end-to-end path on the school server. Commands in the
setup, download, and preflight sections run on the login node. Training and
evaluation run through SLURM on an H100 compute node.

### 1. Update the repository and environment

```bash
cd ~/tiktok-aigc-detect
git fetch origin
git switch matched-patchhead-experiment
git pull --ff-only origin matched-patchhead-experiment
source .venv/bin/activate
python -m pip check
```

If this branch is not yet local:

```bash
git switch --track origin/matched-patchhead-experiment
```

Confirm that the new files are present before submitting anything:

```bash
test -f distortion_aware_approach.md
test -f patchhead/predict.py
test -f tests/test_distortion_aware.py
grep -n 'TAG=.*distortion_aware' slurm/matched_patchhead.sh
```

### 2. Verify download credentials

```bash
hf auth whoami
kaggle datasets list >/dev/null
```

If either command fails, authenticate on the login node:

```bash
hf auth login
kaggle auth login
```

### 3. Cache the DINOv3 backbone

The compute jobs use offline mode, so download the backbone on the
network-enabled login node first:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import hf_hub_download; [print(hf_hub_download('timm/vit_large_patch16_dinov3.lvd1689m', f)) for f in ('config.json', 'model.safetensors')]"
```

This step can be skipped when both files are already in the Hugging Face cache.

### 4. Fetch or verify the matched data

To create `data/matched_refactored/` using the repository fetcher:

```bash
cd ~/tiktok-aigc-detect
bash slurm/fetch_matched_data.sh
```

Normal reruns reuse the downloaded data. The directory must contain:

```text
data/matched_refactored/train.csv
data/matched_refactored/validation.csv
data/matched_refactored/calibration.csv
data/matched_refactored/wildfake_benchmark.csv
data/matched_refactored/sid_eval_200.csv
```

The 1,000-image WildFake benchmark manifest contains 500 COCO real and 500
DALL-E Advanced fake images. It is held out from training. SID is evaluated
separately using `sid_eval_200.csv`.

### 5. Run login-node preflight

```bash
bash slurm/preflight.sh
```

Expected final line:

```text
preflight: OK (GPU availability is checked inside the compute job)
```

Preflight deliberately does not import PyTorch because the login node has a
tight virtual-memory limit. CUDA and the PyTorch-based unit tests are checked
inside the compute job.

### 6. Submit training and evaluation

```bash
sbatch -G h100-47 --export=ALL slurm/matched_patchhead.sh
```

Record the job ID printed by `sbatch`. The job performs, in order:

1. CUDA validation and the distortion-aware unit tests.
2. Distortion-aware PatchHead training for 10 epochs.
3. Checkpoint metadata validation.
4. The combined WildFake benchmark.
5. SID normal, top-3 crop-TTA, and top-5 crop-TTA evaluation.

The default outputs are isolated from the older `refactored` experiment:

```text
patchhead/checkpoints/patchhead_distortion_aware.pt
results/patchhead/results_distortion_aware_wildfake_benchmark/
results/patchhead/results_distortion_aware_sid_normal/
results/patchhead/results_distortion_aware_sid_top3/
results/patchhead/results_distortion_aware_sid_top5/
```

### 7. Monitor the job

Replace `123456` with the ID returned by `sbatch`:

```bash
squeue -j 123456
tail -f job_matched_123456.out
tail -f job_matched_123456.err
sacct -j 123456 --format=JobID,State,Elapsed,MaxRSS
```

Early successful output should include lines resembling:

```text
cuda=NVIDIA H100 ...
Ran 9 tests ... OK
device=cuda
```

After training it should also print:

```text
verified distortion-aware checkpoint .../patchhead_distortion_aware.pt
```

The job is complete only when SLURM reports `COMPLETED` and the output ends with
`=== DONE ===`. A Python traceback, `OUT_OF_MEMORY`, `TIMEOUT`, or a missing
checkpoint-validation line means the run did not finish successfully.

### 8. Rerun evaluation without retraining

Once the checkpoint exists:

```bash
sbatch -G h100-47 --export=ALL slurm/eval_matched_patchhead.sh
```

This job refuses to evaluate an old checkpoint that does not contain
`distortion_aware=True`. To evaluate a separately named run, use the same tag
that was supplied during training:

```bash
sbatch -G h100-47 --export=ALL,TAG=distortion_v2 slurm/eval_matched_patchhead.sh
```

### 9. Inspect the results

Start with the normal-inference CSVs:

```bash
column -s, -t results/patchhead/results_distortion_aware_wildfake_benchmark/robustness.csv | less -S
column -s, -t results/patchhead/results_distortion_aware_sid_normal/robustness.csv | less -S
```

For each transformation, `acc` is the distortion-adjusted result and
`base_acc` is the same checkpoint and image before threshold adjustment. This
is the primary controlled comparison. Pay particular attention to `jpeg30`,
`noise0.05`, and `noise0.10`, then verify that `clean` and WildFake did not
regress.

The detailed distortion-estimation metrics and aggregate robustness results are
in each output directory's `metrics.json`:

```bash
python -m json.tool results/patchhead/results_distortion_aware_sid_normal/metrics.json | less
```

Do not use `oracle` results as the reported deployment result. The normal SLURM
workflow explicitly uses `--distortion-mode predicted`, which estimates the
distortion from the input image alone.

## Purpose

The original PatchHead detector has strong ranking performance under severe
post-processing, but its fixed decision threshold becomes poorly calibrated.
For example, JPEG compression or noise can shift real and generated-image
scores in one direction even when AUC remains high. That means the detector can
still rank the two classes correctly while making avoidable mistakes at the
final real/fake boundary.

This extension makes the detector aware of the post-processing visible in each
image. It estimates the distortion type and severity, then adjusts the binary
AIGC decision threshold for that image. The objective is not to treat
distortion as evidence that an image is fake. It is to interpret the existing
AIGC evidence in the correct distortion context.

## What was implemented and why

### 1. Distortion-labelled training augmentation

Training transformations now return both the transformed image and exact
metadata describing the applied operations. The supported operations are:

- JPEG compression
- Gaussian blur
- resizing
- Gaussian noise
- brightness, contrast, and saturation adjustment
- cropping

The augmentation distribution includes clean images and compositions of one or
two transformations. It deliberately covers stronger severities than the old
training pipeline, including JPEG quality down to 25 and noise sigma up to
0.11. This matters because a model trained only on mild corruption has to
extrapolate when evaluated on `jpeg30`, `noise0.05`, or `noise0.10`.

The old Gaussian-noise augmentation was also corrected. It added one random
value to the entire image; the new version samples independent noise for every
pixel and channel.

### 2. Blind distortion estimation

At inference time the true transformation is unknown, so the model predicts a
multi-label distortion vector from the image. The estimator combines:

- DINOv3 semantic and visual features;
- high-pass residual statistics;
- gradient and Laplacian statistics;
- approximate block-boundary statistics useful for compression artifacts.

This hybrid design was chosen because the two feature families complement one
another. Learned DINO features can recognize complex and rescaled artifacts,
while analytical measurements directly expose local noise, sharpness, and
compression structure. The output contains probabilities for each distortion
type and normalized severity estimates.

Distortion estimation is an auxiliary task, not a replacement for the AIGC
detector. Explicit supervision keeps the distortion representation meaningful
and makes its errors measurable with type precision/recall/F1 and severity MAE.

### 3. Distortion-conditioned decision threshold

The original three-class PatchHead prediction remains intact. Its synthetic and
tampered probabilities are combined into a binary AIGC score as before.

A small threshold adapter consumes the estimated distortion vector and predicts
a bounded shift in logit space. A positive shift makes the base detector require
more evidence before declaring AIGC; a negative shift requires less. The
adapter starts at zero, so training begins with exactly the original decision
rule rather than a random threshold correction.

This approach targets the observed problem directly: high AUC but distortion-
dependent false-positive or false-negative rates. It can recover accuracy by
moving the operating point without discarding the base model's useful ranking.

Normal inference uses only predicted distortion. Known augmentation metadata is
used during training as scheduled guidance and is available during evaluation
only as an explicitly labelled `oracle` upper-bound experiment.

## Manual training commands

The SLURM runbook above is preferred on the school server. For an interactive
GPU environment, run commands from the repository root:

```bash
cd ~/tiktok-aigc-detect
```

The DINOv3 backbone must already be available in the Hugging Face cache. If it
is not cached, download it once:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python -c \
 "from huggingface_hub import hf_hub_download as d; \
  [d('timm/vit_large_patch16_dinov3.lvd1689m', f) \
   for f in ('config.json', 'model.safetensors')]"
```

Train on pooled WildFake and SID data:

```bash
python patchhead/train.py \
  --ds pooled \
  --epochs 10 \
  --bs 16 \
  --out patchhead/checkpoints/patchhead_distortion_aware.pt
```

If normalized manifests are being used, point training at the directory that
contains `train.csv`, `validation.csv`, `calibration.csv`, and either
`matched_test.csv` or `test.csv`:

```bash
python patchhead/train.py \
  --ds pooled \
  --manifest-dir data/matched_refactored \
  --epochs 10 \
  --bs 16 \
  --out patchhead/checkpoints/patchhead_distortion_aware.pt
```

Distortion-aware training is enabled by default. Use
`--no-distortion-aware` only when deliberately training a baseline checkpoint.

## Evaluation

Evaluate the deployable version, which estimates distortion from each image:

```bash
python patchhead/evaluate.py \
  --ds sid_set \
  --ckpt patchhead/checkpoints/patchhead_distortion_aware.pt \
  --distortion-mode predicted \
  --out results/patchhead/results_distortion_predicted
```

Repeat with `--ds wildfake` to check that improving SID robustness has not
regressed WildFake performance.

Evaluate the same checkpoint without threshold adjustment. This is the fairest
way to isolate the value of distortion conditioning because both runs use the
same trained authenticity model:

```bash
python patchhead/evaluate.py \
  --ds sid_set \
  --ckpt patchhead/checkpoints/patchhead_distortion_aware.pt \
  --distortion-mode off \
  --out results/patchhead/results_distortion_off
```

Optionally measure the oracle upper bound using known synthetic-transform
metadata:

```bash
python patchhead/evaluate.py \
  --ds sid_set \
  --ckpt patchhead/checkpoints/patchhead_distortion_aware.pt \
  --distortion-mode oracle \
  --out results/patchhead/results_distortion_oracle
```

Oracle results are diagnostic only and should not be presented as normal model
performance. A large gap between oracle and predicted modes means distortion
estimation—not threshold conditioning—is the main remaining bottleneck.

Each evaluation writes:

- `robustness.csv`: clean and per-distortion accuracy, AUC, base accuracy, and
  mean dynamic threshold;
- `metrics.json`: aggregate robustness and distortion-estimation metrics;
- `preds_clean.json`: adjusted and base scores for each clean image;
- `error_analysis.json`: the highest-confidence false positives and false
  negatives.

The most important comparison is `predicted` versus `off`, especially on
`jpeg30`, `noise0.05`, and `noise0.10`. A useful result should improve the weak
slices without materially reducing clean accuracy or WildFake performance.

## Running on new images

Run distortion-aware inference on one or more local images:

```bash
python patchhead/predict.py example.jpg another-image.png \
  --ckpt patchhead/checkpoints/patchhead_distortion_aware.pt
```

Save the JSON output to a file when needed:

```bash
python patchhead/predict.py example.jpg \
  --ckpt patchhead/checkpoints/patchhead_distortion_aware.pt \
  --out prediction.json
```

The output includes the adjusted AIGC score, the original base score, the
equivalent per-image base threshold, estimated distortion probabilities, and
approximate severities. These values make it possible to explain whether a
decision changed because the image appeared strongly compressed, noisy,
blurred, resized, colour-adjusted, or cropped.

## Interpreting the experiment

The threshold adapter should be judged by operating-point metrics, not AUC
alone. Since its primary job is score calibration, it may substantially improve
accuracy and class recall while leaving AUC nearly unchanged. Report clean
accuracy, AUC, real recall, fake recall, mean transformed accuracy, and the
worst transformation for both `predicted` and `off` modes.

Distortion detection is inherently ambiguous. A naturally noisy photograph may
look synthetically noised, and JPEG artifacts may be weakened by later
rescaling. The distortion estimates should therefore be treated as soft
context, which is why the implementation uses probabilities, bounded threshold
shifts, mixed clean/corrupted training, and a zero-effect initialization.
