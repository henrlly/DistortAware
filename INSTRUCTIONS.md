# Fetch, train, and test

Run commands from the repository root. The full fetch uses a CPU SLURM job;
model training and DID feature extraction use one GPU. The default full fetch
quota is 3,000 images per class. The quick subset remains 200 per class.
Set `REPO` to the absolute checkout path in every `sbatch` command; the
examples below assume you are currently in the repository root.
The training scripts request an H100 MIG resource (`h100-47`). Change the
SBATCH resource line if your cluster exposes a different H100 resource name.

## 1. Fetch the shared dataset

```bash
REPO="$(pwd)"
sbatch --export=ALL,REPO="$REPO" \
  slurm/fetch_harness_data.sh
```

This creates `data/harness_large/` and `data/harness_quick/`. To override the
default quota:

```bash
BASE_PER_CLASS=3000 SID_PER_CLASS=3000 WILDFake_PER_SOURCE=3000 \
  sbatch --export=ALL,REPO="$(pwd)",BASE_PER_CLASS=3000,SID_PER_CLASS=3000,WILDFake_PER_SOURCE=3000 \
  slurm/fetch_harness_data.sh
```

The fetcher is cached. Set `REFRESH=1` when the materialized data must be
rebuilt.

Verify the fetch before submitting training jobs:

```bash
python -m harness.verify_fetch \
  --data-dir data/harness_large \
  --quick-data-dir data/harness_quick \
  --quick-per-class 200
```

This prints `FETCH OK` only after checking metadata, required manifests,
class labels, and every referenced image and mask path.

## 2. Train DID

DID has a pretrained diffusion reconstructor, but its two ResNet classifier
heads must be trained. The job converts the shared three-label manifests to a
binary DID tree, extracts clean and `randaug1` features, and trains the
classifier.

```bash
sbatch --export=ALL,REPO="$(pwd)" \
  slurm/train_did.sh
```

The checkpoint is written to `checkpoints/did/pooled_sd15_resnet18.pt` by
default. Override `TAG`, `RECON`, `RES`, `STEPS`, `EPOCHS`, `BS`, `DATA`, or
`CKPT`, or `BACKBONE` with `--export`.

## 3. Train PatchHead

PatchHead has a pretrained DINOv3 backbone, but its LoRA adapters and detector
heads must be trained. The full training job trains both independent variants:

```bash
sbatch --export=ALL,REPO="$(pwd)" \
  slurm/train_patchhead.sh
```

Outputs are written under `checkpoints/patchhead/pooled/`:

```text
baseline/checkpoint.pt
distortion_aware/checkpoint.pt
```

For a quick 200-per-class smoke run, use `slurm/train_patchhead_quick_compare.sh`
with `DATA=data/harness_quick`.

## 4. Test and generate reports

### DID

```bash
sbatch --export=ALL,REPO="$(pwd)",TAG=pooled_sd15_resnet18 \
  slurm/evaluate_did.sh
```

This writes DID `robustness.csv`, `metrics.json`, and `report.md` under
`results/did/<tag>/`.

### Physics and PatchHead harness

```bash
sbatch --export=ALL,REPO="$(pwd)", \
  DATA="$(pwd)/data/harness_large", \
  BASELINE="$(pwd)/checkpoints/patchhead/pooled/baseline/checkpoint.pt", \
  AWARE="$(pwd)/checkpoints/patchhead/pooled/distortion_aware/checkpoint.pt", \
  OUT="$(pwd)/results/harness/pooled_evaluation" \
  slurm/evaluate_harness.sh
```

The harness writes a combined report under
`results/harness/quick_evaluation/` by default:

```text
records.csv
records.jsonl
metrics.json
report.md
models/<model>/records.csv
models/<model>/report.md
```

The CSV contains one row per image/model/transform. Markdown reports include
coverage, classifier metrics, source/label breakdowns, and Physics-specific
applicability and cue-confidence summaries. Physics is evidence only and does
not produce an AIGC classifier decision.

To run PatchHead and Physics as two parallel SLURM jobs on the same manifest:

```bash
REPO="$(pwd)"
sbatch --export=ALL,REPO="$REPO" slurm/evaluate_patchhead.sh
sbatch --export=ALL,REPO="$REPO" slurm/evaluate_physics.sh
```

To test every method independently in parallel, first ensure both PatchHead
checkpoints and the DID checkpoint exist:

```bash
REPO="$(pwd)"
bash slurm/submit_all_model_tests.sh
```

This submits five jobs: PatchHead baseline, distortion-aware PatchHead,
Physics, DID, and the independent filter baseline. Every job uses the same
`wildfake_benchmark.csv` records and harness transforms. No original WildFake
ZIP archives are needed. To run only one PatchHead variant:

```bash
sbatch --export=ALL,REPO="$REPO",DATA="$REPO/data/harness_large",MANIFEST=wildfake_benchmark.csv,MODELS=patchhead_baseline,OUT="$REPO/results/parallel_evaluation/patchhead" slurm/evaluate_patchhead.sh
sbatch --export=ALL,REPO="$REPO",DATA="$REPO/data/harness_large",MANIFEST=wildfake_benchmark.csv,MODELS=patchhead_distortion_aware,OUT="$REPO/results/parallel_evaluation/distortion_aware" slurm/evaluate_patchhead.sh
```

If baseline training completed but distortion-aware training failed, continue
the distortion-aware run from the baseline weights:

```bash
sbatch --export=ALL,REPO="$REPO",MODE=distortion_aware,INIT="$REPO/checkpoints/patchhead/pooled/baseline/checkpoint.pt" slurm/train_patchhead.sh
```

Use `MANIFEST=wildfake_benchmark.csv` and separate `OUT` directories to run
the same two jobs on the COCO/DALL·E benchmark.

DID can run independently on the same manifest with:

```bash
sbatch --export=ALL,REPO="$REPO",MANIFEST=wildfake_benchmark.csv, \
  OUT="$REPO/results/parallel_evaluation/did_benchmark" \
  slurm/evaluate_did_manifest.sh
```

## 5. Python entry points

Each method exposes a `run(input_path, ...) -> dict` function and a batch CLI
that loads its model once. Both return the shared
keys `method`, `image_path`, `score`, `score_kind`, `confidence`, `threshold`,
`decision`, and `details`. See `docs/METHOD_ENTRYPOINTS.md` for examples.

The shared SLURM environment is installed once from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Physics and the filter baseline remain independently installable outside the
harness:

```bash
pip install -e ./physics
pip install -e ./filter_based_approach
```

PatchHead additionally needs DINOv3 weights and its checkpoint. DID needs the
Stable Diffusion reconstruction weights and its checkpoint. Filter needs
`filter_based_approach/models/mask_classifier.pt`; Physics needs no checkpoint
for its default deterministic cues.

## 6. Validation and Git

```bash
python3 -m unittest discover -s harness/tests -p 'test_*.py' -v
python3 -m py_compile harness/*.py harness/tests/*.py did/*.py \
  patchhead/entrypoint.py filter_based_approach/entrypoint.py
bash -n slurm/*.sh
git diff --check
```

Stage project changes while leaving local IDE settings out of the commit:

```bash
git add -A -- ':!.idea'
```

### Commit evaluation reports without generated artifacts

Do not use `git add -A` after an evaluation. The `views/`, `native/`, and
`records.jsonl` outputs are generated working files and can be large. Stage
only the CSV, JSON, and Markdown reports that you intend to keep.

For example, after the two PatchHead WildFake evaluations complete:

```bash
RESULTS=(
  results/parallel_evaluation/patchhead_baseline
  results/parallel_evaluation/distortion_aware
)

# Stop before committing if a report contains an absolute home path or name.
rg -n '/home/|/Users/' "${RESULTS[@]}" && exit 1

git add \
  results/parallel_evaluation/patchhead_baseline/{metrics.json,report.md,records.csv} \
  results/parallel_evaluation/patchhead_baseline/models/*/{records.csv,report.md} \
  results/parallel_evaluation/distortion_aware/{metrics.json,report.md,records.csv} \
  results/parallel_evaluation/distortion_aware/models/*/{records.csv,report.md}

git status --short
git commit -m "Add PatchHead COCO and DALL-E benchmark results"
git push origin HEAD
```

Confirm that no generated files were staged before committing. If the privacy
scan finds a match, do not publish the report until its paths are sanitized.
