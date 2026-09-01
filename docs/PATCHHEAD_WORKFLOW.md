# PatchHead workflow

The PatchHead path is self-contained. It does not clone or import the DINOv3
repository. The legacy DID implementation under `did/` remains available for
historical comparisons only.

## Data contract

Run fetching on a network-enabled login node. The default output is
`data/matched_refactored/` and is safe to rerun:

```bash
source .venv/bin/activate
bash slurm/fetch_matched_data.sh
```

Before submitting a GPU job, run the login-node preflight:

```bash
bash slurm/preflight.sh
```

The fetcher owns the SID_Set, HEMG, CIFAKE, and WildFake providers. It caches
provider downloads and writes manifests for `train`, `validation`, and
`calibration`. Additional WildFake sources (ImageNet, CelebA-HQ, AFHQ, ADM,
DDIM, DDPM, and VQDM) are training/validation data only.

The official benchmark is one combined manifest:

```text
wildfake_benchmark.csv = 500 COCO real + 500 DALL-E Advanced fake
```

`sid_eval_200.csv` is a deterministic seed-42 diagnostic subset. The new
three-class model labels real as 0, fully synthetic as 1, and tampered as 2;
synthetic and tampered are collapsed into one AI-positive score for binary
metrics.

Use `--refresh` only when deliberately regenerating a dataset:

```bash
python patchhead/cli.py fetch --output-dir data/matched_refactored --refresh
```

## Train locally or on SLURM

```bash
python patchhead/cli.py train \
  --manifest-dir data/matched_refactored \
  --epochs 10 --bs 16 \
  --out patchhead/checkpoints/patchhead_refactored.pt
```

Training uses `train.csv`, selects the best checkpoint on `validation.csv`,
and calibrates thresholds on `calibration.csv`. The held-out benchmark is not
used during training.

Training augmentation is enabled by default and uses seeded mild image-space
perturbations: flips, brightness/contrast/color changes, blur, Gaussian noise,
JPEG recompression, downscale-resize, and random crop-resize. Disable it with
`--no-train-aug` when running `patchhead/train.py`. For tampered SID images,
mask-guided crops are also enabled by default with 20% padding and 50% crop
probability; override with `--mask-crop-prob` and `--mask-padding`.

For the full GPU workflow:

```bash
sbatch -G h100-47 --export=ALL slurm/matched_patchhead.sh
```

The workflow trains once, evaluates the combined WildFake benchmark, then
evaluates SID without TTA and with top-3 and top-5 overlapping-crop TTA.

## Evaluation only

After a checkpoint exists:

```bash
sbatch -G h100-47 --export=ALL slurm/eval_matched_patchhead.sh
```

This never trains. It writes:

```text
results/patchhead/results_refactored_wildfake_benchmark/
results/patchhead/results_refactored_sid_normal/
results/patchhead/results_refactored_sid_top3/
results/patchhead/results_refactored_sid_top5/
```

Each report includes threshold `0.5`, the calibration-balanced threshold, and
the calibration threshold targeting 5% real-image FPR when available. AUC is
the primary threshold-independent comparison metric.

## Direct CLI

```bash
python patchhead/cli.py fetch --help
python patchhead/cli.py train --help
python patchhead/cli.py eval --help
```

The CLI delegates training/evaluation to the existing Python entrypoints so
old direct commands remain usable while the new workflow has one documented
interface.
