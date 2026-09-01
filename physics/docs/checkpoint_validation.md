# Checkpoint-backed validation

Validation date: 2026-08-31 (Asia/Singapore)

## Scope and invariants

The supplied pooled PatchHead checkpoint is now exercised by the unified
inference path. Its released one-logit output layers are reconstructed under the
newer three-class training tree so the stored tensors still load; no checkpoint
weight, preprocessing step, sigmoid score formula, threshold, or verdict is
altered. Physics consumes existing patch logits and, only when requested, the
final DINO grid from the same forward pass. Physics remains a score-independent
explanation sidecar. A future distortion-aware three-class pooled checkpoint
still requires an explicitly versioned inference contract.

The external pooled PatchHead checkpoint, frozen backbones, physics proposal
weights, datasets, caches, overlays, and generated result files remain outside
Git. The separate compact residual-artifact checkpoint intentionally bundled at
`filter_based_approach/models/mask_classifier.pt` is not a physics or primary
PatchHead weight and cannot affect the primary verdict.

## Holding-folder audit

The earlier `tiktok techjam 26` holding directory was compared against this
repository before retirement. Its physics source is version 0.3.0; the official
version 0.6.1 tree is a functional superset with automatic proposals,
checkpoint-backed DINO reuse, newer safety gates, evaluation tools, and broader
tests. Its one full SID shard was preserved under the canonical ignored cache
with the documented hash; the remaining historical environment/reports were
archived outside the project. No engine functionality needed to be copied from
the holding directory.

## Reproducibility

| Artifact | Recorded value |
|---|---|
| Pooled PatchHead SHA-256 | `828fac3ba5c5b814a1ada36477b36848ab4c8366e040e68fff9f4c9fe14b6989` |
| Checkpoint dataset tag | `pooled` |
| Architecture | `patchhead-dinov3-vitl16` |
| Threshold | `0.785` |
| Input size | `256` |
| Stored validation accuracy | `1.0` |
| Frozen backbone | `timm/vit_large_patch16_dinov3.lvd1689m` |
| Backbone snapshot | `30c1109559f65dea34316b0d4842d35c5771fe11` |

The full external directory was checksummed and its metadata/tensor payloads
were read successfully with safe PyTorch loading:

| File | Configuration/tag | Threshold | SHA-256 |
|---|---|---:|---|
| `patchhead_pooled.pt` | pooled DINOv3-L PatchHead | 0.785 | `828fac3ba5c5b814a1ada36477b36848ab4c8366e040e68fff9f4c9fe14b6989` |
| `patchhead_sid_set.pt` | SID_Set DINOv3-L PatchHead | 0.500 | `823ffaf5cde53d54da6cc819cb59f37cc5d0684e18a3edbd9216bc101a547115` |
| `patchhead_wildfake.pt` | WildFake DINOv3-L PatchHead | 0.765 | `1f77fbc2c166e446cfb6d2c1eba1ed540e5d68ed859c12142266f07b86226bfb` |
| `did_pooled_sd15_resnet18.pt` | pooled SD-1.5 / ResNet-18 | 0.530 | `0820e77ebc39b3cae84eab41c1b2d4c0141f2af5d02569b566e86b1c509d3c26` |
| `did.pt` / `did_sd15_resnet18.pt` | WildFake SD-1.5 / ResNet-18 (identical bytes) | 0.670 | `201646b8d483cadc708f830304459227c91866a8517132032a2acb84722bc4dc` |
| `did_sd15_resnet50.pt` | WildFake SD-1.5 / ResNet-50 | 0.460 | `548f58747b33ab608bee200caf06ee683b5ec043b2a1439ab4d5db996df44cca` |
| `did_sana16_resnet18.pt` | WildFake SANA-1.6 / ResNet-18 | 0.510 | `a1c4242f848309708af368d79a5c587605ab59b8b500ee88daf44b8b26288436` |
| `did_sana16_resnet50.pt` | WildFake SANA-1.6 / ResNet-50 | 0.550 | `5bdefbb2f1fa317646a6d3fd63bbc8145291f0d5cd85956c95617c0757e4b97f` |
| `did_sid_set_sd15_resnet18.pt` | SID_Set SD-1.5 / ResNet-18 | 0.610 | `130861f46b1af984e850fe6b5b42afd808c47c74ca7c45648ebabc0991e45344` |

The pooled PatchHead artifact is the release primary. DID remains an ablation; a
new DID reconstruction run was not needed for physics and would additionally
require the multi-gigabyte SD-1.5 or SANA reconstructor assets.

## SID_Set bounded validation

The existing pinned SID manifest was reused rather than expanding more data:

- dataset revision `dc03ead57929879319ce30a82bfcfb8d317b10bd`;
- three validation Parquet shards, 1,496,803,272 source bytes (1.394 GiB);
- 50 real, 50 full-synthetic, and 50 tampered images;
- 86,988,546 extracted image/mask bytes;
- deterministic seed 2026.

### Primary detector

Binary metrics intentionally include only the checkpoint's trained task: real
versus full-synthetic.

| Slice | n | AIGC alerts | Mean score | Range |
|---|---:|---:|---:|---:|
| Real | 50 | 0 (0%) | 0.00417 | 0.00043–0.15025 |
| Full-synthetic | 50 | 50 (100%) | 0.99923 | 0.99784–0.99940 |
| Tampered diagnostic | 50 | 2 (4%) | 0.04587 | 0.00047–0.99738 |

Real-versus-full-synthetic accuracy, balanced accuracy, and ROC AUC were all
1.0 on this bounded sample. This is not the full SID benchmark. The 4% tampered
alert rate is expected to be weak: label 2 was excluded from the pooled binary
training objective, and small local edits can preserve a real image's global
statistics.

### Integration safety and physics coverage

Detector-only and integrated runs used the same checkpoint, input order, device,
and batch size. All 150 scores, component scores, and verdicts were exactly
equal. The integrated run had zero physics errors.

| Label | Perspective applicable | Perspective status | Shadow applicable | Reflection applicable |
|---|---:|---|---:|---:|
| Real | 40/50 | 35 consistent, 5 indeterminate | 0/50 | 0/50 |
| Full-synthetic | 31/50 | 29 consistent, 2 indeterminate | 1/50 (consistent) | 0/50 |
| Tampered | 35/50 | 32 consistent, 3 indeterminate | 0/50 | 0/50 |

The shared PatchHead DINO grid was consulted for five candidate mirror regions
before confidence gates abstained. Each such output now records the backbone,
checkpoint SHA-256, pooled tag, float16 transfer, normalized 16×16 grid, and
`score_independent: true`. No reflection reached the three-pair applicability
minimum. This is conservative coverage, not reflection-pair accuracy.

### Weak tamper localization diagnostic

All 50 tampered masks were evaluable against PatchHead's 16×16 patch map:

| Diagnostic | Mean | Median |
|---|---:|---:|
| Patch ROC AUC | 0.582 | 0.591 |
| Mean inside-minus-outside score | 0.0132 | -0.0002 |
| Top-area IoU | 0.201 | 0.077 |
| Maximum-score patch inside mask | 30% of images | — |

These results show only weak spatial localization. Patch logits inherit the
image label during training and are not a tamper segmentation head. The UI and
API must not present them as a mask or causal attribution.

## Transformation validation

### WildFake range-read pilot

`did/fetch_wildfake.py` used HTTP range reads to select 10 COCO real and 10 ADM
fake test images (about 1.3 MB) without downloading any WildFake archive. The
sample is intentionally tiny and covers only two sources.

The official PatchHead clean-plus-14-transform evaluator produced 100% clean
accuracy/AUC and 99.64% mean transformed accuracy. Thirteen transforms were
20/20; Gaussian blur sigma 2.0 was 19/20 (95% accuracy, 0.99 AUC). The integrated
20-image run also preserved every detector score and verdict and completed with
zero physics errors.

The evaluator now accepts `--workers 0`, which avoids macOS multiprocessing's
inability to pickle the transform lambdas. Its model and metric behavior are
unchanged.

### SID patch and dense-feature stability

A balanced six-image SID slice (two per native label) was evaluated under all 14
transforms using the real pooled checkpoint and same-pass feature export:

| Metric | Result |
|---|---:|
| Verdict flips | 0/84 transformed images |
| Maximum score drift | 0.00637 |
| Mean patch-map Pearson correlation | 0.9890 |
| Minimum mean patch-map correlation | 0.8933 (`crop80`) |
| Mean same-coordinate DINO token cosine | 0.9799 |
| Minimum mean token cosine | 0.8454 (`crop80`) |

The crop is expected to be the weakest coordinate-wise comparison because it
changes which scene content occupies each grid cell. This six-image result is an
engineering stability check, not a benchmark confidence interval.

## Reproduction commands

From the repository root, with external checkpoints and existing ignored caches:

```bash
python infer.py \
  --image-dir physics/outputs/sid_multishard_150/images \
  --ckpt /path/to/checkpoints/patchhead_pooled.pt \
  --out physics/outputs/checkpoint_validation/sid_integrated.json \
  --batch 16 --device cpu --with-physics --physics-auto-proposals \
  --physics-proposal-mask-backend clipseg \
  --physics-proposal-feature-backend patchhead \
  --physics-proposal-object-backend torchvision \
  --physics-proposal-cache-dir cache/physics-auto \
  --physics-proposal-offline --physics-strict-proposal-models --pretty

physics-checkpoint-eval \
  --predictions physics/outputs/checkpoint_validation/sid_integrated.json \
  --baseline physics/outputs/checkpoint_validation/sid_detector_only.json \
  --manifest physics/outputs/sid_multishard_150/manifest.json \
  --dataset-name SID_Set \
  --backbone-revision 30c1109559f65dea34316b0d4842d35c5771fe11 \
  --output physics/outputs/checkpoint_validation/sid_evaluation.json \
  --report physics/outputs/checkpoint_validation/sid_evaluation.md --pretty

python patchhead/evaluate.py \
  --ds wildfake_pilot \
  --ckpt /path/to/checkpoints/patchhead_pooled.pt \
  --out physics/outputs/checkpoint_validation/wildfake_transforms \
  --limit 10 --workers 0

python -m harness.patchhead_robustness \
  --image-dir physics/outputs/sid_multishard_150/images \
  --checkpoint /path/to/checkpoints/patchhead_pooled.pt \
  --per-parent-limit 2 --batch-size 6 \
  --output physics/outputs/checkpoint_validation/sid_patch_dense.json --pretty
```

## What remains external

- Freeze independent reviewed shadow/contact and reflection-correspondence
  labels to measure pair precision and endpoint error.
- Broaden natural-scene mirror/shadow coverage beyond the zero-shot proposal
  models and tiny bounded pilots.
- If compute and licenses permit, compare trained shadow-instance/contact and
  mirror-instance proposal heads through the existing provider interfaces.
- Keep physics low-weight and abstention-aware in the final product; do not tune
  it to imitate SID class labels.
