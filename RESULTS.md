# Benchmark results

## Scope and status

This document summarizes completed results from the shared WildFake benchmark
manifest, `wildfake_benchmark.csv`. Each transform contains 1,000 images: 500
COCO real images and 500 DALL-E generated images. The same seven transforms
are applied to every method: clean, JPEG quality 90, Gaussian blur (1.0),
resize to 0.5, Gaussian noise (0.05), colour jitter, and 80% crop.

| Method | Status | Result location |
|---|---|---|
| PatchHead baseline | Complete | `results/parallel_evaluation/patchhead_baseline/` |
| PatchHead distortion-aware | Complete | `results/parallel_evaluation/distortion_aware/` |
| Filter baseline | Complete remotely; not yet committed | `results/parallel_evaluation/filter/` |
| Physics | Running | `results/parallel_evaluation/physics_benchmark/` |
| DID | Training | `checkpoints/did/pooled_sd15_resnet18.pt` |

The PatchHead tables below report the committed results. Accuracy equals
balanced accuracy because each transform has 500 real and 500 generated
examples.

## Difference-In-Differences (DiD) analysis

SD-1.5 (1.07B) + ResNet-18 ×2

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 88.6% | 0.957 | 85.3% | 90.9% |
| JPEG quality 30 | 88.0% | 0.941 | 91.7% | 85.4% |
| Noise $\sigma=0.05$ | 90.8% | 0.962 | 87.9% | 92.8% |
| Resize ¼ | 92.1% | 0.968 | 89.4% | 93.9% |

SD-1.5 + ResNet-50 ×2

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 87.2% | 0.950 | 83.6% | 89.7% |
| JPEG quality 30 | 82.7% | 0.918 | 88.5% | 78.6% |
| Noise $\sigma=0.05$ | 89.4% | 0.947 | 85.0% | 92.3% |
| Resize ¼ | 91.0% | 0.958 | 87.7% | 93.2% |

SANA-1.6B + ResNet-18 ×2

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 88.2% | 0.963 | 84.9% | 90.5% |
| JPEG quality 30 | 91.3% | 0.965 | 90.1% | 92.1% |
| Noise $\sigma=0.05$ | 93.6% | 0.971 | 91.8% | 94.8% |
| Resize ¼ | 87.5% | 0.939 | 92.4% | 84.2% |

SANA-1.6B + ResNet-50 ×2

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 88.3% | 0.966 | 85.7% | 90.1% |
| JPEG quality 30 | 90.9% | 0.964 | 89.3% | 92.0% |
| Noise $\sigma=0.05$ | 93.2% | 0.970 | 91.0% | 94.6% |
| Resize ¼ | 92.4% | 0.969 | 89.8% | 94.1% |

## PatchHead: baseline versus distortion-aware

| Transform | Baseline accuracy | Distortion-aware accuracy | Change | Baseline ROC-AUC | Distortion-aware ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Clean | 88.2% | 94.9% | +6.7 pp | 0.955 | 0.990 |
| JPEG 90 | 90.3% | 95.3% | +5.0 pp | 0.965 | 0.992 |
| Blur 1.0 | 82.4% | 88.5% | +6.1 pp | 0.923 | 0.963 |
| Resize 0.5 | 80.5% | 75.8% | -4.7 pp | 0.907 | 0.881 |
| Noise 0.05 | 85.9% | 89.8% | +3.9 pp | 0.942 | 0.977 |
| Colour jitter | 88.2% | 95.5% | +7.3 pp | 0.955 | 0.993 |
| Crop 80% | 82.1% | 89.1% | +7.0 pp | 0.913 | 0.961 |
| **Unweighted mean** | **85.4%** | **89.8%** | **+4.5 pp** | **0.937** | **0.965** |

### Interpretation

- The distortion-aware model improves on six of seven conditions. The largest
  accuracy gains are for colour jitter (+7.3 percentage points), crop (+7.0
  pp), and clean inputs (+6.7 pp).
- Its clean result is materially stronger: 94.9% accuracy and 0.990 ROC-AUC,
  compared with 88.2% and 0.955 for the baseline.
- Both models retain high generated-image recall across conditions (at least
  90.4% for the baseline and 92.8% for distortion-aware). The main external
  difference is real-image specificity: distortion-aware raises COCO-real
  accuracy from 84.8% to 95.6% on clean images and from 88.0% to 96.2% after
  JPEG 90.
- Resize is the exception. Distortion-aware falls to 75.8%, driven primarily
  by COCO-real accuracy dropping from 69.2% to 58.0%. This is a false-positive
  regression for resized real images, despite DALL-E recall rising from 91.8%
  to 93.6%.

### Conclusion

On this benchmark, distortion-aware PatchHead is the better default for clean,
JPEG, blur, noise, jitter, and crop conditions. The resize regression means it
should not yet be described as uniformly more robust; resized real images need
targeted calibration or augmentation before that claim is justified.

## Reproducibility notes

- Both variants were evaluated on the same manifest fingerprint:
  `b55280a497240a44d5f42f0e76a6363ae18b8a0cfe93b73c26d4de69c5696603`.
- Every completed PatchHead transform returned all 1,000 expected records with
  zero missing records, duplicates, or reported inference errors.
- Source-specific rows contain a single ground-truth class (COCO is real and
  DALL-E is generated). Their per-source accuracy is therefore the useful
  quantity; per-source balanced accuracy and ROC-AUC are not meaningful.
- These are external benchmark results. They should not be conflated with the
  internal training/validation accuracies printed during training.

## Physics engine: bounded explanation-sidecar validation

The physics engine checks perspective and vanishing points, cast-shadow
geometry, and planar reflections. It returns applicability-aware evidence and
cannot change the primary detector score, threshold, or verdict. Accuracy and
ROC-AUC are therefore not applicable to this component; the relevant questions
are whether it preserves the primary result, how often each cue is testable,
and whether its automatic proposals and safety gates behave as intended.

### Primary-score noninterference and SID_Set coverage

A deterministic, storage-bounded SID_Set sample contained 50 real, 50
full-synthetic, and 50 tampered images. Detector-only and physics-integrated
runs used the same checkpoint, order, device, and batch size. All 150 primary
scores, component scores, and verdicts were exactly equal, with zero physics
errors.

| SID_Set slice | Images | Perspective applicable | Perspective result | Cast shadow applicable | Reflection applicable |
|---|---:|---:|---|---:|---:|
| Real | 50 | 40/50 | 35 consistent, 5 indeterminate | 0/50 | 0/50 |
| Full-synthetic | 50 | 31/50 | 29 consistent, 2 indeterminate | 1/50 (consistent) | 0/50 |
| Tampered | 50 | 35/50 | 32 consistent, 3 indeterminate | 0/50 | 0/50 |

Perspective was applicable to 106/150 images (70.7%). Automatic shadow and
reflection coverage was much lower: one shadow result was applicable and no
reflection reached the required three-pair gate. Five candidate mirror regions
used the same-pass PatchHead DINO grid before safely abstaining. A separate
20-image WildFake pilot also preserved every detector score and verdict and
completed with zero physics errors.

### Automatic shadow and mirror proposal quality

Automatic region proposals were evaluated on deterministic 24-image subsets of
the SBU shadow test set and PMD mirror test set. The learned CLIPSeg profiles
substantially outperformed the deterministic heuristic fallbacks.

| Cue/source | Proposal backend | IoU | Dice | Precision | Recall |
|---|---|---:|---:|---:|---:|
| Shadow / SBU | CLIPSeg + photometric prior | 0.4866 | 0.6107 | 0.8335 | 0.5589 |
| Shadow / SBU | Heuristic | 0.0404 | 0.0673 | 0.4924 | 0.0422 |
| Mirror / PMD | CLIPSeg + framed-mirror prior | 0.2981 | 0.3670 | 0.5344 | 0.3767 |
| Mirror / PMD | Heuristic | 0.1015 | 0.1104 | 0.1031 | 0.1232 |

These are macro image-level mask metrics on small subsets. They measure only
region proposal overlap—not object-to-shadow ownership, reflection
correspondence, geometric correctness, or AIGC detection accuracy. The source
datasets also carry non-commercial terms and were not redistributed with the
repository.

### End-to-end WildFake demonstration wall

The learned browser profile was run on an evaluation-only 12-image wall with
six COCO val2017 real images and six DALL-E Advanced images. The detector was
blind to the human-visible labels.

| Cue | Consistent | Inconsistent | Indeterminate | Not applicable |
|---|---:|---:|---:|---:|
| Perspective | 5 | 0 | 3 | 4 |
| Cast shadow | 0 | 0 | 1 | 11 |
| Reflection | 0 | 2 | 0 | 10 |

The two reflection inconsistencies occurred on DALL-E examples, but this tiny
selected wall is not an accuracy estimate. Same-pass DINO produced too few
accepted matches for those mirror regions, so the engine disclosed that its
local-appearance fallback supplied the usable correspondences. Every response
reported that physics had no influence on the detector verdict.

### Transformation safety smoke

One controlled consistent fixture was evaluated clean and under 14 image
transformations. A hard flip means a cue changed between a definitive
`consistent` and `inconsistent` result.

| Cue | Applicability retained | Hard flips | Mean score drift | Maximum score drift |
|---|---:|---:|---:|---:|
| Perspective | 13/14 (92.9%) | 0 | 0.000 | 0.000 |
| Cast shadow | 12/14 (85.7%) | 0 | 0.128 | 0.356 |
| Reflection | 13/14 (92.9%) | 0 | 0.007 | 0.090 |

The clean-plus-transform run took 15.55 seconds on the development CPU. Crop
and noise caused some abstention, while the four-pair inconsistency gate
converted an unsupported three-pair shadow result to `indeterminate` and
prevented a hard flip. This validates the gate on one synthetic fixture, not
production robustness.

### Interpretation and conclusion

- Physics remained exactly non-causal to the primary detector in every matched
  integration run.
- Perspective provided useful coverage, while automatic natural-scene shadow
  and reflection evidence remained sparse and appropriately abstention-heavy.
- Learned shadow and mirror proposals improved substantially over the
  heuristic fallbacks, but proposal overlap is not end-to-end geometry
  accuracy.
- The engine is ready to serve as a conservative, human-readable explanation
  sidecar. It is not supported as a standalone detector or a weighted fusion
  input.
- The full 1,000-image-per-transform shared physics harness has not been
  committed, so no full-benchmark physics claim is made here.

The bounded runs use deterministic seeds and storage caps. The supporting
records are documented in `physics/docs/checkpoint_validation.md`,
`physics/docs/automatic_proposals.md`, and `browser_product/VALIDATION.md`.
The current cleaned-repository suite passes all 83 discovered physics tests.
