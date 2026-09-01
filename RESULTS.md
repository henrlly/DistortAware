# Benchmark results

## Scope and status

This document consolidates the completed DistortAware benchmark results and the
local PrismGuard evidence ledger as of 2026-09-01. The primary shared WildFake
manifest, `wildfake_benchmark.csv`, contains 1,000 images per transform: 500
COCO real images and 500 DALL-E generated images. The same seven transforms
are applied to every method: clean, JPEG quality 90, Gaussian blur (1.0),
resize to 0.5, Gaussian noise (0.05), colour jitter, and 80% crop.

The PrismGuard sections deliberately separate competition-relevant results
from narrow engineering diagnostics, controlled synthetic validation, and
runtime smoke tests. In particular, the CIFAKE runs cannot select a Track 5
submission, and physics/forensics are interpretability-only with permanent
fusion weight `alpha = 0`.

| Method | Status | Result location |
|---|---|---|
| PatchHead baseline | Complete | `results/parallel_evaluation/patchhead_baseline/` |
| PatchHead distortion-aware | Complete | `results/parallel_evaluation/distortion_aware/` |
| Filter segmentation baseline | Complete local pilots | `results/filter_based_approach/reports/evaluation.json` |
| Physics/light solver | Controlled fixtures complete; no real-image AIGC benchmark | PrismGuard evidence ledger, summarized below |
| DID | Completed historical local evaluations | `results/did/` |
| CIFAKE handcrafted detector | Complete; rejected as submission candidate | PrismGuard evidence ledger, summarized below |
| Frozen DINOv3-L CIFAKE proxy | Complete; selection-ineligible | PrismGuard evidence ledger, summarized below |
| Legal 50K group-OOF PrismGuard | Not run; hard gates remain | Shared-A100 readiness summary below |

The PatchHead tables below report the committed results. Accuracy equals
balanced accuracy because each transform has 500 real and 500 generated
examples.

## Difference-In-Differences (DiD) analysis

SD-1.5 (1.07B) + ResNet-18 ×2

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 88.6% | 0.957 | 85.3% | 90.9% |
| JPEG quality 30 | 88.0% | 0.941 | 91.7% | 85.4% |
| Noise \(\sigma=0.05\) | 90.8% | 0.962 | 87.9% | 92.8% |
| Resize ¼ | 92.1% | 0.968 | 89.4% | 93.9% |

SD-1.5 + ResNet-50 ×2

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 87.2% | 0.950 | 83.6% | 89.7% |
| JPEG quality 30 | 82.7% | 0.918 | 88.5% | 78.6% |
| Noise \(\sigma=0.05\) | 89.4% | 0.947 | 85.0% | 92.3% |
| Resize ¼ | 91.0% | 0.958 | 87.7% | 93.2% |

SANA-1.6B + ResNet-18 ×2

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 88.2% | 0.963 | 84.9% | 90.5% |
| JPEG quality 30 | 91.3% | 0.965 | 90.1% | 92.1% |
| Noise \(\sigma=0.05\) | 93.6% | 0.971 | 91.8% | 94.8% |
| Resize ¼ | 87.5% | 0.939 | 92.4% | 84.2% |

SANA-1.6B + ResNet-50 ×2

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 88.3% | 0.966 | 85.7% | 90.1% |
| JPEG quality 30 | 90.9% | 0.964 | 89.3% | 92.0% |
| Noise \(\sigma=0.05\) | 93.2% | 0.970 | 91.0% | 94.6% |
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

## Filter segmentation baseline

The compact RGB/high-pass residual U-Net was evaluated as an independent
three-class/segmentation sidecar. It is useful as inspectable artifact evidence,
but its mask is weak localization rather than validated forensic segmentation.

| Dataset | Samples | Binary accuracy | Balanced accuracy | Authentic accuracy | AI accuracy | Tamper-mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| SID_Set | 300 | 67.3% | 57.0% | 26.0% | 88.0% | 15.5% |
| WildFake | 200 | 74.0% | — | 82.0% | 66.0% | — |

On WildFake, the mean predicted mask area was 22.9% for authentic images and
21.6% for synthetic images. This near-equality and the low SID mask IoU are why
the sidecar remains explanation-only and never votes on the detector score.

## PrismGuard predictor contract

The locked prediction boundary is:

```text
pred = calibrated_probability(dino_logit)
```

Physics and forensic diagnostics cannot enter features, training loss, head or
calibration selection, thresholding, test-time aggregation, or the returned
challenge score. Their fusion coefficient is permanently `alpha = 0`. The
required challenge output remains exactly `image_path` plus `pred`; optional
explanations are written separately.

No legal 50K, source/generator/content/duplicate-group nested-OOF PrismGuard
model has been trained yet. Results below are therefore either rejected
baselines, selection-ineligible proxies, controlled validation, or frozen
external diagnostics—not a competition-ready PrismGuard score.

## Rejected handcrafted CIFAKE baseline

The first sealed baseline used 102 handcrafted global features at 32×32, a
standardized logistic head, and scalar temperature calibration. It has 104
fitted scalars and 308 deployed prediction-state scalars, no Transformer, no
geometry model, and physics `alpha = 0`.

### Internal CIFAKE diagnostic

| Metric | Clean-only head |
|---|---:|
| Clean ROC-AUC | 0.9212 |
| Equal-family macro corruption ROC-AUC | 0.8390 |
| Hierarchical macro corruption ROC-AUC | 0.8552 |
| Worst-transform ROC-AUC | 0.6913 |
| Worst transform | Gaussian noise, sigma 0.10 |

These strong-looking numbers are not generalization evidence: CIFAKE is 32×32,
uses one fake generator, and its authentic source and fake generator are
perfectly label-confounded. The paired-consistency head was worse than the
clean-only head: hierarchical macro AUC 0.8467 versus 0.8552, observed delta
`-0.00650`, with 95% paired bootstrap interval `[-0.01064, -0.00206]`.

### Frozen external diagnostics

The following datasets were evaluated once after sealing the handcrafted model.
They were not used for training, calibration, thresholding, model selection,
or early stopping.

| Frozen pilot | Sample | Native clean AUC | Native macro AUC | Standardized clean AUC | Standardized macro AUC |
|---|---|---:|---:|---:|---:|
| SID_Set, real vs full synthetic | 40 + 40 | 0.5213 | 0.5202 | 0.4944 | 0.5018 |
| SID_Set, real vs any manipulated | 40 + 80 | 0.5656 | 0.5569 | 0.5338 | 0.5354 |
| Organizer WildFake demo, COCO vs DALL-E 3 | 40 + 40 | 0.5425 | 0.5702 | 0.5425 | 0.5438 |
| Broad WildFake publisher test | 80 + 80 | 0.6113 | 0.6235 | — | — |
| CommunityForensics CompEval | 40 + 40 | 0.5519 | — | 0.5231 | 0.5168 |

Additional diagnostic details:

- SID_Set real-versus-tampered achieved native clean/macro AUC 0.6100/0.5936,
  falling to 0.5731/0.5690 after symmetric square-512/JPEG-Q95 normalization.
- The broad WildFake macro AUC 95% interval was `[0.5330, 0.7041]`, but generator
  clean AUC ranged from 0.3891 for DDPM and 0.4922 for VQGAN to 0.8641 for
  DALL-E 2. This is generator dependence, not robust generalization.
- CommunityForensics standardized worst-transform AUC was 0.4838 on colour
  jitter. Its macro AUC 95% interval was `[0.3931, 0.6301]`.
- The organizer demo pilot used exact publisher pools, while the broad
  WildFake pilot had zero intersection with the 13,841 organizer demo paths.
  These path checks do not replace the still-required independent full
  SHA-256 plus dHash64 organizer trust root.

All four external pilots reject the handcrafted model as a submission
candidate. It should be discarded when a legal Transformer model is available.

## Frozen DINOv3-L CIFAKE proxy

A real frozen DINOv3-L/16 proxy was run on the same narrow CIFAKE setting. The
backbone is `timm/vit_large_patch16_dinov3.lvd1689m` at pinned revision
`30c1109559f65dea34316b0d4842d35c5771fe11`, with checkpoint SHA-256
`45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941`.
All 303,079,424 backbone parameters were frozen.

| Frozen feature head | Clean AUC | Equal-family macro AUC | Worst-transform AUC |
|---|---:|---:|---:|
| CLS only | 0.9931 | 0.9730 | 0.9403 |
| CLS + fixed 4×4 spatial summary | 0.9942 | 0.9773 | 0.9465 |
| Candidate minus CLS | +0.00113 | +0.00432 | +0.00618 |

The paired atomic-image bootstrap mean macro gain was `+0.00431`, with 95%
interval `[+0.00244, +0.00612]` over 1,000 replicates. The positive interval is
useful engineering evidence for retaining a spatial-summary bake-off, but it
does not pass the predeclared `+0.005` macro-gain threshold and does not include
split/refit/source/generator uncertainty. The data remain 32×32,
source-confounded, and single-generator, so neither head can qualify a Track 5
submission.

The local browser-demo head is a deterministic CPU refit of the locked policy.
Against the original CUDA calibrated logits it reached Pearson correlation
`0.9999999903`, mean absolute difference `0.000858`, and maximum absolute
difference `0.00476`. A direct Mac CPU smoke produced a real DINO score in
10.6 seconds for one 384 px crop; requesting unavailable diagnostics left the
verdict byte-identical. This establishes runnable plumbing, not accuracy.

## Physics and forensic diagnostics

### Explicit distant-light solver

The linear-light, robustly cross-fitted four-term model
`[1, nx, ny, nz]` passed its controlled analytic-oracle suite:

- 20/20 clean Lambertian scenes were usable; median held-out Q90 absolute log
  residual was `0.000937`.
- Maximum light-vector absolute error was `0.000453`; maximum relative albedo
  RMSE was `0.000378`.
- Across 100 injected local-light-mismatch cases, every scene was monotonic;
  median Spearman correlation was effectively 1.0 and median endpoint effect
  was `0.01306`, above the locked `0.01` threshold.
- Four non-proportional spatial-albedo layouts remained stable, with residual
  score range `3.13e-8`.
- Constant normals correctly abstained with quality `q = 0`; shuffled normals
  destroyed the clean base fit; linear-light luminance outperformed fitting in
  gamma-coded luminance.
- All seven representative official views remained usable. Six of eight
  deliberately severe stress views abstained (75%) because of quality,
  bootstrap, local-zone, solver, or cross-fit failures, demonstrating the
  intended fail-open diagnostic behavior.

This suite uses analytic depth/normals and a simple Lambertian renderer. It does
not validate Metric3Dv2, Depth Anything V2, automatic real-image regions, or
real AIGC discrimination.

### Four-term versus nine-term lighting

The nine-term spherical-harmonic ablation was less reliable than the simple
four-term model. SH9 was usable for only 67% of scene/strength combinations,
versus 100% for the four-term solver. Its median endpoint effect was 0.01268
versus 0.01415 on paired usable scenes, an effect ratio of 0.896. SH9 therefore
remains rejected as over-capacity for the core solver.

### Six independent forensic cue cards

Controlled fixtures cover illuminant/colour temperature, cast-shadow direction,
reflection/highlight geometry, perspective/vanishing points, noise/sensor
residuals, and spectral/codec traces.

| Aggregate fixture metric | Result |
|---|---:|
| Clean applicability correctness | 100% |
| Clean evidence-direction correctness | 100% |
| Consistent-scene false-evidence rate | 0% |
| Violation detection across official views | 94.4% |
| Mean applicability/direction stability | 98.6% |

Noise/sensor residuals were the weakest cue, with 73.3% violation detection;
spectral/codec traces reached 93.3%, and the other four controlled cues reached
100%. Shadow, reflection, and perspective tests use caller-supplied geometry
and exact crop transport, so these figures validate cue arithmetic and schema
behavior—not automatic scene understanding or real-world forensic accuracy.

### Physics conclusion

There is no real held-out source/generator/corruption AUC or paired-bootstrap
gain for physics, no completed alternate learned-geometry benchmark, and no
hard-authentic/physically-relit/flat-overcast falsification suite on real data.
Physics has therefore not earned—and under the current architecture cannot
earn—prediction admission. It remains lazy, fail-open, explanation-only, and
permanently neutral with `alpha = 0`.

## Operational and release readiness

- The local VFM preflight is not ready and recorded 14 blockers, including the
  legal checkpoint and dataset ledgers, source-verified corpus manifest, full
  organizer inventory, suitable accelerator, RAM, and persistent disk.
- The shared-A100 job pack's focused trust-boundary and signed-cache fixture
  tests pass, but the pack remains explicitly **NOT AUTHORIZED TO RUN**. No VM
  connection, CUDA extraction, 50K training run, or target-A100 throughput/VRAM
  measurement has occurred.
- The serious next model remains frozen DINOv3-L/16 at locked 384 px
  preprocessing, followed by nested group-OOF DINO-only head/calibration
  selection. DINOv3-H+/16 is conditional on an L benchmark and resource gate;
  DINOv3-7B is excluded.
- SID_Set, organizer WildFake, broad WildFake, and CommunityForensics remain
  one-shot frozen evaluation sets and cannot enter training, calibration,
  architecture choice, thresholding, or early stopping.

## PrismGuard evidence integrity

The consolidated local evidence is bound to immutable payloads rather than
copied training examples. Key bindings include:

- DINOv3-L checkpoint SHA-256:
  `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941`.
- DINO proxy evidence payload SHA-256:
  `ec42269910c7182c541a1aa7dc7e4facbc48fed9b22694a218c370a610b4972a`.
- Official corruption harness SHA-256:
  `70a43a82d636dd34f8c6a6da3c968ed1b8b6bf1ae5ff85e01d5b6ce1780f84ee`.
- Controlled solver report SHA-256:
  `19e783ff31f9908572bf1862ee4f8ce32457b25fc93f4cf698446f409fc3720a`.
- Forensic-cue validation SHA-256:
  `b05fd94d592bcae7b078a530a8cfcc03d7d79931b4a246a8d769d77f9ac93a88`.
- SH9 ablation report SHA-256:
  `3507444309b3cf33480143fa27e7a60d8b34bbb9e9380ed3f78f33b310c74c29`.

The PrismGuard machine-readable artifacts remain in the separate local
evidence package and are summarized here without copying datasets, checkpoints,
absolute home paths, or organizer demonstration items into this repository.

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
