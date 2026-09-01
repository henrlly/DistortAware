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
