# PatchHead detector — robustness summary

- **Clean accuracy:** 99.6%  (AUC 1.000)
- **Mean over 14 transforms:** 98.6%
- **Worst transform:** 96.3%  (resize0.25)

| | PatchHead | DID (SD-1.5 / RN-18) |
|---|---:|---:|
| clean acc | 99.6% | 87.2% |
| clean AUC | 1.000 | 0.946 |
| mean transformed | 98.6% | 89.3% |
| worst transformed | 96.3% | 82.3% |

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1500 | 99.6% | 1.000 | 99.3% | 99.9% |
| jpeg90 | 600 | 99.7% | 1.000 | 100.0% | 99.3% |
| jpeg70 | 600 | 99.2% | 0.999 | 99.3% | 99.0% |
| jpeg50 | 600 | 98.5% | 0.998 | 98.7% | 98.3% |
| jpeg30 | 600 | 98.0% | 0.998 | 98.7% | 97.3% |
| blur0.5 | 600 | 99.8% | 1.000 | 100.0% | 99.7% |
| blur1.0 | 600 | 99.2% | 1.000 | 98.7% | 99.7% |
| blur2.0 | 600 | 98.2% | 0.999 | 96.7% | 99.7% |
| resize0.5 | 600 | 99.0% | 1.000 | 98.7% | 99.3% |
| resize0.25 | 600 | 96.3% | 0.996 | 98.3% | 94.3% |
| noise0.02 | 600 | 99.3% | 1.000 | 99.3% | 99.3% |
| noise0.05 | 600 | 97.8% | 0.999 | 97.3% | 98.3% |
| noise0.10 | 600 | 97.3% | 0.998 | 95.7% | 99.0% |
| jitter | 600 | 99.8% | 1.000 | 100.0% | 99.7% |
| crop80 | 600 | 98.2% | 0.999 | 99.7% | 96.7% |

## Error analysis (clean test set)

- False positives (real flagged AIGC): **5**
- False negatives (AIGC missed): **1**

Worst false positives:
  - `wildfake/real/imagenet_00038`  score=0.999
  - `wildfake/real/coco_00119`  score=0.999
  - `wildfake/real/imagenet_00047`  score=0.991
  - `wildfake/real/coco_00100`  score=0.956
  - `wildfake/real/imagenet_00081`  score=0.859

Worst false negatives:
  - `sid_set/fake/sid_fake_00070`  score=0.737
