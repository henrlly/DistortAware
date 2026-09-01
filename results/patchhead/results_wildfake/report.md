# PatchHead detector — robustness summary

- **Clean accuracy:** 98.5%  (AUC 1.000)
- **Mean over 14 transforms:** 99.0%
- **Worst transform:** 97.0%  (noise0.10)

| | PatchHead | DID (SD-1.5 / RN-18) |
|---|---:|---:|
| clean acc | 98.5% | 88.6% |
| clean AUC | 1.000 | 0.957 |
| mean transformed | 99.0% | 93.0% |
| worst transformed | 97.0% | 88.0% |

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1200 | 98.5% | 1.000 | 97.0% | 100.0% |
| jpeg90 | 300 | 99.3% | 1.000 | 100.0% | 98.7% |
| jpeg70 | 300 | 99.3% | 1.000 | 100.0% | 98.7% |
| jpeg50 | 300 | 99.0% | 1.000 | 99.3% | 98.7% |
| jpeg30 | 300 | 99.0% | 1.000 | 99.3% | 98.7% |
| blur0.5 | 300 | 100.0% | 1.000 | 100.0% | 100.0% |
| blur1.0 | 300 | 99.7% | 1.000 | 99.3% | 100.0% |
| blur2.0 | 300 | 98.3% | 1.000 | 99.3% | 97.3% |
| resize0.5 | 300 | 99.7% | 1.000 | 99.3% | 100.0% |
| resize0.25 | 300 | 98.7% | 1.000 | 99.3% | 98.0% |
| noise0.02 | 300 | 99.3% | 1.000 | 99.3% | 99.3% |
| noise0.05 | 300 | 98.7% | 1.000 | 99.3% | 98.0% |
| noise0.10 | 300 | 97.0% | 1.000 | 99.3% | 94.7% |
| jitter | 300 | 100.0% | 1.000 | 100.0% | 100.0% |
| crop80 | 300 | 98.7% | 1.000 | 100.0% | 97.3% |

## Error analysis (clean test set)

- False positives (real flagged AIGC): **18**
- False negatives (AIGC missed): **0**

Worst false positives:
  - `wildfake/real/imagenet_00038`  score=1.000
  - `wildfake/real/coco_00119`  score=1.000
  - `wildfake/real/imagenet_00037`  score=0.998
  - `wildfake/real/imagenet_00055`  score=0.994
  - `wildfake/real/imagenet_00058`  score=0.974

Worst false negatives:
