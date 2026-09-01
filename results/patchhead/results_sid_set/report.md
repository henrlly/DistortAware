# PatchHead detector — robustness summary

- **Clean accuracy:** 100.0%  (AUC 1.000)
- **Mean over 14 transforms:** 99.3%
- **Worst transform:** 98.3%  (resize0.25)

| | PatchHead | DID (SD-1.5 / RN-18) |
|---|---:|---:|
| clean acc | 100.0% | 92.7% |
| clean AUC | 1.000 | 0.971 |
| mean transformed | 99.3% | 87.4% |
| worst transformed | 98.3% | 78.0% |

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 300 | 100.0% | 1.000 | 100.0% | 100.0% |
| jpeg90 | 300 | 100.0% | 1.000 | 100.0% | 100.0% |
| jpeg70 | 300 | 99.7% | 1.000 | 99.3% | 100.0% |
| jpeg50 | 300 | 100.0% | 1.000 | 100.0% | 100.0% |
| jpeg30 | 300 | 99.0% | 1.000 | 100.0% | 98.0% |
| blur0.5 | 300 | 100.0% | 1.000 | 100.0% | 100.0% |
| blur1.0 | 300 | 99.0% | 1.000 | 98.0% | 100.0% |
| blur2.0 | 300 | 98.7% | 1.000 | 97.3% | 100.0% |
| resize0.5 | 300 | 99.3% | 1.000 | 98.7% | 100.0% |
| resize0.25 | 300 | 98.3% | 0.998 | 98.0% | 98.7% |
| noise0.02 | 300 | 99.3% | 1.000 | 100.0% | 98.7% |
| noise0.05 | 300 | 98.7% | 1.000 | 99.3% | 98.0% |
| noise0.10 | 300 | 98.3% | 0.999 | 98.7% | 98.0% |
| jitter | 300 | 100.0% | 1.000 | 100.0% | 100.0% |
| crop80 | 300 | 100.0% | 1.000 | 100.0% | 100.0% |

## Error analysis (clean test set)

- False positives (real flagged AIGC): **0**
- False negatives (AIGC missed): **0**

Worst false positives:

Worst false negatives:
