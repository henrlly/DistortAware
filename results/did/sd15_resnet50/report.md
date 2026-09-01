# Robustness Evaluation Summary

- **Clean accuracy:** 87.2%  (AUC 0.950)
- **Mean accuracy over transformed test sets:** 90.2%
- **Worst-case transformed accuracy:** 82.7%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1200 | 87.2% | 0.950 | 82.3% | 92.2% |
| jpeg90 | 300 | 88.7% | 0.978 | 98.7% | 78.7% |
| jpeg70 | 300 | 85.7% | 0.974 | 99.3% | 72.0% |
| jpeg50 | 300 | 83.0% | 0.971 | 100.0% | 66.0% |
| jpeg30 | 300 | 82.7% | 0.974 | 99.3% | 66.0% |
| blur0.5 | 300 | 95.0% | 0.992 | 98.0% | 92.0% |
| blur1.0 | 300 | 95.3% | 0.994 | 97.3% | 93.3% |
| blur2.0 | 300 | 92.0% | 0.982 | 95.3% | 88.7% |
| resize0.5 | 300 | 95.7% | 0.996 | 99.3% | 92.0% |
| resize0.25 | 300 | 91.3% | 0.991 | 98.0% | 84.7% |
| noise0.02 | 300 | 92.3% | 0.980 | 99.3% | 85.3% |
| noise0.05 | 300 | 87.0% | 0.973 | 99.3% | 74.7% |
| noise0.10 | 300 | 85.0% | 0.953 | 98.0% | 72.0% |
| jitter | 300 | 95.0% | 0.990 | 99.3% | 90.7% |
| crop80 | 300 | 94.7% | 0.998 | 100.0% | 89.3% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **106**
- False negatives (AIGC missed): **47**

Representative worst false positives (real, high AIGC score):
  - `imagenet_00065.npz`  score=1.000
  - `imagenet_00123.npz`  score=1.000
  - `imagenet_00147.npz`  score=1.000
  - `imagenet_00038.npz`  score=1.000
  - `imagenet_00121.npz`  score=0.999

Representative worst false negatives (AIGC, low score):
  - `VQDM_00017.npz`  score=0.002
  - `ADM_00136.npz`  score=0.002
  - `ADM_00032.npz`  score=0.002
  - `VQDM_00033.npz`  score=0.004
  - `VQDM_00014.npz`  score=0.007
