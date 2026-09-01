# Robustness Evaluation Summary

- **Clean accuracy:** 88.6%  (AUC 0.957)
- **Mean accuracy over transformed test sets:** 93.0%
- **Worst-case transformed accuracy:** 88.0%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1200 | 88.6% | 0.957 | 86.2% | 91.0% |
| jpeg90 | 300 | 95.3% | 0.996 | 100.0% | 90.7% |
| jpeg70 | 300 | 89.7% | 0.994 | 100.0% | 79.3% |
| jpeg50 | 300 | 89.0% | 0.988 | 100.0% | 78.0% |
| jpeg30 | 300 | 88.0% | 0.988 | 100.0% | 76.0% |
| blur0.5 | 300 | 97.7% | 0.998 | 99.3% | 96.0% |
| blur1.0 | 300 | 97.3% | 0.996 | 98.0% | 96.7% |
| blur2.0 | 300 | 91.7% | 0.982 | 96.0% | 87.3% |
| resize0.5 | 300 | 97.3% | 0.997 | 100.0% | 94.7% |
| resize0.25 | 300 | 89.7% | 0.977 | 99.3% | 80.0% |
| noise0.02 | 300 | 95.7% | 0.996 | 99.3% | 92.0% |
| noise0.05 | 300 | 93.3% | 0.995 | 99.3% | 87.3% |
| noise0.10 | 300 | 92.0% | 0.986 | 99.3% | 84.7% |
| jitter | 300 | 96.3% | 0.998 | 100.0% | 92.7% |
| crop80 | 300 | 89.3% | 0.995 | 100.0% | 78.7% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **83**
- False negatives (AIGC missed): **54**

Representative worst false positives (real, high AIGC score):
  - `imagenet_00038.npz`  score=1.000
  - `imagenet_00121.npz`  score=1.000
  - `imagenet_00022.npz`  score=1.000
  - `imagenet_00045.npz`  score=0.999
  - `imagenet_00126.npz`  score=0.999

Representative worst false negatives (AIGC, low score):
  - `VQDM_00033.npz`  score=0.000
  - `VQDM_00089.npz`  score=0.008
  - `VQDM_00116.npz`  score=0.010
  - `VQDM_00146.npz`  score=0.010
  - `ADM_00071.npz`  score=0.018
