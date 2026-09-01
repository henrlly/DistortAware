# Robustness Evaluation Summary

- **Clean accuracy:** 85.3%  (AUC 0.944)
- **Mean accuracy over transformed test sets:** 78.8%
- **Worst-case transformed accuracy:** 68.7%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 300 | 85.3% | 0.945 | 76.0% | 94.7% |
| jpeg90 | 300 | 85.0% | 0.927 | 80.7% | 89.3% |
| jpeg70 | 300 | 84.3% | 0.925 | 86.0% | 82.7% |
| jpeg50 | 300 | 79.7% | 0.893 | 84.0% | 75.3% |
| jpeg30 | 300 | 78.3% | 0.879 | 79.3% | 77.3% |
| blur0.5 | 300 | 83.3% | 0.941 | 72.7% | 94.0% |
| blur1.0 | 300 | 76.3% | 0.895 | 58.0% | 94.7% |
| blur2.0 | 300 | 71.3% | 0.836 | 53.3% | 89.3% |
| resize0.5 | 300 | 76.7% | 0.893 | 59.3% | 94.0% |
| resize0.25 | 300 | 68.7% | 0.835 | 47.3% | 90.0% |
| noise0.02 | 300 | 83.3% | 0.904 | 83.3% | 83.3% |
| noise0.05 | 300 | 77.0% | 0.880 | 84.0% | 70.0% |
| noise0.10 | 300 | 71.0% | 0.801 | 76.7% | 65.3% |
| jitter | 300 | 84.7% | 0.945 | 76.7% | 92.7% |
| crop80 | 300 | 84.0% | 0.916 | 76.7% | 91.3% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **36**
- False negatives (AIGC missed): **8**

Representative worst false positives (real, high AIGC score):
  - `sid_real_00146.npz`  score=0.994
  - `sid_real_00036.npz`  score=0.991
  - `sid_real_00030.npz`  score=0.973
  - `sid_real_00007.npz`  score=0.967
  - `sid_real_00009.npz`  score=0.967

Representative worst false negatives (AIGC, low score):
  - `sid_fake_00041.npz`  score=0.030
  - `sid_fake_00117.npz`  score=0.048
  - `sid_fake_00005.npz`  score=0.062
  - `sid_fake_00059.npz`  score=0.182
  - `sid_fake_00067.npz`  score=0.292
