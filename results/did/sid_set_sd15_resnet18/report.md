# Robustness Evaluation Summary

- **Clean accuracy:** 92.7%  (AUC 0.971)
- **Mean accuracy over transformed test sets:** 87.4%
- **Worst-case transformed accuracy:** 78.0%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 300 | 92.7% | 0.971 | 94.0% | 91.3% |
| jpeg90 | 300 | 91.3% | 0.968 | 92.0% | 90.7% |
| jpeg70 | 300 | 89.7% | 0.949 | 88.0% | 91.3% |
| jpeg50 | 300 | 86.3% | 0.936 | 86.0% | 86.7% |
| jpeg30 | 300 | 87.7% | 0.937 | 88.0% | 87.3% |
| blur0.5 | 300 | 91.7% | 0.972 | 92.0% | 91.3% |
| blur1.0 | 300 | 90.0% | 0.959 | 92.7% | 87.3% |
| blur2.0 | 300 | 83.0% | 0.918 | 84.0% | 82.0% |
| resize0.5 | 300 | 90.7% | 0.956 | 89.3% | 92.0% |
| resize0.25 | 300 | 83.0% | 0.899 | 76.7% | 89.3% |
| noise0.02 | 300 | 86.0% | 0.939 | 93.3% | 78.7% |
| noise0.05 | 300 | 85.0% | 0.931 | 88.7% | 81.3% |
| noise0.10 | 300 | 78.0% | 0.867 | 82.7% | 73.3% |
| jitter | 300 | 91.3% | 0.974 | 90.0% | 92.7% |
| crop80 | 300 | 89.3% | 0.966 | 92.7% | 86.0% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **9**
- False negatives (AIGC missed): **13**

Representative worst false positives (real, high AIGC score):
  - `sid_real_00133.npz`  score=0.999
  - `sid_real_00091.npz`  score=0.992
  - `sid_real_00116.npz`  score=0.949
  - `sid_real_00099.npz`  score=0.920
  - `sid_real_00046.npz`  score=0.864

Representative worst false negatives (AIGC, low score):
  - `sid_fake_00117.npz`  score=0.003
  - `sid_fake_00066.npz`  score=0.046
  - `sid_fake_00077.npz`  score=0.077
  - `sid_fake_00041.npz`  score=0.089
  - `sid_fake_00061.npz`  score=0.147
