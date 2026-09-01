# Robustness Evaluation Summary

- **Clean accuracy:** 58.3%  (AUC 0.575)
- **Mean accuracy over transformed test sets:** 51.3%
- **Worst-case transformed accuracy:** 46.0%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 300 | 58.3% | 0.575 | 58.7% | 58.0% |
| jpeg90 | 300 | 55.0% | 0.547 | 76.7% | 33.3% |
| jpeg70 | 300 | 50.0% | 0.491 | 84.7% | 15.3% |
| jpeg50 | 300 | 47.7% | 0.453 | 82.0% | 13.3% |
| jpeg30 | 300 | 48.7% | 0.471 | 79.3% | 18.0% |
| blur0.5 | 300 | 57.7% | 0.592 | 40.7% | 74.7% |
| blur1.0 | 300 | 51.0% | 0.537 | 22.7% | 79.3% |
| blur2.0 | 300 | 53.3% | 0.544 | 39.3% | 67.3% |
| resize0.5 | 300 | 50.3% | 0.506 | 30.7% | 70.0% |
| resize0.25 | 300 | 48.7% | 0.527 | 34.7% | 62.7% |
| noise0.02 | 300 | 50.0% | 0.491 | 61.3% | 38.7% |
| noise0.05 | 300 | 48.3% | 0.468 | 62.0% | 34.7% |
| noise0.10 | 300 | 46.0% | 0.451 | 42.7% | 49.3% |
| jitter | 300 | 57.0% | 0.578 | 57.3% | 56.7% |
| crop80 | 300 | 54.0% | 0.540 | 48.0% | 60.0% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **62**
- False negatives (AIGC missed): **63**

Representative worst false positives (real, high AIGC score):
  - `sid_real_00030.npz`  score=1.000
  - `sid_real_00032.npz`  score=1.000
  - `sid_real_00135.npz`  score=1.000
  - `sid_real_00126.npz`  score=0.999
  - `sid_real_00128.npz`  score=0.999

Representative worst false negatives (AIGC, low score):
  - `sid_fake_00006.npz`  score=0.000
  - `sid_fake_00138.npz`  score=0.002
  - `sid_fake_00031.npz`  score=0.006
  - `sid_fake_00127.npz`  score=0.006
  - `sid_fake_00012.npz`  score=0.006
