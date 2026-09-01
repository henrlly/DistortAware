# Robustness Evaluation Summary

- **Clean accuracy:** 87.2%  (AUC 0.946)
- **Mean accuracy over transformed test sets:** 89.3%
- **Worst-case transformed accuracy:** 82.3%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1200 | 87.2% | 0.946 | 85.8% | 88.5% |
| jpeg90 | 300 | 90.7% | 0.991 | 99.3% | 82.0% |
| jpeg70 | 300 | 89.0% | 0.990 | 98.7% | 79.3% |
| jpeg50 | 300 | 85.7% | 0.990 | 100.0% | 71.3% |
| jpeg30 | 300 | 85.3% | 0.982 | 98.7% | 72.0% |
| blur0.5 | 300 | 93.0% | 0.992 | 97.3% | 88.7% |
| blur1.0 | 300 | 93.0% | 0.980 | 92.7% | 93.3% |
| blur2.0 | 300 | 93.7% | 0.981 | 97.3% | 90.0% |
| resize0.5 | 300 | 93.3% | 0.988 | 96.7% | 90.0% |
| resize0.25 | 300 | 89.7% | 0.972 | 97.3% | 82.0% |
| noise0.02 | 300 | 91.7% | 0.990 | 98.7% | 84.7% |
| noise0.05 | 300 | 86.7% | 0.993 | 99.3% | 74.0% |
| noise0.10 | 300 | 82.3% | 0.984 | 99.3% | 65.3% |
| jitter | 300 | 93.0% | 0.995 | 98.7% | 87.3% |
| crop80 | 300 | 83.7% | 0.982 | 100.0% | 67.3% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **85**
- False negatives (AIGC missed): **69**

Representative worst false positives (real, high AIGC score):
  - `imagenet_00130.npz`  score=1.000
  - `coco_00119.npz`  score=1.000
  - `imagenet_00022.npz`  score=0.998
  - `imagenet_00028.npz`  score=0.997
  - `imagenet_00121.npz`  score=0.996

Representative worst false negatives (AIGC, low score):
  - `DDPM_00086.npz`  score=0.006
  - `VQDM_00113.npz`  score=0.006
  - `VQDM_00107.npz`  score=0.013
  - `VQDM_00116.npz`  score=0.014
  - `VQDM_00017.npz`  score=0.018
