# Robustness Evaluation Summary

- **Clean accuracy:** 86.8%  (AUC 0.944)
- **Mean accuracy over transformed test sets:** 84.1%
- **Worst-case transformed accuracy:** 76.7%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1500 | 86.8% | 0.944 | 83.9% | 89.7% |
| jpeg90 | 600 | 87.8% | 0.954 | 90.0% | 85.7% |
| jpeg70 | 600 | 86.7% | 0.954 | 92.3% | 81.0% |
| jpeg50 | 600 | 82.7% | 0.938 | 92.0% | 73.3% |
| jpeg30 | 600 | 81.8% | 0.926 | 89.0% | 74.7% |
| blur0.5 | 600 | 88.2% | 0.959 | 85.0% | 91.3% |
| blur1.0 | 600 | 84.7% | 0.934 | 75.3% | 94.0% |
| blur2.0 | 600 | 82.5% | 0.904 | 75.3% | 89.7% |
| resize0.5 | 600 | 85.0% | 0.935 | 78.0% | 92.0% |
| resize0.25 | 600 | 79.2% | 0.885 | 72.3% | 86.0% |
| noise0.02 | 600 | 87.5% | 0.948 | 91.0% | 84.0% |
| noise0.05 | 600 | 81.8% | 0.939 | 91.7% | 72.0% |
| noise0.10 | 600 | 76.7% | 0.894 | 88.0% | 65.3% |
| jitter | 600 | 88.8% | 0.965 | 87.7% | 90.0% |
| crop80 | 600 | 83.8% | 0.921 | 88.3% | 79.3% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **121**
- False negatives (AIGC missed): **77**

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
