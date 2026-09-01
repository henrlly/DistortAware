# Robustness Evaluation Summary

- **Clean accuracy:** 88.3%  (AUC 0.966)
- **Mean accuracy over transformed test sets:** 94.9%
- **Worst-case transformed accuracy:** 87.1%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1200 | 88.3% | 0.967 | 80.0% | 96.7% |
| jpeg90 | 240 | 97.1% | 1.000 | 100.0% | 94.2% |
| jpeg70 | 240 | 94.6% | 0.999 | 100.0% | 89.2% |
| jpeg50 | 240 | 92.1% | 0.997 | 100.0% | 84.2% |
| jpeg30 | 240 | 90.8% | 0.992 | 100.0% | 81.7% |
| blur0.5 | 240 | 99.6% | 0.999 | 99.2% | 100.0% |
| blur1.0 | 240 | 98.8% | 0.999 | 97.5% | 100.0% |
| blur2.0 | 240 | 91.2% | 0.982 | 95.8% | 86.7% |
| resize0.5 | 240 | 97.9% | 0.999 | 99.2% | 96.7% |
| resize0.25 | 240 | 87.1% | 0.989 | 98.3% | 75.8% |
| noise0.02 | 240 | 98.8% | 0.998 | 98.3% | 99.2% |
| noise0.05 | 240 | 96.7% | 0.999 | 100.0% | 93.3% |
| noise0.10 | 240 | 91.2% | 0.985 | 96.7% | 85.8% |
| jitter | 240 | 97.1% | 0.999 | 99.2% | 95.0% |
| crop80 | 240 | 96.2% | 0.997 | 99.2% | 93.3% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **120**
- False negatives (AIGC missed): **20**

Representative worst false positives (real, high AIGC score):
  - `coco_00100.npz`  score=1.000
  - `imagenet_00146.npz`  score=1.000
  - `imagenet_00065.npz`  score=1.000
  - `imagenet_00105.npz`  score=0.999
  - `imagenet_00063.npz`  score=0.999

Representative worst false negatives (AIGC, low score):
  - `VQDM_00033.npz`  score=0.019
  - `ADM_00136.npz`  score=0.023
  - `VQDM_00128.npz`  score=0.044
  - `VQDM_00096.npz`  score=0.049
  - `VQDM_00147.npz`  score=0.273
