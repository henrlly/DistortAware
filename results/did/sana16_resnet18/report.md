# Robustness Evaluation Summary

- **Clean accuracy:** 88.2%  (AUC 0.963)
- **Mean accuracy over transformed test sets:** 95.3%
- **Worst-case transformed accuracy:** 87.5%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1200 | 88.2% | 0.963 | 80.5% | 96.0% |
| jpeg90 | 240 | 97.9% | 0.998 | 99.2% | 96.7% |
| jpeg70 | 240 | 95.4% | 1.000 | 100.0% | 90.8% |
| jpeg50 | 240 | 92.1% | 0.997 | 99.2% | 85.0% |
| jpeg30 | 240 | 90.8% | 0.994 | 98.3% | 83.3% |
| blur0.5 | 240 | 99.2% | 0.999 | 99.2% | 99.2% |
| blur1.0 | 240 | 98.3% | 0.999 | 99.2% | 97.5% |
| blur2.0 | 240 | 94.6% | 0.992 | 98.3% | 90.8% |
| resize0.5 | 240 | 98.8% | 1.000 | 99.2% | 98.3% |
| resize0.25 | 240 | 87.5% | 0.992 | 100.0% | 75.0% |
| noise0.02 | 240 | 98.3% | 0.999 | 99.2% | 97.5% |
| noise0.05 | 240 | 95.4% | 0.992 | 98.3% | 92.5% |
| noise0.10 | 240 | 92.1% | 0.988 | 98.3% | 85.8% |
| jitter | 240 | 97.9% | 0.998 | 99.2% | 96.7% |
| crop80 | 240 | 95.4% | 0.994 | 99.2% | 91.7% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **117**
- False negatives (AIGC missed): **24**

Representative worst false positives (real, high AIGC score):
  - `coco_00100.npz`  score=1.000
  - `coco_00025.npz`  score=1.000
  - `imagenet_00121.npz`  score=1.000
  - `imagenet_00143.npz`  score=1.000
  - `imagenet_00141.npz`  score=1.000

Representative worst false negatives (AIGC, low score):
  - `VQDM_00116.npz`  score=0.001
  - `ADM_00136.npz`  score=0.002
  - `VQDM_00051.npz`  score=0.034
  - `VQDM_00113.npz`  score=0.037
  - `VQDM_00146.npz`  score=0.056
