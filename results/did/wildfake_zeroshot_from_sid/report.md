# Robustness Evaluation Summary

- **Clean accuracy:** 46.2%  (AUC 0.416)
- **Mean accuracy over transformed test sets:** 46.4%
- **Worst-case transformed accuracy:** 40.0%

| Transform | n | Accuracy | AUC | Real acc | Fake acc |
|---|---:|---:|---:|---:|---:|
| clean | 1200 | 46.2% | 0.416 | 81.5% | 11.0% |
| jpeg90 | 300 | 48.7% | 0.420 | 86.0% | 11.3% |
| jpeg70 | 300 | 45.3% | 0.384 | 80.0% | 10.7% |
| jpeg50 | 300 | 47.0% | 0.447 | 80.0% | 14.0% |
| jpeg30 | 300 | 49.0% | 0.479 | 84.0% | 14.0% |
| blur0.5 | 300 | 48.7% | 0.424 | 89.3% | 8.0% |
| blur1.0 | 300 | 50.0% | 0.469 | 90.0% | 10.0% |
| blur2.0 | 300 | 42.0% | 0.339 | 64.7% | 19.3% |
| resize0.5 | 300 | 53.0% | 0.502 | 91.3% | 14.7% |
| resize0.25 | 300 | 47.3% | 0.450 | 66.0% | 28.7% |
| noise0.02 | 300 | 43.7% | 0.311 | 82.7% | 4.7% |
| noise0.05 | 300 | 40.0% | 0.275 | 70.7% | 9.3% |
| noise0.10 | 300 | 42.7% | 0.318 | 68.7% | 16.7% |
| jitter | 300 | 44.3% | 0.363 | 77.3% | 11.3% |
| crop80 | 300 | 48.0% | 0.468 | 91.3% | 4.7% |

## Error Analysis (clean test set)

- False positives (real flagged as AIGC): **111**
- False negatives (AIGC missed): **534**

Representative worst false positives (real, high AIGC score):
  - `celebahq_00087.npz`  score=0.995
  - `celebahq_00009.npz`  score=0.990
  - `celebahq_00065.npz`  score=0.990
  - `celebahq_00075.npz`  score=0.989
  - `imagenet_00092.npz`  score=0.987

Representative worst false negatives (AIGC, low score):
  - `VQDM_00029.npz`  score=0.000
  - `ADM_00029.npz`  score=0.000
  - `ADM_00060.npz`  score=0.000
  - `ADM_00111.npz`  score=0.000
  - `DDPM_00109.npz`  score=0.000
