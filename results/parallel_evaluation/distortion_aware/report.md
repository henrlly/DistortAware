# Harness evaluation report

- Data: `/home/v/vincentl/tiktok-aigc-detect/data/harness_large`
- Manifest: `wildfake_benchmark.csv`
- Manifest fingerprint: `b55280a497240a44d5f42f0e76a6363ae18b8a0cfe93b73c26d4de69c5696603`
- Records: **7000**

## Coverage

| Model / transform | Expected | Returned | Missing | Duplicates | Errors |
|---|---:|---:|---:|---:|---:|
| patchhead_distortion_aware:blur1.0 | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_distortion_aware:clean | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_distortion_aware:crop80 | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_distortion_aware:jitter | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_distortion_aware:jpeg90 | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_distortion_aware:noise0.05 | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_distortion_aware:resize0.5 | 1000 | 1000 | 0 | 0 | 0 |

## Model reports

- [patchhead_distortion_aware](models/patchhead_distortion_aware/report.md)
