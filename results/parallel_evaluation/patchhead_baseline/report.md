# Harness evaluation report

- Data: `/home/v/vincentl/tiktok-aigc-detect/data/harness_large`
- Manifest: `wildfake_benchmark.csv`
- Manifest fingerprint: `b55280a497240a44d5f42f0e76a6363ae18b8a0cfe93b73c26d4de69c5696603`
- Records: **7000**

## Coverage

| Model / transform | Expected | Returned | Missing | Duplicates | Errors |
|---|---:|---:|---:|---:|---:|
| patchhead_baseline:blur1.0 | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_baseline:clean | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_baseline:crop80 | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_baseline:jitter | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_baseline:jpeg90 | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_baseline:noise0.05 | 1000 | 1000 | 0 | 0 | 0 |
| patchhead_baseline:resize0.5 | 1000 | 1000 | 0 | 0 | 0 |

## Model reports

- [patchhead_baseline](models/patchhead_baseline/report.md)
