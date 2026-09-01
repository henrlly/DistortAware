# patchhead_baseline

| Transform | N | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| blur1.0 | 1000 | 82.4% | 82.4% | 76.9% | 92.6% | 84.0% | 0.923058 |
| clean | 1000 | 88.2% | 88.2% | 85.8% | 91.6% | 88.6% | 0.954582 |
| crop80 | 1000 | 82.1% | 82.1% | 77.5% | 90.4% | 83.5% | 0.913482 |
| jitter | 1000 | 88.2% | 88.2% | 85.9% | 91.4% | 88.6% | 0.955316 |
| jpeg90 | 1000 | 90.3% | 90.3% | 88.5% | 92.6% | 90.5% | 0.964646 |
| noise0.05 | 1000 | 85.9% | 85.9% | 82.2% | 91.6% | 86.7% | 0.942212 |
| resize0.5 | 1000 | 80.5% | 80.5% | 74.9% | 91.8% | 82.5% | 0.907294 |

## By source

| Transform / source | N | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| blur1.0 / wildfake_coco | 500 | 72.2% | 36.1% | n/a |
| blur1.0 / wildfake_dalle | 500 | 92.6% | 46.3% | n/a |
| clean / wildfake_coco | 500 | 84.8% | 42.4% | n/a |
| clean / wildfake_dalle | 500 | 91.6% | 45.8% | n/a |
| crop80 / wildfake_coco | 500 | 73.8% | 36.9% | n/a |
| crop80 / wildfake_dalle | 500 | 90.4% | 45.2% | n/a |
| jitter / wildfake_coco | 500 | 85.0% | 42.5% | n/a |
| jitter / wildfake_dalle | 500 | 91.4% | 45.7% | n/a |
| jpeg90 / wildfake_coco | 500 | 88.0% | 44.0% | n/a |
| jpeg90 / wildfake_dalle | 500 | 92.6% | 46.3% | n/a |
| noise0.05 / wildfake_coco | 500 | 80.2% | 40.1% | n/a |
| noise0.05 / wildfake_dalle | 500 | 91.6% | 45.8% | n/a |
| resize0.5 / wildfake_coco | 500 | 69.2% | 34.6% | n/a |
| resize0.5 / wildfake_dalle | 500 | 91.8% | 45.9% | n/a |

## By label

| Transform / label | N | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| blur1.0 / 0 | 500 | 72.2% | 36.1% | n/a |
| blur1.0 / 1 | 500 | 92.6% | 46.3% | n/a |
| clean / 0 | 500 | 84.8% | 42.4% | n/a |
| clean / 1 | 500 | 91.6% | 45.8% | n/a |
| crop80 / 0 | 500 | 73.8% | 36.9% | n/a |
| crop80 / 1 | 500 | 90.4% | 45.2% | n/a |
| jitter / 0 | 500 | 85.0% | 42.5% | n/a |
| jitter / 1 | 500 | 91.4% | 45.7% | n/a |
| jpeg90 / 0 | 500 | 88.0% | 44.0% | n/a |
| jpeg90 / 1 | 500 | 92.6% | 46.3% | n/a |
| noise0.05 / 0 | 500 | 80.2% | 40.1% | n/a |
| noise0.05 / 1 | 500 | 91.6% | 45.8% | n/a |
| resize0.5 / 0 | 500 | 69.2% | 34.6% | n/a |
| resize0.5 / 1 | 500 | 91.8% | 45.9% | n/a |
