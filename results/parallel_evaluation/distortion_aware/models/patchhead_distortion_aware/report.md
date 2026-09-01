# patchhead_distortion_aware

| Transform | N | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| blur1.0 | 1000 | 88.5% | 88.5% | 85.2% | 93.2% | 89.0% | 0.962968 |
| clean | 1000 | 94.9% | 94.9% | 95.5% | 94.2% | 94.9% | 0.990006 |
| crop80 | 1000 | 89.1% | 89.1% | 86.4% | 92.8% | 89.5% | 0.960724 |
| jitter | 1000 | 95.5% | 95.5% | 96.5% | 94.4% | 95.4% | 0.993276 |
| jpeg90 | 1000 | 95.3% | 95.3% | 96.1% | 94.4% | 95.3% | 0.991924 |
| noise0.05 | 1000 | 89.8% | 89.8% | 85.9% | 95.2% | 90.3% | 0.977246 |
| resize0.5 | 1000 | 75.8% | 75.8% | 69.0% | 93.6% | 79.5% | 0.881008 |

## By source

| Transform / source | N | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| blur1.0 / wildfake_coco | 500 | 83.8% | 41.9% | n/a |
| blur1.0 / wildfake_dalle | 500 | 93.2% | 46.6% | n/a |
| clean / wildfake_coco | 500 | 95.6% | 47.8% | n/a |
| clean / wildfake_dalle | 500 | 94.2% | 47.1% | n/a |
| crop80 / wildfake_coco | 500 | 85.4% | 42.7% | n/a |
| crop80 / wildfake_dalle | 500 | 92.8% | 46.4% | n/a |
| jitter / wildfake_coco | 500 | 96.6% | 48.3% | n/a |
| jitter / wildfake_dalle | 500 | 94.4% | 47.2% | n/a |
| jpeg90 / wildfake_coco | 500 | 96.2% | 48.1% | n/a |
| jpeg90 / wildfake_dalle | 500 | 94.4% | 47.2% | n/a |
| noise0.05 / wildfake_coco | 500 | 84.4% | 42.2% | n/a |
| noise0.05 / wildfake_dalle | 500 | 95.2% | 47.6% | n/a |
| resize0.5 / wildfake_coco | 500 | 58.0% | 29.0% | n/a |
| resize0.5 / wildfake_dalle | 500 | 93.6% | 46.8% | n/a |

## By label

| Transform / label | N | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| blur1.0 / 0 | 500 | 83.8% | 41.9% | n/a |
| blur1.0 / 1 | 500 | 93.2% | 46.6% | n/a |
| clean / 0 | 500 | 95.6% | 47.8% | n/a |
| clean / 1 | 500 | 94.2% | 47.1% | n/a |
| crop80 / 0 | 500 | 85.4% | 42.7% | n/a |
| crop80 / 1 | 500 | 92.8% | 46.4% | n/a |
| jitter / 0 | 500 | 96.6% | 48.3% | n/a |
| jitter / 1 | 500 | 94.4% | 47.2% | n/a |
| jpeg90 / 0 | 500 | 96.2% | 48.1% | n/a |
| jpeg90 / 1 | 500 | 94.4% | 47.2% | n/a |
| noise0.05 / 0 | 500 | 84.4% | 42.2% | n/a |
| noise0.05 / 1 | 500 | 95.2% | 47.6% | n/a |
| resize0.5 / 0 | 500 | 58.0% | 29.0% | n/a |
| resize0.5 / 1 | 500 | 93.6% | 46.8% | n/a |
