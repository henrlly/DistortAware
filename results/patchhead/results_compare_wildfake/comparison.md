# PatchHead vs DID — image-for-image comparison (clean test set)

1200 images scored by both detectors (`PatchHead`-only keys: 0, `DID`-only keys: 0).

## Headline metrics

| | PatchHead | DID |
|---|---:|---:|
| clean acc (this comparison) | 0.985 | 0.886 |
| clean AUC | 1.000 | 0.957 |
| mean over 14 transforms | 0.990 | 0.930 |
| worst transform | 0.970 | 0.880 |

## Error agreement

| | DID correct | DID wrong | total |
|---|---:|---:|---:|
| **PatchHead correct** | 1058 | 124 | 1182 |
| **PatchHead wrong** | 5 | 13 | 18 |
| **total** | 1063 | 137 | 1200 |

- **PatchHead errors: 18** — of which **13 (72%)** are *also* wrong in DID, **5** are unique to PatchHead.
- **DID errors: 137** — of which **13 (9%)** are *also* wrong in PatchHead, **124** are unique to DID.
- **Both wrong on the same image: 13** (13 false positives / real images called fake, 0 false negatives / fakes called real).
- Union of all errors: 142.  An oracle that picked the better detector per image would score **98.9%** (vs 98.5% for the better single model) — the headroom an ensemble could reach.

## Are the errors correlated?

- phi coefficient between the two 'is-wrong' indicators: **+0.236** (weakly correlated). phi≈0 ⇒ the detectors fail on largely *different* images (complementary); phi→1 ⇒ they trip on the *same* hard images.
- Shared errors observed: 13.  If the two error sets were independent you'd expect ≈ 2.1.
- McNemar (do the two disagree asymmetrically?): χ²=107.94, p=2.77e-25 — a significant difference in which detector is more accurate.

## The 13 images both detectors get wrong

These are the genuinely hard cases — a bigger ensemble won't fix them.

- `wildfake/real/coco_00119` (real→fake);  PatchHead score 1.000, DID score 0.996
- `wildfake/real/coco_00121` (real→fake);  PatchHead score 0.957, DID score 0.957
- `wildfake/real/imagenet_00037` (real→fake);  PatchHead score 0.998, DID score 0.706
- `wildfake/real/imagenet_00038` (real→fake);  PatchHead score 1.000, DID score 1.000
- `wildfake/real/imagenet_00045` (real→fake);  PatchHead score 0.969, DID score 0.999
- `wildfake/real/imagenet_00046` (real→fake);  PatchHead score 0.837, DID score 0.816
- `wildfake/real/imagenet_00055` (real→fake);  PatchHead score 0.994, DID score 0.828
- `wildfake/real/imagenet_00063` (real→fake);  PatchHead score 0.931, DID score 0.802
- `wildfake/real/imagenet_00073` (real→fake);  PatchHead score 0.830, DID score 0.948
- `wildfake/real/imagenet_00075` (real→fake);  PatchHead score 0.809, DID score 0.888
- `wildfake/real/imagenet_00105` (real→fake);  PatchHead score 0.869, DID score 0.788
- `wildfake/real/imagenet_00123` (real→fake);  PatchHead score 0.826, DID score 0.999
- `wildfake/real/imagenet_00133` (real→fake);  PatchHead score 0.935, DID score 0.998

## Images only PatchHead gets wrong (5)

- `wildfake/real/coco_00129` (real→fake);  PatchHead 0.860 vs DID 0.059
- `wildfake/real/imagenet_00047` (real→fake);  PatchHead 0.968 vs DID 0.016
- `wildfake/real/imagenet_00058` (real→fake);  PatchHead 0.974 vs DID 0.006
- `wildfake/real/imagenet_00084` (real→fake);  PatchHead 0.838 vs DID 0.379
- `wildfake/real/imagenet_00108` (real→fake);  PatchHead 0.887 vs DID 0.477

## Images only DID gets wrong (124)

- `wildfake/real/afhq_00099` (real→fake);  PatchHead 0.000 vs DID 0.804
- `wildfake/real/coco_00004` (real→fake);  PatchHead 0.000 vs DID 0.910
- `wildfake/real/coco_00025` (real→fake);  PatchHead 0.000 vs DID 0.825
- `wildfake/real/coco_00031` (real→fake);  PatchHead 0.000 vs DID 0.923
- `wildfake/real/coco_00038` (real→fake);  PatchHead 0.000 vs DID 0.993
- `wildfake/real/coco_00043` (real→fake);  PatchHead 0.000 vs DID 0.732
- `wildfake/real/coco_00047` (real→fake);  PatchHead 0.023 vs DID 0.994
- `wildfake/real/coco_00051` (real→fake);  PatchHead 0.002 vs DID 0.876
- `wildfake/real/coco_00057` (real→fake);  PatchHead 0.002 vs DID 0.779
- `wildfake/real/coco_00060` (real→fake);  PatchHead 0.000 vs DID 0.926
- `wildfake/real/coco_00068` (real→fake);  PatchHead 0.715 vs DID 0.945
- `wildfake/real/coco_00075` (real→fake);  PatchHead 0.000 vs DID 0.793
- `wildfake/real/coco_00078` (real→fake);  PatchHead 0.000 vs DID 0.979
- `wildfake/real/coco_00086` (real→fake);  PatchHead 0.001 vs DID 0.937
- `wildfake/real/coco_00088` (real→fake);  PatchHead 0.105 vs DID 0.861
- `wildfake/real/coco_00090` (real→fake);  PatchHead 0.000 vs DID 0.973
- `wildfake/real/coco_00091` (real→fake);  PatchHead 0.000 vs DID 0.984
- `wildfake/real/coco_00092` (real→fake);  PatchHead 0.016 vs DID 0.733
- `wildfake/real/coco_00100` (real→fake);  PatchHead 0.005 vs DID 0.969
- `wildfake/real/coco_00105` (real→fake);  PatchHead 0.000 vs DID 0.926
- `wildfake/real/coco_00115` (real→fake);  PatchHead 0.155 vs DID 0.918
- `wildfake/real/coco_00118` (real→fake);  PatchHead 0.000 vs DID 0.768
- `wildfake/real/coco_00135` (real→fake);  PatchHead 0.001 vs DID 0.903
- `wildfake/real/coco_00137` (real→fake);  PatchHead 0.033 vs DID 0.937
- `wildfake/real/imagenet_00002` (real→fake);  PatchHead 0.002 vs DID 0.852
- `wildfake/real/imagenet_00006` (real→fake);  PatchHead 0.003 vs DID 0.730
- `wildfake/real/imagenet_00007` (real→fake);  PatchHead 0.000 vs DID 0.789
- `wildfake/real/imagenet_00012` (real→fake);  PatchHead 0.001 vs DID 0.983
- `wildfake/real/imagenet_00018` (real→fake);  PatchHead 0.000 vs DID 0.737
- `wildfake/real/imagenet_00019` (real→fake);  PatchHead 0.000 vs DID 0.979
- `wildfake/real/imagenet_00022` (real→fake);  PatchHead 0.001 vs DID 1.000
- `wildfake/real/imagenet_00025` (real→fake);  PatchHead 0.003 vs DID 0.835
- `wildfake/real/imagenet_00028` (real→fake);  PatchHead 0.001 vs DID 0.805
- `wildfake/real/imagenet_00048` (real→fake);  PatchHead 0.002 vs DID 0.749
- `wildfake/real/imagenet_00050` (real→fake);  PatchHead 0.058 vs DID 0.922
- `wildfake/real/imagenet_00056` (real→fake);  PatchHead 0.001 vs DID 0.832
- `wildfake/real/imagenet_00061` (real→fake);  PatchHead 0.391 vs DID 0.734
- `wildfake/real/imagenet_00065` (real→fake);  PatchHead 0.002 vs DID 0.996
- `wildfake/real/imagenet_00068` (real→fake);  PatchHead 0.011 vs DID 0.884
- `wildfake/real/imagenet_00070` (real→fake);  PatchHead 0.006 vs DID 0.905
