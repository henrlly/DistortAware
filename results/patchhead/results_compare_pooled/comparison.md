# PatchHead vs DID — image-for-image comparison (clean test set)

1500 images scored by both detectors (`PatchHead`-only keys: 0, `DID`-only keys: 0).

## Headline metrics

| | PatchHead | DID |
|---|---:|---:|
| clean acc (this comparison) | 0.996 | 0.868 |
| clean AUC | 1.000 | 0.946 |
| mean over 14 transforms | 0.986 | 0.893 |
| worst transform | 0.963 | 0.823 |

## Error agreement

| | DID correct | DID wrong | total |
|---|---:|---:|---:|
| **PatchHead correct** | 1299 | 195 | 1494 |
| **PatchHead wrong** | 3 | 3 | 6 |
| **total** | 1302 | 198 | 1500 |

- **PatchHead errors: 6** — of which **3 (50%)** are *also* wrong in DID, **3** are unique to PatchHead.
- **DID errors: 198** — of which **3 (2%)** are *also* wrong in PatchHead, **195** are unique to DID.
- **Both wrong on the same image: 3** (3 false positives / real images called fake, 0 false negatives / fakes called real).
- Union of all errors: 201.  An oracle that picked the better detector per image would score **99.8%** (vs 99.6% for the better single model) — the headroom an ensemble could reach.

## Are the errors correlated?

- phi coefficient between the two 'is-wrong' indicators: **+0.069** (essentially independent). phi≈0 ⇒ the detectors fail on largely *different* images (complementary); phi→1 ⇒ they trip on the *same* hard images.
- Shared errors observed: 3.  If the two error sets were independent you'd expect ≈ 0.8.
- McNemar (do the two disagree asymmetrically?): χ²=184.25, p=5.73e-42 — a significant difference in which detector is more accurate.

## The 3 images both detectors get wrong

These are the genuinely hard cases — a bigger ensemble won't fix them.

- `wildfake/real/coco_00100` (real→fake);  PatchHead score 0.956, DID score 0.989
- `wildfake/real/coco_00119` (real→fake);  PatchHead score 0.999, DID score 1.000
- `wildfake/real/imagenet_00038` (real→fake);  PatchHead score 0.999, DID score 0.994

## Images only PatchHead gets wrong (3)

- `wildfake/real/imagenet_00047` (real→fake);  PatchHead 0.991 vs DID 0.001
- `wildfake/real/imagenet_00081` (real→fake);  PatchHead 0.859 vs DID 0.379
- `sid_set/fake/sid_fake_00070` (fake→real);  PatchHead 0.737 vs DID 0.999

## Images only DID gets wrong (195)

- `sid_set/real/sid_real_00003` (real→fake);  PatchHead 0.001 vs DID 0.606
- `sid_set/real/sid_real_00006` (real→fake);  PatchHead 0.002 vs DID 0.919
- `sid_set/real/sid_real_00007` (real→fake);  PatchHead 0.002 vs DID 0.967
- `sid_set/real/sid_real_00009` (real→fake);  PatchHead 0.001 vs DID 0.967
- `sid_set/real/sid_real_00013` (real→fake);  PatchHead 0.009 vs DID 0.545
- `sid_set/real/sid_real_00015` (real→fake);  PatchHead 0.000 vs DID 0.760
- `sid_set/real/sid_real_00018` (real→fake);  PatchHead 0.001 vs DID 0.859
- `sid_set/real/sid_real_00027` (real→fake);  PatchHead 0.000 vs DID 0.655
- `sid_set/real/sid_real_00030` (real→fake);  PatchHead 0.061 vs DID 0.973
- `sid_set/real/sid_real_00031` (real→fake);  PatchHead 0.001 vs DID 0.719
- `sid_set/real/sid_real_00036` (real→fake);  PatchHead 0.109 vs DID 0.991
- `sid_set/real/sid_real_00040` (real→fake);  PatchHead 0.001 vs DID 0.530
- `sid_set/real/sid_real_00045` (real→fake);  PatchHead 0.001 vs DID 0.606
- `sid_set/real/sid_real_00046` (real→fake);  PatchHead 0.050 vs DID 0.673
- `sid_set/real/sid_real_00056` (real→fake);  PatchHead 0.003 vs DID 0.623
- `sid_set/real/sid_real_00060` (real→fake);  PatchHead 0.000 vs DID 0.851
- `sid_set/real/sid_real_00062` (real→fake);  PatchHead 0.001 vs DID 0.587
- `sid_set/real/sid_real_00063` (real→fake);  PatchHead 0.004 vs DID 0.893
- `sid_set/real/sid_real_00067` (real→fake);  PatchHead 0.015 vs DID 0.674
- `sid_set/real/sid_real_00073` (real→fake);  PatchHead 0.000 vs DID 0.671
- `sid_set/real/sid_real_00075` (real→fake);  PatchHead 0.007 vs DID 0.572
- `sid_set/real/sid_real_00078` (real→fake);  PatchHead 0.001 vs DID 0.923
- `sid_set/real/sid_real_00088` (real→fake);  PatchHead 0.061 vs DID 0.672
- `sid_set/real/sid_real_00091` (real→fake);  PatchHead 0.001 vs DID 0.782
- `sid_set/real/sid_real_00099` (real→fake);  PatchHead 0.001 vs DID 0.925
- `sid_set/real/sid_real_00102` (real→fake);  PatchHead 0.028 vs DID 0.850
- `sid_set/real/sid_real_00108` (real→fake);  PatchHead 0.003 vs DID 0.910
- `sid_set/real/sid_real_00109` (real→fake);  PatchHead 0.007 vs DID 0.638
- `sid_set/real/sid_real_00116` (real→fake);  PatchHead 0.001 vs DID 0.948
- `sid_set/real/sid_real_00118` (real→fake);  PatchHead 0.001 vs DID 0.745
- `sid_set/real/sid_real_00119` (real→fake);  PatchHead 0.001 vs DID 0.785
- `sid_set/real/sid_real_00121` (real→fake);  PatchHead 0.026 vs DID 0.864
- `sid_set/real/sid_real_00125` (real→fake);  PatchHead 0.001 vs DID 0.598
- `sid_set/real/sid_real_00133` (real→fake);  PatchHead 0.009 vs DID 0.846
- `sid_set/real/sid_real_00136` (real→fake);  PatchHead 0.001 vs DID 0.832
- `sid_set/real/sid_real_00146` (real→fake);  PatchHead 0.001 vs DID 0.994
- `wildfake/real/afhq_00026` (real→fake);  PatchHead 0.002 vs DID 0.583
- `wildfake/real/afhq_00099` (real→fake);  PatchHead 0.000 vs DID 0.991
- `wildfake/real/celebahq_00000` (real→fake);  PatchHead 0.001 vs DID 0.579
- `wildfake/real/coco_00000` (real→fake);  PatchHead 0.005 vs DID 0.554
