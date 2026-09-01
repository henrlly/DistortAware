# PatchHead vs DID — image-for-image comparison (clean test set)

300 images scored by both detectors (`PatchHead`-only keys: 0, `DID`-only keys: 0).

## Headline metrics

| | PatchHead | DID |
|---|---:|---:|
| clean acc (this comparison) | 1.000 | 0.927 |
| clean AUC | 1.000 | 0.971 |
| mean over 14 transforms | 0.993 | 0.874 |
| worst transform | 0.983 | 0.780 |

## Error agreement

| | DID correct | DID wrong | total |
|---|---:|---:|---:|
| **PatchHead correct** | 278 | 22 | 300 |
| **PatchHead wrong** | 0 | 0 | 0 |
| **total** | 278 | 22 | 300 |

- **PatchHead errors: 0** — of which **0 (0%)** are *also* wrong in DID, **0** are unique to PatchHead.
- **DID errors: 22** — of which **0 (0%)** are *also* wrong in PatchHead, **22** are unique to DID.
- **Both wrong on the same image: 0** (0 false positives / real images called fake, 0 false negatives / fakes called real).
- Union of all errors: 22.  An oracle that picked the better detector per image would score **100.0%** (vs 100.0% for the better single model) — the headroom an ensemble could reach.

## Are the errors correlated?

- phi coefficient between the two 'is-wrong' indicators: **+0.000** (essentially independent). phi≈0 ⇒ the detectors fail on largely *different* images (complementary); phi→1 ⇒ they trip on the *same* hard images.
- Shared errors observed: 0.  If the two error sets were independent you'd expect ≈ 0.0.
- McNemar (do the two disagree asymmetrically?): χ²=20.05, p=7.56e-06 — a significant difference in which detector is more accurate.

## The 0 images both detectors get wrong

These are the genuinely hard cases — a bigger ensemble won't fix them.


## Images only PatchHead gets wrong (0)


## Images only DID gets wrong (22)

- `sid_set/real/sid_real_00009` (real→fake);  PatchHead 0.002 vs DID 0.691
- `sid_set/real/sid_real_00046` (real→fake);  PatchHead 0.004 vs DID 0.864
- `sid_set/real/sid_real_00060` (real→fake);  PatchHead 0.002 vs DID 0.863
- `sid_set/real/sid_real_00078` (real→fake);  PatchHead 0.006 vs DID 0.747
- `sid_set/real/sid_real_00091` (real→fake);  PatchHead 0.002 vs DID 0.992
- `sid_set/real/sid_real_00099` (real→fake);  PatchHead 0.002 vs DID 0.920
- `sid_set/real/sid_real_00116` (real→fake);  PatchHead 0.001 vs DID 0.949
- `sid_set/real/sid_real_00121` (real→fake);  PatchHead 0.011 vs DID 0.761
- `sid_set/real/sid_real_00133` (real→fake);  PatchHead 0.005 vs DID 0.999
- `sid_set/fake/sid_fake_00005` (fake→real);  PatchHead 0.999 vs DID 0.540
- `sid_set/fake/sid_fake_00018` (fake→real);  PatchHead 0.999 vs DID 0.443
- `sid_set/fake/sid_fake_00020` (fake→real);  PatchHead 0.999 vs DID 0.544
- `sid_set/fake/sid_fake_00040` (fake→real);  PatchHead 0.999 vs DID 0.573
- `sid_set/fake/sid_fake_00041` (fake→real);  PatchHead 0.998 vs DID 0.089
- `sid_set/fake/sid_fake_00060` (fake→real);  PatchHead 0.999 vs DID 0.400
- `sid_set/fake/sid_fake_00061` (fake→real);  PatchHead 0.999 vs DID 0.147
- `sid_set/fake/sid_fake_00066` (fake→real);  PatchHead 0.998 vs DID 0.046
- `sid_set/fake/sid_fake_00077` (fake→real);  PatchHead 0.999 vs DID 0.077
- `sid_set/fake/sid_fake_00100` (fake→real);  PatchHead 0.999 vs DID 0.341
- `sid_set/fake/sid_fake_00117` (fake→real);  PatchHead 0.999 vs DID 0.003
- `sid_set/fake/sid_fake_00129` (fake→real);  PatchHead 0.999 vs DID 0.291
- `sid_set/fake/sid_fake_00138` (fake→real);  PatchHead 0.999 vs DID 0.494
