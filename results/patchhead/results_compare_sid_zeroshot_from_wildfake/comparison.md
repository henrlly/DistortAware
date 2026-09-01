# PatchHead(WF->SID) vs DID(SID-native) — image-for-image comparison (clean test set)

300 images scored by both detectors (`PatchHead(WF->SID)`-only keys: 0, `DID(SID-native)`-only keys: 0).

## Error agreement

| | DID(SID-native) correct | DID(SID-native) wrong | total |
|---|---:|---:|---:|
| **PatchHead(WF->SID) correct** | 133 | 9 | 142 |
| **PatchHead(WF->SID) wrong** | 145 | 13 | 158 |
| **total** | 278 | 22 | 300 |

- **PatchHead(WF->SID) errors: 158** — of which **13 (8%)** are *also* wrong in DID(SID-native), **145** are unique to PatchHead(WF->SID).
- **DID(SID-native) errors: 22** — of which **13 (59%)** are *also* wrong in PatchHead(WF->SID), **9** are unique to DID(SID-native).
- **Both wrong on the same image: 13** (0 false positives / real images called fake, 13 false negatives / fakes called real).
- Union of all errors: 167.  An oracle that picked the better detector per image would score **95.7%** (vs 92.7% for the better single model) — the headroom an ensemble could reach.

## Are the errors correlated?

- phi coefficient between the two 'is-wrong' indicators: **+0.036** (essentially independent). phi≈0 ⇒ the detectors fail on largely *different* images (complementary); phi→1 ⇒ they trip on the *same* hard images.
- Shared errors observed: 13.  If the two error sets were independent you'd expect ≈ 11.6.
- McNemar (do the two disagree asymmetrically?): χ²=118.34, p=1.46e-27 — a significant difference in which detector is more accurate.

## The 13 images both detectors get wrong

These are the genuinely hard cases — a bigger ensemble won't fix them.

- `sid_set/fake/sid_fake_00005` (fake→real);  PatchHead(WF->SID) score 0.000, DID(SID-native) score 0.540
- `sid_set/fake/sid_fake_00018` (fake→real);  PatchHead(WF->SID) score 0.001, DID(SID-native) score 0.443
- `sid_set/fake/sid_fake_00020` (fake→real);  PatchHead(WF->SID) score 0.000, DID(SID-native) score 0.544
- `sid_set/fake/sid_fake_00040` (fake→real);  PatchHead(WF->SID) score 0.001, DID(SID-native) score 0.573
- `sid_set/fake/sid_fake_00041` (fake→real);  PatchHead(WF->SID) score 0.000, DID(SID-native) score 0.089
- `sid_set/fake/sid_fake_00060` (fake→real);  PatchHead(WF->SID) score 0.076, DID(SID-native) score 0.400
- `sid_set/fake/sid_fake_00061` (fake→real);  PatchHead(WF->SID) score 0.000, DID(SID-native) score 0.147
- `sid_set/fake/sid_fake_00066` (fake→real);  PatchHead(WF->SID) score 0.065, DID(SID-native) score 0.046
- `sid_set/fake/sid_fake_00077` (fake→real);  PatchHead(WF->SID) score 0.083, DID(SID-native) score 0.077
- `sid_set/fake/sid_fake_00100` (fake→real);  PatchHead(WF->SID) score 0.001, DID(SID-native) score 0.341
- `sid_set/fake/sid_fake_00117` (fake→real);  PatchHead(WF->SID) score 0.000, DID(SID-native) score 0.003
- `sid_set/fake/sid_fake_00129` (fake→real);  PatchHead(WF->SID) score 0.089, DID(SID-native) score 0.291
- `sid_set/fake/sid_fake_00138` (fake→real);  PatchHead(WF->SID) score 0.001, DID(SID-native) score 0.494

## Images only PatchHead(WF->SID) gets wrong (145)

- `sid_set/real/sid_real_00016` (real→fake);  PatchHead(WF->SID) 0.998 vs DID(SID-native) 0.000
- `sid_set/real/sid_real_00019` (real→fake);  PatchHead(WF->SID) 1.000 vs DID(SID-native) 0.030
- `sid_set/real/sid_real_00022` (real→fake);  PatchHead(WF->SID) 0.827 vs DID(SID-native) 0.173
- `sid_set/real/sid_real_00030` (real→fake);  PatchHead(WF->SID) 0.995 vs DID(SID-native) 0.221
- `sid_set/real/sid_real_00036` (real→fake);  PatchHead(WF->SID) 0.934 vs DID(SID-native) 0.355
- `sid_set/real/sid_real_00044` (real→fake);  PatchHead(WF->SID) 0.980 vs DID(SID-native) 0.008
- `sid_set/real/sid_real_00048` (real→fake);  PatchHead(WF->SID) 0.999 vs DID(SID-native) 0.000
- `sid_set/real/sid_real_00061` (real→fake);  PatchHead(WF->SID) 0.996 vs DID(SID-native) 0.010
- `sid_set/real/sid_real_00074` (real→fake);  PatchHead(WF->SID) 0.990 vs DID(SID-native) 0.002
- `sid_set/real/sid_real_00093` (real→fake);  PatchHead(WF->SID) 0.997 vs DID(SID-native) 0.000
- `sid_set/real/sid_real_00108` (real→fake);  PatchHead(WF->SID) 0.993 vs DID(SID-native) 0.000
- `sid_set/real/sid_real_00134` (real→fake);  PatchHead(WF->SID) 0.997 vs DID(SID-native) 0.003
- `sid_set/real/sid_real_00135` (real→fake);  PatchHead(WF->SID) 0.787 vs DID(SID-native) 0.071
- `sid_set/fake/sid_fake_00000` (fake→real);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 0.991
- `sid_set/fake/sid_fake_00001` (fake→real);  PatchHead(WF->SID) 0.005 vs DID(SID-native) 1.000
- `sid_set/fake/sid_fake_00002` (fake→real);  PatchHead(WF->SID) 0.393 vs DID(SID-native) 0.968
- `sid_set/fake/sid_fake_00003` (fake→real);  PatchHead(WF->SID) 0.039 vs DID(SID-native) 0.836
- `sid_set/fake/sid_fake_00006` (fake→real);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 1.000
- `sid_set/fake/sid_fake_00007` (fake→real);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 0.977
- `sid_set/fake/sid_fake_00008` (fake→real);  PatchHead(WF->SID) 0.002 vs DID(SID-native) 0.774
- `sid_set/fake/sid_fake_00009` (fake→real);  PatchHead(WF->SID) 0.006 vs DID(SID-native) 1.000
- `sid_set/fake/sid_fake_00010` (fake→real);  PatchHead(WF->SID) 0.043 vs DID(SID-native) 0.998
- `sid_set/fake/sid_fake_00011` (fake→real);  PatchHead(WF->SID) 0.016 vs DID(SID-native) 0.961
- `sid_set/fake/sid_fake_00012` (fake→real);  PatchHead(WF->SID) 0.000 vs DID(SID-native) 0.972
- `sid_set/fake/sid_fake_00014` (fake→real);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 1.000
- `sid_set/fake/sid_fake_00015` (fake→real);  PatchHead(WF->SID) 0.000 vs DID(SID-native) 0.997
- `sid_set/fake/sid_fake_00016` (fake→real);  PatchHead(WF->SID) 0.003 vs DID(SID-native) 0.917
- `sid_set/fake/sid_fake_00017` (fake→real);  PatchHead(WF->SID) 0.002 vs DID(SID-native) 0.628
- `sid_set/fake/sid_fake_00019` (fake→real);  PatchHead(WF->SID) 0.005 vs DID(SID-native) 0.995
- `sid_set/fake/sid_fake_00021` (fake→real);  PatchHead(WF->SID) 0.002 vs DID(SID-native) 0.952
- `sid_set/fake/sid_fake_00022` (fake→real);  PatchHead(WF->SID) 0.011 vs DID(SID-native) 0.926
- `sid_set/fake/sid_fake_00023` (fake→real);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 0.990
- `sid_set/fake/sid_fake_00024` (fake→real);  PatchHead(WF->SID) 0.028 vs DID(SID-native) 0.994
- `sid_set/fake/sid_fake_00025` (fake→real);  PatchHead(WF->SID) 0.000 vs DID(SID-native) 0.997
- `sid_set/fake/sid_fake_00026` (fake→real);  PatchHead(WF->SID) 0.000 vs DID(SID-native) 0.996
- `sid_set/fake/sid_fake_00027` (fake→real);  PatchHead(WF->SID) 0.007 vs DID(SID-native) 0.989
- `sid_set/fake/sid_fake_00028` (fake→real);  PatchHead(WF->SID) 0.003 vs DID(SID-native) 0.999
- `sid_set/fake/sid_fake_00029` (fake→real);  PatchHead(WF->SID) 0.008 vs DID(SID-native) 0.999
- `sid_set/fake/sid_fake_00030` (fake→real);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 0.998
- `sid_set/fake/sid_fake_00031` (fake→real);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 1.000

## Images only DID(SID-native) gets wrong (9)

- `sid_set/real/sid_real_00009` (real→fake);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 0.691
- `sid_set/real/sid_real_00046` (real→fake);  PatchHead(WF->SID) 0.066 vs DID(SID-native) 0.864
- `sid_set/real/sid_real_00060` (real→fake);  PatchHead(WF->SID) 0.000 vs DID(SID-native) 0.863
- `sid_set/real/sid_real_00078` (real→fake);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 0.747
- `sid_set/real/sid_real_00091` (real→fake);  PatchHead(WF->SID) 0.003 vs DID(SID-native) 0.992
- `sid_set/real/sid_real_00099` (real→fake);  PatchHead(WF->SID) 0.001 vs DID(SID-native) 0.920
- `sid_set/real/sid_real_00116` (real→fake);  PatchHead(WF->SID) 0.000 vs DID(SID-native) 0.949
- `sid_set/real/sid_real_00121` (real→fake);  PatchHead(WF->SID) 0.000 vs DID(SID-native) 0.761
- `sid_set/real/sid_real_00133` (real→fake);  PatchHead(WF->SID) 0.000 vs DID(SID-native) 0.999
