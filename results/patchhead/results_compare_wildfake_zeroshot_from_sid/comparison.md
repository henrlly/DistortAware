# PatchHead(SID->WF) vs DID(WF-native) — image-for-image comparison (clean test set)

1200 images scored by both detectors (`PatchHead(SID->WF)`-only keys: 0, `DID(WF-native)`-only keys: 0).

## Error agreement

| | DID(WF-native) correct | DID(WF-native) wrong | total |
|---|---:|---:|---:|
| **PatchHead(SID->WF) correct** | 541 | 80 | 621 |
| **PatchHead(SID->WF) wrong** | 522 | 57 | 579 |
| **total** | 1063 | 137 | 1200 |

- **PatchHead(SID->WF) errors: 579** — of which **57 (10%)** are *also* wrong in DID(WF-native), **522** are unique to PatchHead(SID->WF).
- **DID(WF-native) errors: 137** — of which **57 (42%)** are *also* wrong in PatchHead(SID->WF), **80** are unique to DID(WF-native).
- **Both wrong on the same image: 57** (6 false positives / real images called fake, 51 false negatives / fakes called real).
- Union of all errors: 659.  An oracle that picked the better detector per image would score **95.2%** (vs 88.6% for the better single model) — the headroom an ensemble could reach.

## Are the errors correlated?

- phi coefficient between the two 'is-wrong' indicators: **-0.048** (essentially independent). phi≈0 ⇒ the detectors fail on largely *different* images (complementary); phi→1 ⇒ they trip on the *same* hard images.
- Shared errors observed: 57.  If the two error sets were independent you'd expect ≈ 66.1.
- McNemar (do the two disagree asymmetrically?): χ²=323.06, p=3.12e-72 — a significant difference in which detector is more accurate.

## The 57 images both detectors get wrong

These are the genuinely hard cases — a bigger ensemble won't fix them.

- `wildfake/real/coco_00100` (real→fake);  PatchHead(SID->WF) score 0.978, DID(WF-native) score 0.969
- `wildfake/real/coco_00119` (real→fake);  PatchHead(SID->WF) score 0.674, DID(WF-native) score 0.996
- `wildfake/real/imagenet_00012` (real→fake);  PatchHead(SID->WF) score 0.903, DID(WF-native) score 0.983
- `wildfake/real/imagenet_00038` (real→fake);  PatchHead(SID->WF) score 0.986, DID(WF-native) score 1.000
- `wildfake/real/imagenet_00135` (real→fake);  PatchHead(SID->WF) score 0.559, DID(WF-native) score 0.998
- `wildfake/real/imagenet_00147` (real→fake);  PatchHead(SID->WF) score 0.597, DID(WF-native) score 0.999
- `wildfake/fake/ADM_00003` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.612
- `wildfake/fake/ADM_00043` (fake→real);  PatchHead(SID->WF) score 0.006, DID(WF-native) score 0.660
- `wildfake/fake/ADM_00071` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.018
- `wildfake/fake/ADM_00074` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.629
- `wildfake/fake/ADM_00078` (fake→real);  PatchHead(SID->WF) score 0.003, DID(WF-native) score 0.621
- `wildfake/fake/ADM_00087` (fake→real);  PatchHead(SID->WF) score 0.008, DID(WF-native) score 0.535
- `wildfake/fake/ADM_00105` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.465
- `wildfake/fake/ADM_00124` (fake→real);  PatchHead(SID->WF) score 0.003, DID(WF-native) score 0.028
- `wildfake/fake/ADM_00136` (fake→real);  PatchHead(SID->WF) score 0.006, DID(WF-native) score 0.395
- `wildfake/fake/ADM_00137` (fake→real);  PatchHead(SID->WF) score 0.005, DID(WF-native) score 0.650
- `wildfake/fake/DDIM_00023` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.564
- `wildfake/fake/DDIM_00078` (fake→real);  PatchHead(SID->WF) score 0.004, DID(WF-native) score 0.656
- `wildfake/fake/DDIM_00084` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.483
- `wildfake/fake/DDIM_00107` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.638
- `wildfake/fake/DDIM_00116` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.317
- `wildfake/fake/DDPM_00019` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.622
- `wildfake/fake/DDPM_00063` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.625
- `wildfake/fake/DDPM_00085` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.055
- `wildfake/fake/DDPM_00086` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.181
- `wildfake/fake/DDPM_00103` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.093
- `wildfake/fake/DDPM_00111` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.224
- `wildfake/fake/DDPM_00133` (fake→real);  PatchHead(SID->WF) score 0.004, DID(WF-native) score 0.213
- `wildfake/fake/DDPM_00134` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.511
- `wildfake/fake/VQDM_00008` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.201
- `wildfake/fake/VQDM_00009` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.489
- `wildfake/fake/VQDM_00011` (fake→real);  PatchHead(SID->WF) score 0.004, DID(WF-native) score 0.285
- `wildfake/fake/VQDM_00014` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.106
- `wildfake/fake/VQDM_00016` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.653
- `wildfake/fake/VQDM_00017` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.102
- `wildfake/fake/VQDM_00028` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.648
- `wildfake/fake/VQDM_00029` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.498
- `wildfake/fake/VQDM_00051` (fake→real);  PatchHead(SID->WF) score 0.042, DID(WF-native) score 0.139
- `wildfake/fake/VQDM_00061` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.046
- `wildfake/fake/VQDM_00071` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.030
- `wildfake/fake/VQDM_00072` (fake→real);  PatchHead(SID->WF) score 0.004, DID(WF-native) score 0.658
- `wildfake/fake/VQDM_00075` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.650
- `wildfake/fake/VQDM_00078` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.639
- `wildfake/fake/VQDM_00084` (fake→real);  PatchHead(SID->WF) score 0.405, DID(WF-native) score 0.346
- `wildfake/fake/VQDM_00086` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.034
- `wildfake/fake/VQDM_00089` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.008
- `wildfake/fake/VQDM_00091` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.437
- `wildfake/fake/VQDM_00095` (fake→real);  PatchHead(SID->WF) score 0.004, DID(WF-native) score 0.232
- `wildfake/fake/VQDM_00107` (fake→real);  PatchHead(SID->WF) score 0.003, DID(WF-native) score 0.531
- `wildfake/fake/VQDM_00109` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.296
- `wildfake/fake/VQDM_00111` (fake→real);  PatchHead(SID->WF) score 0.013, DID(WF-native) score 0.220
- `wildfake/fake/VQDM_00113` (fake→real);  PatchHead(SID->WF) score 0.005, DID(WF-native) score 0.165
- `wildfake/fake/VQDM_00116` (fake→real);  PatchHead(SID->WF) score 0.002, DID(WF-native) score 0.010
- `wildfake/fake/VQDM_00128` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.635
- `wildfake/fake/VQDM_00138` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.062
- `wildfake/fake/VQDM_00141` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.374
- `wildfake/fake/VQDM_00146` (fake→real);  PatchHead(SID->WF) score 0.001, DID(WF-native) score 0.010

## Images only PatchHead(SID->WF) gets wrong (522)

- `wildfake/real/afhq_00028` (real→fake);  PatchHead(SID->WF) 0.607 vs DID(WF-native) 0.001
- `wildfake/real/celebahq_00058` (real→fake);  PatchHead(SID->WF) 0.675 vs DID(WF-native) 0.000
- `wildfake/real/celebahq_00092` (real→fake);  PatchHead(SID->WF) 0.951 vs DID(WF-native) 0.276
- `wildfake/real/celebahq_00103` (real→fake);  PatchHead(SID->WF) 0.960 vs DID(WF-native) 0.002
- `wildfake/fake/ADM_00000` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.999
- `wildfake/fake/ADM_00001` (fake→real);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00002` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.891
- `wildfake/fake/ADM_00004` (fake→real);  PatchHead(SID->WF) 0.010 vs DID(WF-native) 0.986
- `wildfake/fake/ADM_00005` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.992
- `wildfake/fake/ADM_00006` (fake→real);  PatchHead(SID->WF) 0.004 vs DID(WF-native) 0.998
- `wildfake/fake/ADM_00007` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00008` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00009` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00010` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.998
- `wildfake/fake/ADM_00011` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.967
- `wildfake/fake/ADM_00012` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00013` (fake→real);  PatchHead(SID->WF) 0.005 vs DID(WF-native) 0.980
- `wildfake/fake/ADM_00014` (fake→real);  PatchHead(SID->WF) 0.005 vs DID(WF-native) 0.997
- `wildfake/fake/ADM_00015` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.972
- `wildfake/fake/ADM_00016` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.999
- `wildfake/fake/ADM_00017` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.996
- `wildfake/fake/ADM_00018` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.988
- `wildfake/fake/ADM_00019` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.999
- `wildfake/fake/ADM_00020` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00021` (fake→real);  PatchHead(SID->WF) 0.004 vs DID(WF-native) 0.996
- `wildfake/fake/ADM_00022` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.920
- `wildfake/fake/ADM_00023` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00024` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.994
- `wildfake/fake/ADM_00025` (fake→real);  PatchHead(SID->WF) 0.009 vs DID(WF-native) 0.841
- `wildfake/fake/ADM_00026` (fake→real);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 0.999
- `wildfake/fake/ADM_00027` (fake→real);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00028` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.977
- `wildfake/fake/ADM_00029` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.963
- `wildfake/fake/ADM_00030` (fake→real);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00031` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00032` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.681
- `wildfake/fake/ADM_00033` (fake→real);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 0.997
- `wildfake/fake/ADM_00034` (fake→real);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 1.000
- `wildfake/fake/ADM_00035` (fake→real);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.996
- `wildfake/fake/ADM_00036` (fake→real);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 1.000

## Images only DID(WF-native) gets wrong (80)

- `wildfake/real/afhq_00099` (real→fake);  PatchHead(SID->WF) 0.017 vs DID(WF-native) 0.804
- `wildfake/real/coco_00004` (real→fake);  PatchHead(SID->WF) 0.005 vs DID(WF-native) 0.910
- `wildfake/real/coco_00025` (real→fake);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 0.825
- `wildfake/real/coco_00031` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.923
- `wildfake/real/coco_00038` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.993
- `wildfake/real/coco_00043` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.732
- `wildfake/real/coco_00047` (real→fake);  PatchHead(SID->WF) 0.007 vs DID(WF-native) 0.994
- `wildfake/real/coco_00051` (real→fake);  PatchHead(SID->WF) 0.005 vs DID(WF-native) 0.876
- `wildfake/real/coco_00057` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.779
- `wildfake/real/coco_00060` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.926
- `wildfake/real/coco_00068` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.945
- `wildfake/real/coco_00075` (real→fake);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.793
- `wildfake/real/coco_00078` (real→fake);  PatchHead(SID->WF) 0.015 vs DID(WF-native) 0.979
- `wildfake/real/coco_00086` (real→fake);  PatchHead(SID->WF) 0.001 vs DID(WF-native) 0.937
- `wildfake/real/coco_00088` (real→fake);  PatchHead(SID->WF) 0.010 vs DID(WF-native) 0.861
- `wildfake/real/coco_00090` (real→fake);  PatchHead(SID->WF) 0.004 vs DID(WF-native) 0.973
- `wildfake/real/coco_00091` (real→fake);  PatchHead(SID->WF) 0.004 vs DID(WF-native) 0.984
- `wildfake/real/coco_00092` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.733
- `wildfake/real/coco_00105` (real→fake);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 0.926
- `wildfake/real/coco_00115` (real→fake);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 0.918
- `wildfake/real/coco_00118` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.768
- `wildfake/real/coco_00121` (real→fake);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 0.957
- `wildfake/real/coco_00135` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.903
- `wildfake/real/coco_00137` (real→fake);  PatchHead(SID->WF) 0.014 vs DID(WF-native) 0.937
- `wildfake/real/imagenet_00002` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.852
- `wildfake/real/imagenet_00006` (real→fake);  PatchHead(SID->WF) 0.010 vs DID(WF-native) 0.730
- `wildfake/real/imagenet_00007` (real→fake);  PatchHead(SID->WF) 0.005 vs DID(WF-native) 0.789
- `wildfake/real/imagenet_00018` (real→fake);  PatchHead(SID->WF) 0.003 vs DID(WF-native) 0.737
- `wildfake/real/imagenet_00019` (real→fake);  PatchHead(SID->WF) 0.012 vs DID(WF-native) 0.979
- `wildfake/real/imagenet_00022` (real→fake);  PatchHead(SID->WF) 0.010 vs DID(WF-native) 1.000
- `wildfake/real/imagenet_00025` (real→fake);  PatchHead(SID->WF) 0.013 vs DID(WF-native) 0.835
- `wildfake/real/imagenet_00028` (real→fake);  PatchHead(SID->WF) 0.009 vs DID(WF-native) 0.805
- `wildfake/real/imagenet_00037` (real→fake);  PatchHead(SID->WF) 0.002 vs DID(WF-native) 0.706
- `wildfake/real/imagenet_00045` (real→fake);  PatchHead(SID->WF) 0.010 vs DID(WF-native) 0.999
- `wildfake/real/imagenet_00046` (real→fake);  PatchHead(SID->WF) 0.007 vs DID(WF-native) 0.816
- `wildfake/real/imagenet_00048` (real→fake);  PatchHead(SID->WF) 0.004 vs DID(WF-native) 0.749
- `wildfake/real/imagenet_00050` (real→fake);  PatchHead(SID->WF) 0.007 vs DID(WF-native) 0.922
- `wildfake/real/imagenet_00055` (real→fake);  PatchHead(SID->WF) 0.004 vs DID(WF-native) 0.828
- `wildfake/real/imagenet_00056` (real→fake);  PatchHead(SID->WF) 0.005 vs DID(WF-native) 0.832
- `wildfake/real/imagenet_00061` (real→fake);  PatchHead(SID->WF) 0.005 vs DID(WF-native) 0.734
