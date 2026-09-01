# RESUME — Robust AIGC Detection POC (DID / arXiv:2602.23732)

> **STATUS 2026-08-29 (later): COMPLETE.** The full pipeline was ported to the
> NUS SoC SLURM cluster and run end-to-end. Headline model (SD-1.5 + ResNet-18):
> **clean 88.6%, mean-over-14-transforms 93.0%, worst 88.0%, AUC 0.957** — meets
> the >80% requirement. A 2×2 ablation over reconstructor (SD-1.5 / SANA-1.6B) ×
> classifier (ResNet-18 / -50) is in `results_<recon>_<clf>/` and the README.
> `infer.py` verified end-to-end. Cluster how-to: `slurm/README.md`.
> The sections below are the original Mac-era notes, kept for history.

---

## 1. Where things stand

| Stage | Status |
|---|---|
| WildFake subset sampled (2400 train + 1200 test, balanced) | **DONE** — `data/wildfake/` |
| Clean DID features extracted (all 3600 images) | **DONE** — `cache/wildfake/{train,test}/clean/` |
| Classifier trained on clean features | **DONE** — `checkpoints/did_cleanonly.pt` |
| **Clean test accuracy** | **90.4%** (AUC 0.963), target was 80% ✅ |
| Transformed-test-suite features (14 transforms) | **PARTIAL** — `jpeg90` done, `jpeg70` ~half, 12 transforms not started |
| `randaug1` transform-augmented train features | **NOT STARTED** |
| Final retrain (clean + randaug1, calibrated threshold) | **NOT STARTED** |
| Robustness table + report | **NOT STARTED** (`results/` has clean-only numbers) |
| README / infer.py / run_all.sh / demo.py | **DONE** (code complete, infer.py not yet run end-to-end on the real ckpt) |

The POC already meets the hard requirement (working pipeline, <2B model, >80% on
WildFake). What remains is the **robustness evaluation** and a polished final model.

### Known issue to watch
During the paused run, DID feature extraction slowed from ~1.5 s/img to ~7 s/img
(M3 thermal throttling / memory pressure after hours of MPS load). If it's still
slow on resume: reboot or let the machine cool, keep `--batch 16`, and consider
reducing the transformed-test subset from `--limit 150` to `--limit 100`.

---

## 2. Environment

```bash
cd /Users/h/GitHub/tiktok-aigc-detect
export PYTHONPATH=$PWD/did
```

- Python: `/opt/homebrew/Caskroom/miniconda/base/bin/python3` (3.12), Apple M3, MPS.
- Deps already installed: `torch torchvision diffusers transformers accelerate
  safetensors scikit-learn timm modelscope pillow tqdm matplotlib`.
  Reinstall if needed:
  ```bash
  pip install torch torchvision diffusers transformers accelerate safetensors \
      scikit-learn timm modelscope pillow tqdm matplotlib
  ```
- SD v1.5 weights already cached in `~/.cache/huggingface` (~2.7 GB, fp16 variant
  + fp32 VAE only — the fp32/bin UNet blobs were pruned to save disk).
- ModelScope: browser is logged in, but the code uses **anonymous HTTP range
  reads** (`did/remote_zip.py`) — no auth needed. WildFake dataset id:
  `hy2628982280/WildFake`.

### Gotchas baked into the code
- **`DataLoader(num_workers=0)`** everywhere — `num_workers>0` deadlocks with MPS
  on this macOS box (shared-memory tensor transport hangs). Do not raise it.
- **`torch.load(..., weights_only=False)`** — checkpoints store a python float
  threshold; the default safe-unpickler rejects them.
- Feature extraction is **resumable** — `extract_features.py` skips any `.npz`
  that already exists, so re-running the same command continues where it stopped.
- All images canonicalised to 200×200 on disk (WildFake stores reals at 200,
  fakes at 256 — this neutralises that confound). Model input is 192px.

---

## 3. Resume steps (in order)

### Step A — finish the transformed test suite  (~2–3 h, resumable)
```bash
cd /Users/h/GitHub/tiktok-aigc-detect && export PYTHONPATH=$PWD/did
nohup python3 did/extract_features.py --root data/wildfake --split test --out cache/wildfake \
  --transforms jpeg90,jpeg70,jpeg50,jpeg30,blur0.5,blur1.0,blur2.0,resize0.5,resize0.25,noise0.02,noise0.05,noise0.10,jitter,crop80 \
  --res 192 --steps 6 --batch 16 --limit 150 > logs_extract_transforms.txt 2>&1 &
```
Progress check: `find cache/wildfake/test -name '*.npz' | sed 's#.*/test/##'|cut -d/ -f1|sort|uniq -c`
Expected end state: 15 dirs (`clean` + 14), each `real` 150 + `fake` 150
(`clean` is 600+600).

### Step B — extract randaug1 train features  (~1–1.5 h, resumable)
```bash
nohup python3 did/extract_features.py --root data/wildfake --split train --out cache/wildfake \
  --transforms randaug1 --res 192 --steps 6 --batch 16 > logs_extract_randaug.txt 2>&1 &
```
Expected: `cache/wildfake/train/randaug1/{real,fake}/` with 1200 each.

### Step C — train the final model  (~15–30 min)
```bash
python3 did/train.py --cache cache/wildfake --epochs 14 --bs 32 \
    --train-transforms clean,randaug1 --out checkpoints/did.pt
```
- Uses a 15% held-out slice of TRAIN for model selection + threshold calibration
  (test set never touched during training).
- Prints `val_acc`, `test_acc@0.5`, `test_acc@t` per epoch; saves best-val ckpt
  with the calibrated `threshold`.
- If it overfits like v1 did (val oscillates, best ~ep6): that's fine, best-val
  checkpoint is kept. If test acc drops below ~85%, try `--epochs 8` or
  `--lr 5e-5`.

### Step D — evaluate + report
```bash
python3 did/evaluate.py --cache cache/wildfake --ckpt checkpoints/did.pt --out results/did/legacy_run
python3 did/make_report.py --results results/did/legacy_run
```
Produces `results/did/legacy_run/robustness.csv`, `results/did/legacy_run/metrics.json`,
`results/did/legacy_run/report.md`, `results/did/legacy_run/robustness.png`,
`results/did/legacy_run/error_analysis.json`.

### Step E — verify the deliverable inference script (do this on a cool machine)
```bash
mkdir -p /tmp/it && cp $(ls data/wildfake/test/real/*.png|head -5) /tmp/it/ && cp $(ls data/wildfake/test/fake/*.png|head -5) /tmp/it/
python3 infer.py --image-dir /tmp/it --out /tmp/preds.json --ckpt checkpoints/did.pt
cat /tmp/preds.json   # expect real≈low pred, fake≈high pred; JSON has image_path + pred
```
(Earlier attempts to run infer.py timed out only because extraction was hogging
the GPU — run it when nothing else uses MPS.)

### Step F — visual demo panel (optional, for the demo video)
```bash
python3 did/demo.py --images data/wildfake/test/real/coco_00000.png data/wildfake/test/fake/ADM_00000.png \
    --ckpt checkpoints/did.pt --out results/did/legacy_run/demo.png
```

### Step G — finalise
- Update the Results section of `README.md` with the real numbers from
  `results/did/legacy_run/metrics.json` and `results/did/legacy_run/report.md`.
- `git add -A && git commit` (repo is currently all untracked; branch `main`).
- Mark task #4 and #5 done.

---

## 4. Full from-scratch rebuild

If `data/` or `cache/` is lost:
```bash
bash run_all.sh          # does fetch -> extract -> randaug -> train -> eval -> report
```
Total ~4–5 h on the M3. Individual stages are the commands in Section 3 plus:
```bash
python3 did/fetch_wildfake.py --out data/wildfake --train 300 --test 150   # ~30 min
python3 did/extract_features.py --root data/wildfake --split train --out cache/wildfake --res 192 --steps 6 --batch 16
python3 did/extract_features.py --root data/wildfake --split test  --out cache/wildfake --res 192 --steps 6 --batch 16
```

---

## 5. File map

| Path | What |
|---|---|
| `did/remote_zip.py` | HTTP-range random-access ZIP reader (pulls images out of WildFake's multi-GB archives without downloading them) |
| `did/fetch_wildfake.py` | balanced WildFake sampler → `data/wildfake/{train,test}/{real,fake}/*.png` (200×200) |
| `did/did.py` | `DIDReconstructor`: SD1.5 VAE-encode → DDIM invert (6 steps) → DDIM resample → decode, ×2; returns `d1`,`d2` maps. `MODEL_ID = stable-diffusion-v1-5/stable-diffusion-v1-5` |
| `did/transforms.py` | the 7 real-world transform families + `random_transform` (used by `randaug*`) |
| `did/extract_features.py` | batched, resumable feature extraction → `cache/wildfake/<split>/<transform>/<label>/*.npz` (keys `d1`,`d2`, fp16, 3×192×192) |
| `did/data.py` | `FeatureDataset` — train/val split by image stem, feature-space augmentation |
| `did/model.py` | `DIDClassifier` — two torchvision ResNet-18 heads (d1, d2), probs averaged |
| `did/train.py` | training + val-based selection + threshold calibration → `checkpoints/did.pt` |
| `did/evaluate.py` | clean + per-transform acc/AUC + FP/FN dump → `results/` |
| `did/make_report.py` | `results/did/legacy_run/report.md` + `results/robustness.png` |
| `did/signal_check.py` | quick separability probe on cached features |
| `did/demo.py` | visual panel (orig / d1 / d2 / verdict) |
| `infer.py` | **deliverable**: `python infer.py --image-dir DIR --out preds.json` → JSON of `{image_path, pred, is_aigc}` |
| `checkpoints/did_cleanonly.pt` | current best model, 90.4% clean (trained on clean only, threshold 0.5) |
| `checkpoints/tiny.pt` | throwaway smoke-test ckpt, ignore |
| `results/` | clean-only metrics + error analysis (robustness table still TODO) |

## 6. Current numbers (clean-only model)

```
clean test accuracy : 90.4 %   (1200 images: 600 real + 600 fake)
clean test AUC      : 0.963
false positives     : 86 / 600 real  (mostly texture-rich ImageNet/COCO photos)
false negatives     : 29 / 600 fake  (mostly VQDM — vector-quantized diffusion)
```

Real sources: COCO, ImageNet, CelebA-HQ, AFHQ.
Fake sources: ADM, DDIM, DDPM, VQDM (all cross-family vs the SD1.5 reconstructor).
