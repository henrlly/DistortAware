# PatchHead — a second detector, compared image-for-image with DID

A from-scratch, hackathon-scale reproduction of **"PatchHead: Learning Spatial
Patch Evidence for Generalizable AI-Generated Image Detection"**
(arXiv:2608.09223), run on the **same** WildFake / SID_Set subsets and the
**same** 14-transform robustness suite as the DID detector in [`../did`](../did),
so the two can be compared on every individual test image.

## Method

| Piece | This repo |
|---|---|
| Backbone | **DINOv3 ViT-L/16**, frozen (`timm/vit_large_patch16_dinov3.lvd1689m` — the non-gated mirror of `facebook/dinov3-*`). 256px input → 16×16 = 256 patch tokens, dim 1024. |
| Adaptation | **LoRA** rank 8 (α=16) on every `qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` — 96 adapters, **3.15M** params. |
| Head | **PatchHead**: reshape patch tokens to a (B,1024,16,16) grid → depth-wise 3×3 conv → 1×1 proj to 256 → GroupNorm/GELU → 1×1 conv to a **per-patch logit map**; the image logit is the mean patch logit (GAP over the evidence map). Plus a small CLS-token aux head. **0.28M** params. |
| Training | BCE on the image logit + 0.5·BCE on the CLS logit + 0.3·BCE on every patch logit (each patch inherits the image label). AdamW, cosine, 8–10 epochs. Threshold = balanced-accuracy-optimal on a held-out **train** slice (test never touched), stored in the ckpt — identical protocol to `../did/train.py`. |
| Total params | **306.5M** (of which **3.42M / 1.1% trained**) — vs the DID pipeline's ~1.07B frozen SD-1.5 reconstructor + 22M ResNet heads. |

### Distortion-aware extension

New checkpoints train with distortion awareness enabled by default:

- training augmentations return exact multi-label type and normalized severity
  targets for JPEG, blur, resize, Gaussian noise, colour jitter, and crop;
- the augmentation range covers JPEG quality 25–96, noise sigma .003–.11,
  blur sigma .15–2.25, and downscale factors .2–.97;
- a hybrid auxiliary head combines pooled DINO features with blind analytical
  high-pass, gradient, Laplacian, and block-boundary statistics;
- a zero-initialized threshold adapter predicts a per-image logit-space
  threshold shift from the estimated distortion vector.

Normal evaluation uses predicted distortion only. `--distortion-mode oracle`
is an ablation using known synthetic-transform metadata, and must not be used as
a deployment result. `--distortion-mode off` reports the unadjusted base model.
Evaluation writes base and adjusted accuracy, mean dynamic threshold,
distortion-type precision/recall/F1, and severity MAE.

```bash
python patchhead/evaluate.py --ckpt patchhead/checkpoints/patchhead.pt \
  --distortion-mode predicted --out results/patchhead/results_distortion_aware
python patchhead/evaluate.py --ckpt patchhead/checkpoints/patchhead.pt \
  --distortion-mode oracle --out results/patchhead/results_distortion_oracle
```

Distortion-aware inference exposes both the adjusted AIGC score and the
estimated distortion:

```bash
python patchhead/predict.py image.jpg --ckpt patchhead/checkpoints/patchhead.pt
```

Preprocessing matches the DID pipeline exactly: every image is canonicalised to
200×200 first (WildFake stores reals at 200px / fakes at 256px — forcing both
removes that confound) and then resized to the model input; robustness transforms
are applied at native resolution first.

Checkpoints store **only trainable tensors** (LoRA, authenticity heads,
distortion estimator, and threshold adapter); the frozen backbone is reloaded
from the timm cache at eval time.

## Run it

### Inference contract

The repository-root inference command now defaults to PatchHead and can expose
the model's existing patch logits from the same forward pass:

```bash
python infer.py \
  --image-dir path/to/images \
  --ckpt path/to/patchhead_pooled.pt \
  --out preds.json \
  --export-patch-evidence \
  --with-physics \
  --physics-auto-proposals \
  --physics-proposal-mask-backend clipseg \
  --physics-proposal-feature-backend patchhead \
  --pretty
```

`patchhead/infer.py` provides the detector-only equivalent. Default output omits
the large patch maps; request them explicitly. The score and checkpoint threshold
match evaluation: `0.5 * (sigmoid(image_logit) + sigmoid(cls_logit))`. Relative
paths disambiguate duplicate basenames, EXIF orientation is applied, corrupt
images are isolated, and the checkpoint SHA-256 is recorded.

The pooled checkpoint is not tracked in Git and must be supplied externally.
The supplied artifact has now passed preflight and bounded end-to-end validation;
its SHA-256 and DINO snapshot are recorded in
[`../physics/docs/checkpoint_validation.md`](../physics/docs/checkpoint_validation.md).
Primary inference rejects checkpoints whose stored dataset tag is not `pooled`
and validates threshold/size/model-state metadata before constructing the DINO
backbone. Contract tests still use a deterministic fake runtime so clean clones
can validate preprocessing, score composition, JSON shape, and optional physics
plumbing without model downloads.

Compatibility note: the supplied pooled artifact is the released one-logit
binary checkpoint, while the newer distortion-aware training model uses three
classes. `patchhead.inference` detects the stored one-output tensor shape and
reconstructs only the matching binary output layers before loading the original
weights. Its sigmoid score, threshold, LoRA tensors, and backbone are unchanged.
Three-class pooled inference remains deliberately rejected until a matching
checkpoint and versioned score/threshold contract are available; oracle
distortion metadata is never substituted in the browser path.

For automatic planar-reflection proposals, `PatchHeadDetector.forward_with_features`
can return the final dense DINO token grid from the same forward pass. Root
`infer.py` passes it to physics only when `--physics-proposal-feature-backend
patchhead` is requested. The grid is float16, held in memory rather than JSON,
and capped at 512 MiB by default. The normal `forward` return tuple and official
score formula are unchanged.

For macOS or interactive evaluation, pass `--workers 0` to `evaluate.py`. The
default remains eight workers for cluster throughput; worker count does not
change model or metric behavior.

### Training and evaluation

```bash
# one job per dataset: train → evaluate (clean + 14 transforms) → compare vs DID
sbatch --export=ALL,DS=wildfake,EPOCHS=8  patchhead/pipeline.sh
sbatch --export=ALL,DS=pooled,EPOCHS=8    patchhead/pipeline.sh   # WildFake + SID_Set
sbatch --export=ALL,DS=sid_set,EPOCHS=10  patchhead/pipeline.sh

# zero-shot cross-dataset (eval only, mirrors the DID zero-shot study)
sbatch patchhead/zeroshot.sh

# fast de-risk (build model, 1 mini-epoch, clean eval, compare)
sbatch patchhead/smoke.sh
```

Needs the DINOv3 weights in the HF cache (login node, ~1.2GB):

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python -c \
 "from huggingface_hub import hf_hub_download as d; \
  [d('timm/vit_large_patch16_dinov3.lvd1689m', f) for f in ('config.json','model.safetensors')]"
```

`pipeline.sh` auto-selects the DID checkpoint / feature cache / metrics to compare
against per `DS` (wildfake → `checkpoints/did_sd15_resnet18.pt`, etc.).

## Results — PatchHead vs DID (SD-1.5 / ResNet-18), same test sets

| Test set | metric | **PatchHead** | DID |
|---|---|---:|---:|
| WildFake | clean acc / AUC | **98.5% / 1.000** | 88.6% / 0.957 |
| WildFake | mean over 14 transforms | **99.0%** | 93.0% |
| WildFake | worst transform | **97.0%** (noise0.10) | 88.0% (jpeg30) |
| SID_Set (native) | clean acc / AUC | **100% / 1.000** | 92.7% / 0.971 |
| SID_Set (native) | mean over 14 transforms | **99.3%** | 87.4% |
| Pooled (WF+SID), eval on the 1500-img pooled test | clean acc / AUC | **99.6% / 1.000** | 87.2% / 0.946 |
| Pooled | mean over 14 transforms | **98.6%** | 89.3% |

PatchHead is better on clean accuracy, AUC, and every robustness slice, on all
three datasets. On WildFake it makes **zero false negatives** — it never misses a
fake; all 18 clean errors are real photos flagged as AI.

Full per-transform tables + charts: `results_<ds>/report.md`,
`results_<ds>/robustness.png`.

## Which images does each detector get wrong? (`results_compare_<ds>/comparison.md`)

The brief's question — *for the images predicted wrong, is it both detectors or
only one?* — answered on the clean test set:

| | WildFake | Pooled | SID_Set |
|---|---:|---:|---:|
| Both correct | 1058 | 1299 | 278 |
| **Only DID wrong** | **124** | **195** | **22** |
| **Only PatchHead wrong** | **5** | **3** | **0** |
| **Both wrong** | **13** | **3** | **0** |
| error φ-coefficient | +0.24 | +0.07 | 0.00 |
| McNemar p | 3e-25 | 6e-42 | 8e-6 |

- **It is overwhelmingly one-way.** PatchHead's error set is very nearly a
  **subset** of DID's: 13 of 18 WildFake errors (72%), 3 of 6 pooled errors, are
  images DID also gets wrong; PatchHead has only 5 / 3 / 0 errors that DID
  handles. DID has 124 / 195 / 22 errors that PatchHead handles.
- **The shared hard core is all false positives** — real natural photos both
  detectors call fake: `imagenet_00037/38/45/46/55/63/73/75/105/123/133`,
  `coco_00119/121`. These are the texture-rich ImageNet/COCO photos already
  flagged as DID's worst FPs; they sit close to a diffusion manifold *and* look
  atypical to DINOv3. An ensemble would not fix them.
- **φ ≈ 0–0.24**: the errors are only weakly correlated — the two detectors
  mostly fail on *different* images — but that is moot here because PatchHead
  simply dominates. An oracle picking the better detector per image would reach
  98.9% / 99.8% / 100% (vs 98.5 / 99.6 / 100 for PatchHead alone), so there is
  almost no ensemble headroom.

## Zero-shot cross-dataset — PatchHead collapses too

`patchhead/zeroshot.sh`, mirroring the DID zero-shot study:

| Train → test | PatchHead acc / AUC | DID acc / AUC |
|---|---:|---:|
| WildFake → SID_Set | 47.3% / 0.569 | 58.3% / 0.575 |
| SID_Set → WildFake | 51.8% / **0.255** | 46.3% / 0.416 |

Same conclusion as DID: **a single-dataset detector is near-chance on an unseen
dataset**, both directions — the LoRA+head learn the source domain's real-image
prior and its generator artifacts, not a universal "is-synthetic" rule. (SID→WF
AUC 0.255 < 0.5: the SID-trained model's score is actually *anti*-correlated on
WildFake.) **Pooling the two datasets fixes it** for PatchHead just as it did for
DID — the pooled model scores 99.6% on the combined test set — and pooled
PatchHead recovers far more of the gap than pooled DID (99.6% vs 87.2%).

## Files

| File | Role |
|---|---|
| `model.py` | `PatchHeadDetector` (DINOv3 + `LoRALinear` + `PatchHead`), `load_detector` |
| `inference.py` / `infer.py` | reusable and command-line pooled inference with optional same-pass patch evidence and in-memory dense features for physics |
| `tests/` | checkpoint-independent score, preprocessing, path, error-isolation, and unified-physics contracts |
| `data.py` | image dataset over `data/<ds>/<split>/<label>/*.png`, DID-matched preprocessing, shared `src.transforms` suite |
| `train.py` | training loop, held-out-train-slice threshold calibration, 14MB trainable-only checkpoint |
| `evaluate.py` | clean + 14-transform acc/AUC, full per-image prediction dump (`preds_clean.json`), error analysis |
| `harness/did_predictions.py` | dump per-image clean predictions from a DID checkpoint, in the same key format |
| `compare.py` | image-for-image agreement: contingency table, φ-coefficient, McNemar, the both-wrong / only-A / only-B image lists |
| `make_report.py` | `results_<ds>/report.md` + PatchHead-vs-DID per-transform bar chart |
| `pipeline.sh` / `smoke.sh` / `zeroshot.sh` | SLURM jobs |

## Caveats

- These are **in-distribution** numbers (train/test split of one dataset, or of a
  pooled pair). The near-ceiling scores partly reflect how separable WildFake /
  SID fakes are in DINOv3 feature space — the zero-shot section shows the honest
  limit. The paper's generalization claims would need many more held-out
  generators to verify.
- Threshold calibration saturates (val accuracy hits 100% within 1–2 epochs), so
  the decision threshold is the midpoint of the tying plateau rather than a
  sharply-identified value. `acc@0.5` and `acc@t` are both reported.
- 8–10 epochs, LoRA rank 8, single input resolution (256). No ensemble, no
  test-time augmentation.
