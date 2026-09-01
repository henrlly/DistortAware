# Residual-Aware AIGC Detection Baseline

A compact, explainable baseline for the **TikTok TechJam 2026 AIGC image
detection challenge**. The model looks for low-level inconsistencies in image
detail by combining the RGB image with a high-pass residual, then predicts:

1. whether the image is authentic, fully synthetic, or AI-tampered; and
2. a pixel-level heatmap showing where synthetic or manipulated evidence may be
   present.

For the challenge's binary decision, `fully_synthetic` and `ai_tampered` are
collapsed into a single **AI** class after inference. The network remains
three-class during training because that representation generalized better than
training the classifier as binary.

This is deliberately a small forensic baseline, not a production moderation
system. It has **386,932 trainable parameters**, well below the challenge's
2-billion-parameter limit.

## Current results

The selected checkpoint is [`models/mask_classifier.pt`](models/mask_classifier.pt).
The complete machine-readable report is in
[`reports/evaluation.json`](reports/evaluation.json).

| Evaluation set | Samples | Accuracy | Balanced accuracy | Authentic recall | AI recall |
|---|---:|---:|---:|---:|---:|
| SID validation | 300 | 67.3% | 57.0% | 26.0% | 88.0% |
| WildFake cross-dataset | 200 | 74.0% | 74.0% | 82.0% | 66.0% |

Additional localization result:

- SID tampered-region mask IoU: **15.5%**

SID binary confusion matrix (`rows = truth`, `columns = prediction`):

| | Predicted authentic | Predicted AI |
|---|---:|---:|
| Authentic | 26 | 74 |
| AI (synthetic or tampered) | 24 | 176 |

### Reading these results honestly

- **WildFake is the more meaningful generalization result.** The model is
  trained only on SID, so WildFake measures transfer to a different dataset and
  image pipeline.
- The SID figure is a **validation result**, not an untouched final test result.
  The same SID validation subset participates in checkpoint selection.
- SID has 100 authentic and 200 combined-AI examples in this evaluation.
  Ordinary accuracy is therefore optimistic; balanced accuracy exposes the
  model's strong tendency to call SID images AI.
- WildFake evaluation uses 100 CelebA-HQ authentic images and 100 DDIM synthetic
  images, starting at archive offset 1,100. These are not used for training.
- The localization head is still experimental. A 15.5% IoU indicates weak
  localization, and its heatmap should be treated as supporting evidence rather
  than a reliable segmentation.

## How the technique works

### 1. Construct a high-pass residual

Every image is converted to RGB and resized to 192×192. We then calculate a
grayscale residual:

```text
gray     = RGB-to-grayscale(image)
smooth   = GaussianBlur(gray, sigma=1)
residual = gray - smooth
```

The residual suppresses broad shapes and slowly changing colour, emphasizing
fine texture, sharpening, compression boundaries, noise and generator traces.
The signed residual is scaled and centred at 0.5 so positive and negative detail
remain distinguishable.

The final network input has four channels:

```text
[red, green, blue, signed high-pass residual]
```

This is more useful than reducing residual entropy to one scalar. A global
entropy value mixes together real sensor noise, scene texture, JPEG artifacts,
resizing and synthetic traces. Preserving the residual spatially lets the model
learn *where* statistics change and whether those changes align with a
manipulated region.

### 2. Multi-task U-Net

The four-channel tensor enters a small U-Net with two output heads:

```text
RGB + residual
      │
      ▼
shared encoder ───────────────► 3-class classification head
      │                          authentic / synthetic / tampered
      ▼
U-Net decoder ────────────────► pixel-level mask head
```

The classification head uses bottleneck features to make an image-level
decision. The decoder combines coarse semantic features with high-resolution
skip connections to predict a mask.

### 3. Mask supervision

SID's three classes receive different mask targets:

| SID label | Classification target | Segmentation target |
|---|---|---|
| Authentic | `authentic` | all-zero mask |
| Fully synthetic | `fully_synthetic` | all-one mask |
| AI-tampered | `ai_tampered` | SID's supplied manipulation mask |

The total objective is:

```text
classification cross-entropy
+ mask binary cross-entropy
+ mask Dice loss
```

At inference time, any winning class other than `authentic` becomes the binary
prediction `ai`.

## Datasets and evaluation protocol

### SID_Set

- Loaded directly from Hugging Face in streaming mode.
- Training: 300 images from each of the three classes (900 total).
- Validation/checkpoint selection: 100 images per class (300 total).
- Only horizontal flipping is currently applied as training augmentation.

### WildFake

WildFake is used only for cross-dataset evaluation:

- Authentic: CelebA-HQ archive.
- Synthetic: DDIM archive.
- 100 images per class, read directly from ZIP files without extraction.

The competition demonstration groups **COCO val2017** and **DALL·E Advanced**
are not referenced by the training or evaluation code.

## Setup

Requirements:

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- Internet access while streaming SID
- Apple Silicon MPS or CPU; CUDA can be added if needed

Install the locked environment:

```bash
uv sync
```

## Run inference

Classify one image:

```bash
uv run python scripts/predict.py path/to/image.jpg
```

Save the predicted evidence mask as well:

```bash
uv run python scripts/predict.py path/to/image.jpg \
  --mask-output predicted_mask.png
```

Example response shape:

```json
{
  "prediction": "fully_synthetic",
  "binary_prediction": "ai",
  "confidence": 0.81,
  "ai_probability": 0.93,
  "class_probabilities": {
    "authentic": 0.07,
    "fully_synthetic": 0.81,
    "ai_tampered": 0.12
  },
  "predicted_mask_area": 0.76
}
```

These numbers illustrate the output format; they are not a recorded benchmark
sample.

Use another checkpoint if required:

```bash
uv run python scripts/predict.py image.jpg \
  --checkpoint models/another_checkpoint.pt
```

## Train

The default command streams SID, loads a balanced 900-image training subset and
300-image validation subset, trains for 12 epochs, and selects the checkpoint
with the best sum of three-class validation accuracy and tamper-mask IoU.

```bash
uv run python scripts/train.py
```

Explicit equivalent:

```bash
uv run python scripts/train.py \
  --train-per-class 300 \
  --validation-per-class 100 \
  --size 192 \
  --epochs 12 \
  --output models/mask_classifier.pt
```

Useful smoke test:

```bash
uv run python scripts/train.py \
  --train-per-class 10 \
  --validation-per-class 5 \
  --epochs 1 \
  --output models/smoke.pt
```

Training automatically uses Apple MPS when available and otherwise runs on CPU.

## Evaluate

SID is streamed automatically. Place the permitted WildFake archives at:

```text
data/wildfake/Images/Real/celebahq.zip
data/wildfake/Images/Diffusion_based/DDIM.zip
```

Then run:

```bash
uv run python scripts/evaluate.py
```

Equivalent explicit command:

```bash
uv run python scripts/evaluate.py \
  --checkpoint models/mask_classifier.pt \
  --per-class 100 \
  --offset 1100 \
  --real data/wildfake/Images/Real/celebahq.zip \
  --fake data/wildfake/Images/Diffusion_based/DDIM.zip \
  --output reports/evaluation.json
```

## Alignment with the TechJam rubric

| Rubric area | Current evidence | Remaining work |
|---|---|---|
| Clear technical approach | Fixed residual construction, compact multi-task U-Net and binary decision rule are documented and reproducible. | Ablate RGB-only versus residual-only versus combined input. |
| Image-level AIGC detection | Evaluated on SID and cross-dataset WildFake. | Add an untouched SID test partition and more WildFake generators. |
| Robustness to redistribution | The residual input is intended to expose low-level traces. | **Not yet benchmarked:** JPEG, blur, crop, rescale, sharpening and colour adjustments must be measured explicitly. |
| Generalization | 74% accuracy on WildFake without WildFake training. | Report confidence intervals and generator-held-out results. |
| False positives | Per-class recall and confusion matrix reveal the 74% SID authentic false-positive rate. | Tune/calibrate the binary threshold on a separate calibration split. |
| Explainability | Returns class probabilities and a spatial evidence mask. | Improve mask IoU and add qualitative overlays/error examples. |
| Compute constraint | 386,932 parameters; trains on a laptop-scale subset. | Report training time, peak memory and inference latency on the demonstration machine. |

### Required robustness matrix

Before submission, evaluate the same untouched images under at least:

| Transformation | Suggested levels |
|---|---|
| JPEG compression | quality 95, 75, 50, 30 |
| Gaussian blur | sigma 0.5, 1, 2 |
| Rescaling | 0.5×, 0.75×, 1.5× |
| Cropping | retain 90%, 75%, 50% |
| Sharpening | mild and strong |
| Colour adjustment | brightness/contrast/saturation ±20% |

Report balanced accuracy, authentic recall, AI recall and the change from clean
performance for every transformation. Apply identical transformations to both
classes so the transformation itself cannot become a label shortcut.

## Known limitations and trade-offs

1. **Residual evidence is not uniquely AI evidence.** Sensor noise, denoising,
   screenshots, JPEG compression, sharpening and resizing all alter the same
   frequencies.
2. **False positives are currently high on SID.** The selected model detects
   most AI samples but accepts only 26% of SID authentic images.
3. **Dataset effects remain possible.** WildFake's real and synthetic classes
   differ in source and content as well as authenticity.
4. **The model resizes everything to 192×192.** This is efficient, but may erase
   small edits and fine generator fingerprints.
5. **Mask localization is weak.** The current 15.5% IoU is insufficient for
   production localization.
6. **Probabilities are not calibrated.** `ai_probability` is a model score, not
   a guaranteed real-world probability.
7. **Metadata is not proof.** The provenance helper reports EXIF hints only; it
   does not yet cryptographically verify C2PA credentials or SynthID.

The detector should therefore be presented as one explainable signal in an
ensemble, not as definitive proof that an image is authentic or generated.

## Repository layout

```text
src/ai_detection/
  data.py          SID streaming
  model.py         residual preprocessing, U-Net and losses
  inference.py     checkpoint loading and prediction API
  provenance.py    conservative metadata inspection

scripts/
  train.py         SID training and checkpoint selection
  evaluate.py      SID + held-out WildFake evaluation
  predict.py       single-image CLI

models/            selected checkpoint
reports/           evaluation JSON
data/              local datasets; excluded from Git
```

## Research context

The design is motivated by research showing that synthetic images can exhibit
mid/high-frequency and autocorrelation artifacts, while also recognizing that
post-processing can weaken those traces:

- Corvi et al., [*Intriguing Properties of Synthetic Images: From Generative
  Adversarial Networks to Diffusion Models*](https://arxiv.org/abs/2304.06408)
- Zhu et al., [*GenImage: A Million-Scale Benchmark for Detecting AI-Generated
  Image*](https://arxiv.org/abs/2306.08571)
- Grommelt et al., [*Fake or JPEG? Revealing Common Biases in Generated Image
  Detection Datasets*](https://arxiv.org/abs/2403.17608)

These papers also motivate our emphasis on cross-dataset testing, class-specific
error reporting and the planned post-processing robustness matrix.
