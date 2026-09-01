"""PatchHead preprocessing, robustness transforms, and distortion metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

TYPE_NAMES = ("jpeg", "blur", "resize", "noise", "jitter", "crop")
MAGNITUDE_NAMES = (
    "jpeg_severity", "blur_sigma", "resize_loss", "noise_sigma",
    "brightness_delta", "contrast_delta", "saturation_delta", "crop_loss",
)
DISTORTION_TARGET_DIM = len(TYPE_NAMES) + len(MAGNITUDE_NAMES)

TRANSFORMS = {
    "clean": None,
    "jpeg90": ("jpeg", 90), "jpeg70": ("jpeg", 70),
    "jpeg50": ("jpeg", 50), "jpeg30": ("jpeg", 30),
    "blur0.5": ("blur", .5), "blur1.0": ("blur", 1.0),
    "blur2.0": ("blur", 2.0), "resize0.5": ("resize", .5),
    "resize0.25": ("resize", .25), "noise0.02": ("noise", .02),
    "noise0.05": ("noise", .05), "noise0.10": ("noise", .10),
    "jitter": ("jitter", None), "crop80": ("crop", .8),
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class DistortionSpec:
    """Multi-label distortion description with normalized magnitudes.

    Multiple operations may be present because redistribution pipelines often
    compose JPEG, resize, noise, and colour changes.
    """

    types: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in TYPE_NAMES})
    magnitudes: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in MAGNITUDE_NAMES})

    def mark(self, kind: str, **values: float) -> "DistortionSpec":
        self.types[kind] = 1.0
        for name, value in values.items():
            self.magnitudes[name] = max(self.magnitudes[name], _clip01(value))
        return self

    def merge(self, other: "DistortionSpec") -> "DistortionSpec":
        for name in TYPE_NAMES:
            self.types[name] = max(self.types[name], other.types[name])
        for name in MAGNITUDE_NAMES:
            self.magnitudes[name] = max(self.magnitudes[name], other.magnitudes[name])
        return self

    def target(self) -> np.ndarray:
        return np.asarray(
            [self.types[name] for name in TYPE_NAMES]
            + [self.magnitudes[name] for name in MAGNITUDE_NAMES],
            dtype=np.float32,
        )

    @classmethod
    def clean(cls) -> "DistortionSpec":
        return cls()


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _noise(image: Image.Image, sigma: float, seed: int | None = None) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed).normal(0.0, sigma, array.shape).astype(np.float32)
    return Image.fromarray((np.clip(array + noise, 0, 1) * 255).astype(np.uint8), "RGB")


def apply_with_spec(
    image: Image.Image,
    name: str,
    *,
    rng: random.Random | None = None,
) -> tuple[Image.Image, DistortionSpec]:
    """Apply one named benchmark transform and return its exact metadata."""
    spec = TRANSFORMS[name]
    if spec is None:
        return image, DistortionSpec.clean()
    rng = rng or random.Random(0)
    kind, value = spec
    result = DistortionSpec.clean()
    if kind == "jpeg":
        quality = int(value)
        return _jpeg(image, quality), result.mark("jpeg", jpeg_severity=(100 - quality) / 70)
    if kind == "blur":
        sigma = float(value)
        return image.filter(ImageFilter.GaussianBlur(sigma)), result.mark("blur", blur_sigma=sigma / 2)
    if kind == "resize":
        scale = float(value)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        output = image.resize(size, Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC)
        return output, result.mark("resize", resize_loss=(1 - scale) / .75)
    if kind == "noise":
        sigma = float(value)
        return _noise(image, sigma, rng.randrange(2**32)), result.mark("noise", noise_sigma=sigma / .10)
    if kind == "jitter":
        brightness = rng.uniform(.8, 1.2)
        contrast = rng.uniform(.8, 1.2)
        saturation = rng.uniform(.8, 1.2)
        output = ImageEnhance.Brightness(image).enhance(brightness)
        output = ImageEnhance.Contrast(output).enhance(contrast)
        output = ImageEnhance.Color(output).enhance(saturation)
        return output, result.mark(
            "jitter",
            brightness_delta=abs(brightness - 1) / .2,
            contrast_delta=abs(contrast - 1) / .2,
            saturation_delta=abs(saturation - 1) / .2,
        )
    if kind == "crop":
        fraction = float(value)
        width, height = image.size
        crop_w, crop_h = max(1, int(width * fraction)), max(1, int(height * fraction))
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        output = image.crop((left, top, left + crop_w, top + crop_h))
        return output, result.mark("crop", crop_loss=(1 - fraction) / .5)
    raise ValueError(f"Unknown transform: {name}")


def apply(image: Image.Image, name: str) -> Image.Image:
    """Backward-compatible image-only benchmark transform."""
    return apply_with_spec(image, name)[0]


def _severity_tier(rng: random.Random) -> str:
    value = rng.random()
    if value < .35:
        return "mild"
    if value < .75:
        return "moderate"
    return "strong"


def _sample_range(rng: random.Random, tier: str, ranges: dict[str, tuple[float, float]]) -> float:
    low, high = ranges[tier]
    return rng.uniform(low, high)


def _random_operation(image: Image.Image, kind: str, tier: str, rng: random.Random) -> tuple[Image.Image, DistortionSpec]:
    spec = DistortionSpec.clean()
    if kind == "jpeg":
        quality = round(_sample_range(rng, tier, {
            "mild": (80, 96), "moderate": (55, 80), "strong": (25, 55),
        }))
        return _jpeg(image, quality), spec.mark("jpeg", jpeg_severity=(100 - quality) / 75)
    if kind == "blur":
        sigma = _sample_range(rng, tier, {
            "mild": (.15, .6), "moderate": (.6, 1.25), "strong": (1.25, 2.25),
        })
        return image.filter(ImageFilter.GaussianBlur(sigma)), spec.mark("blur", blur_sigma=sigma / 2.25)
    if kind == "resize":
        scale = _sample_range(rng, tier, {
            "mild": (.8, .97), "moderate": (.5, .8), "strong": (.2, .5),
        })
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        output = image.resize(size, Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC)
        return output, spec.mark("resize", resize_loss=(1 - scale) / .8)
    if kind == "noise":
        sigma = _sample_range(rng, tier, {
            "mild": (.003, .02), "moderate": (.02, .055), "strong": (.055, .11),
        })
        return _noise(image, sigma, rng.randrange(2**32)), spec.mark("noise", noise_sigma=sigma / .11)
    if kind == "jitter":
        delta = {"mild": .10, "moderate": .20, "strong": .35}[tier]
        brightness = rng.uniform(1 - delta, 1 + delta)
        contrast = rng.uniform(1 - delta, 1 + delta)
        saturation = rng.uniform(1 - delta, 1 + delta)
        output = ImageEnhance.Brightness(image).enhance(brightness)
        output = ImageEnhance.Contrast(output).enhance(contrast)
        output = ImageEnhance.Color(output).enhance(saturation)
        return output, spec.mark(
            "jitter",
            brightness_delta=abs(brightness - 1) / .35,
            contrast_delta=abs(contrast - 1) / .35,
            saturation_delta=abs(saturation - 1) / .35,
        )
    if kind == "crop":
        fraction = _sample_range(rng, tier, {
            "mild": (.85, .97), "moderate": (.65, .85), "strong": (.45, .65),
        })
        width, height = image.size
        crop_w, crop_h = max(1, int(width * fraction)), max(1, int(height * fraction))
        left = rng.randint(0, max(0, width - crop_w))
        top = rng.randint(0, max(0, height - crop_h))
        output = image.crop((left, top, left + crop_w, top + crop_h)).resize(image.size, Image.Resampling.BICUBIC)
        return output, spec.mark("crop", crop_loss=(1 - fraction) / .55)
    raise ValueError(kind)


def random_train_transform(
    image: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, DistortionSpec]:
    """Apply a severity-balanced, optionally composed augmentation policy.

    Strong JPEG/noise/blur/downscale examples are intentional: they match and
    slightly exceed the robustness suite rather than stopping at mild levels.
    """
    if rng.random() < .5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < .1:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    count_draw = rng.random()
    operation_count = 0 if count_draw < .25 else (1 if count_draw < .80 else 2)
    kinds = rng.sample(list(TYPE_NAMES), operation_count)
    combined = DistortionSpec.clean()
    for kind in kinds:
        image, spec = _random_operation(image, kind, _severity_tier(rng), rng)
        combined.merge(spec)
    return image, combined
