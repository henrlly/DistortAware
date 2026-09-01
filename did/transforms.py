"""Real-world post-processing transforms from the problem statement (5.2).

Each function takes and returns a PIL.Image (RGB).  `TRANSFORMS` maps a short name
to a callable; `eval_suite()` yields (name, fn) for the robustness table.
"""
import io
import numpy as np
from PIL import Image, ImageFilter


def _np(im):
    return np.asarray(im).astype(np.float32)


def jpeg(im, quality=70):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gblur(im, sigma=1.0):
    return im.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_cycle(im, scale=0.5):
    w, h = im.size
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)
    return small.resize((w, h), Image.Resampling.BICUBIC)


def gnoise(im, sigma=0.05):
    a = _np(im) / 255.0
    a = a + np.random.normal(0, sigma, a.shape)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def color_jitter(im, amount=0.2):
    from PIL import ImageEnhance
    rng = np.random.RandomState(0)
    for Enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        f = 1.0 + rng.uniform(-amount, amount)
        im = Enh(im).enhance(f)
    return im


def center_crop(im, frac=0.8):
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch)).resize((w, h), Image.Resampling.BICUBIC)


TRANSFORMS = {
    "clean": lambda im: im,
    "jpeg90": lambda im: jpeg(im, 90),
    "jpeg70": lambda im: jpeg(im, 70),
    "jpeg50": lambda im: jpeg(im, 50),
    "jpeg30": lambda im: jpeg(im, 30),
    "blur0.5": lambda im: gblur(im, 0.5),
    "blur1.0": lambda im: gblur(im, 1.0),
    "blur2.0": lambda im: gblur(im, 2.0),
    "resize0.5": lambda im: resize_cycle(im, 0.5),
    "resize0.25": lambda im: resize_cycle(im, 0.25),
    "noise0.02": lambda im: gnoise(im, 0.02),
    "noise0.05": lambda im: gnoise(im, 0.05),
    "noise0.10": lambda im: gnoise(im, 0.10),
    "jitter": lambda im: color_jitter(im, 0.2),
    "crop80": lambda im: center_crop(im, 0.8),
}


def eval_suite():
    return list(TRANSFORMS.items())


def random_transform(im, rng):
    """One random transform (for training-time robustness augmentation)."""
    ops = []
    if rng.random() < 0.5:
        ops.append(lambda i: jpeg(i, rng.choice([90, 70, 50, 30])))
    if rng.random() < 0.3:
        ops.append(lambda i: gblur(i, rng.choice([0.5, 1.0, 2.0])))
    if rng.random() < 0.3:
        ops.append(lambda i: resize_cycle(i, rng.choice([0.5, 0.25])))
    if rng.random() < 0.3:
        ops.append(lambda i: gnoise(i, rng.choice([0.02, 0.05, 0.10])))
    if rng.random() < 0.3:
        ops.append(lambda i: color_jitter(i, 0.2))
    if rng.random() < 0.3:
        ops.append(lambda i: center_crop(i, rng.uniform(0.8, 0.95)))
    rng.shuffle(ops)
    for op in ops:
        im = op(im)
    return im
