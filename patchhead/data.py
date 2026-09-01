"""Image dataset for the PatchHead detector.

Reads the same folder tree the DID pipeline uses
(`data/<ds>/<split>/<label>/<name>.png`) and applies the same preprocessing:
every image is first canonicalised to 200x200 (WildFake stores reals at 200px
and fakes at 256px -- forcing both removes that confound, exactly as
`did/extract_features.to_tensor` does) and then resized to the model input.

Robustness transforms are owned by `patchhead.transforms`, applied before
canonicalisation so JPEG/blur/noise act at native resolution.
"""
import glob
import os
import random
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from transforms import DistortionSpec, apply_with_spec, random_train_transform


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def list_items(roots, split, transform_name="clean", limit=0):
    """roots: one dir or a list of dirs (for pooled datasets)."""
    if isinstance(roots, str):
        roots = [roots]
    items = []
    for root in roots:
        ds = os.path.basename(os.path.normpath(root))
        for label, y in (("real", 0), ("fake", 1)):
            paths = sorted(glob.glob(os.path.join(root, split, label, "*.png")))
            if limit:
                paths = paths[:limit]
            for p in paths:
                items.append((p, y, f"{ds}/{label}/{_stem(p)}", ""))
    return items


def list_manifest(path, limit=0, exclude_categories=(), stratified_limit=False):
    """Read a normalized CSV manifest as (path, label, stable key) items."""
    base = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("category", "") in exclude_categories:
                continue
            image_path = row["image_path"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(base, image_path)
            label = int(row["label"])
            key = row.get("group_id") or row.get("source") or _stem(image_path)
            mask_path = row.get("mask_path", "")
            if mask_path and not os.path.isabs(mask_path):
                mask_path = os.path.join(base, mask_path)
            items.append((image_path, label, key, mask_path))
    if limit:
        if stratified_limit:
            selected = []
            for label in (0, 1, 2):
                selected.extend([item for item in items if item[1] == label][:limit])
            items = selected
        else:
            items = items[:limit]
    return items


class ImageDataset(Dataset):
    def __init__(self, roots, split, transform_name="clean", train_aug=False,
                 subset=None, val_frac=0.15, seed=0, limit=0, canon=200, size=256,
                 manifest=None, exclude_categories=(), mask_crop_prob=0.5,
                 mask_padding=0.20, stratified_limit=False,
                 return_distortion=False):
        items = (list_manifest(manifest, limit, exclude_categories, stratified_limit)
                 if manifest else list_items(roots, split, transform_name, limit))
        items.sort(key=lambda t: t[2])
        random.Random(seed).shuffle(items)

        if subset in ("train", "val"):
            keys = sorted({k for _, _, k, _ in items})
            random.Random(seed + 1).shuffle(keys)
            n_val = int(len(keys) * val_frac)
            val_keys = set(keys[:n_val])
            items = [it for it in items
                     if (it[2] in val_keys) == (subset == "val")]

        self.items = items
        self.tname = transform_name
        self.train_aug = train_aug
        self.seed = seed
        self.canon = canon
        self.size = size
        self.mask_crop_prob = mask_crop_prob
        self.mask_padding = mask_padding
        self.return_distortion = return_distortion

    def __len__(self):
        return len(self.items)

    def _mask_crop(self, image, mask_path):
        if not mask_path:
            return image, DistortionSpec.clean()
        try:
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L")) > 0
            ys, xs = np.where(mask)
            if not len(xs):
                return image, DistortionSpec.clean()
            left, right = int(xs.min()), int(xs.max()) + 1
            top, bottom = int(ys.min()), int(ys.max()) + 1
            pad_x = max(1, int((right - left) * self.mask_padding))
            pad_y = max(1, int((bottom - top) * self.mask_padding))
            left = max(0, left - pad_x); top = max(0, top - pad_y)
            right = min(image.width, right + pad_x); bottom = min(image.height, bottom + pad_y)
            retained = min((right - left) / max(image.width, 1),
                           (bottom - top) / max(image.height, 1))
            spec = DistortionSpec.clean().mark(
                "crop", crop_loss=(1 - retained) / .55)
            return image.crop((left, top, right, bottom)), spec
        except (OSError, ValueError):
            return image, DistortionSpec.clean()

    def _prep(self, im, idx):
        distortion = DistortionSpec.clean()
        if self.tname != "clean":
            im, named = apply_with_spec(
                im, self.tname,
                rng=random.Random(self.seed + idx * 1000003 + 17),
            )
            distortion.merge(named)
        if self.train_aug:
            rng = random.Random(self.seed + idx * 1000003)
            if rng.random() < 0.8:
                im, sampled = random_train_transform(im, rng)
                distortion.merge(sampled)
        im = im.convert("RGB").resize((self.canon, self.canon), Image.Resampling.BICUBIC)
        im = im.resize((self.size, self.size), Image.Resampling.BICUBIC)
        a = np.asarray(im).astype(np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1)
        if self.train_aug:
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[2])
            if rng.random() < 0.2:
                x = torch.flip(x, dims=[1])
        return x, torch.from_numpy(distortion.target())

    def __getitem__(self, i):
        p, y, key, mask_path = self.items[i]
        im = Image.open(p).convert("RGB")
        crop_spec = DistortionSpec.clean()
        if self.train_aug and int(y) == 2 and self.mask_crop_prob > 0:
            rng = random.Random(self.seed + i * 1000003)
            if rng.random() < self.mask_crop_prob:
                im, crop_spec = self._mask_crop(im, mask_path)
        x, distortion = self._prep(im, i)
        if crop_spec.types["crop"]:
            distortion = torch.maximum(distortion, torch.from_numpy(crop_spec.target()))
        item = (x, torch.tensor(float(y)), key)
        return (*item, distortion) if self.return_distortion else item
