"""Dataset over cached DID feature .npz files produced by extract_features.py."""
import glob, os, random
import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, cache_root, split, transform_names=("clean",), augment=False,
                 subset=None, val_frac=0.15, seed=0):
        """subset: None (all) | 'train' | 'val' — a deterministic per-image split so
        model selection / threshold calibration never touch the held-out test set."""
        items = []
        for tname in transform_names:
            for label, y in (("real", 0), ("fake", 1)):
                d = os.path.join(cache_root, split, tname, label)
                for p in glob.glob(os.path.join(d, "*.npz")):
                    items.append((p, y))
        random.Random(seed).shuffle(items)

        if subset in ("train", "val"):
            # split by the image stem so all transform variants of an image stay
            # on the same side of the split
            stems = sorted({os.path.basename(p)[:-4].replace("randaug1_", "")
                            .replace("randaug2_", "") for p, _ in items})
            random.Random(seed + 1).shuffle(stems)
            n_val = int(len(stems) * val_frac)
            val_stems = set(stems[:n_val])
            keep = []
            for p, y in items:
                stem = os.path.basename(p)[:-4].replace("randaug1_", "").replace("randaug2_", "")
                in_val = stem in val_stems
                if (subset == "val") == in_val:
                    keep.append((p, y))
            items = keep
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _aug(self, d1, d2):
        if random.random() < 0.5:
            d1 = d1[:, :, ::-1].copy(); d2 = d2[:, :, ::-1].copy()
        if random.random() < 0.5:
            d1 = d1[:, ::-1, :].copy(); d2 = d2[:, ::-1, :].copy()
        if random.random() < 0.4:
            s = random.uniform(0.75, 1.0)
            h = int(d1.shape[1] * s)
            top = random.randint(0, d1.shape[1] - h)
            left = random.randint(0, d1.shape[2] - h)
            d1 = d1[:, top:top + h, left:left + h]
            d2 = d2[:, top:top + h, left:left + h]
        if random.random() < 0.3:  # sensor-noise-like jitter on the error maps
            n = np.random.normal(0, random.uniform(0.005, 0.02), d1.shape).astype(np.float32)
            d1 = d1 + n; d2 = d2 + n
        if random.random() < 0.25:  # random erasing
            h = d1.shape[1]
            eh = random.randint(h // 8, h // 3)
            ew = random.randint(h // 8, h // 3)
            ty = random.randint(0, h - eh); tx = random.randint(0, d1.shape[2] - ew)
            d1 = d1.copy(); d2 = d2.copy()
            d1[:, ty:ty + eh, tx:tx + ew] = 0; d2[:, ty:ty + eh, tx:tx + ew] = 0
        return d1, d2

    def __getitem__(self, i):
        p, y = self.items[i]
        z = np.load(p)
        d1 = z["d1"].astype(np.float32)
        d2 = z["d2"].astype(np.float32)
        if self.augment:
            d1, d2 = self._aug(d1, d2)
        d1 = torch.from_numpy(np.ascontiguousarray(d1))
        d2 = torch.from_numpy(np.ascontiguousarray(d2))
        if d1.shape[-1] != 256:
            d1 = torch.nn.functional.interpolate(d1[None], size=256, mode="bilinear",
                                                 align_corners=False)[0]
            d2 = torch.nn.functional.interpolate(d2[None], size=256, mode="bilinear",
                                                 align_corners=False)[0]
        return d1, d2, torch.tensor(float(y))
