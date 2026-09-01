"""Dump DID predictions for harness comparisons with other detectors.

Output: <out> JSON  {threshold, preds: {"<ds>/<label>/<stem>": {label, score, pred}}}
"""
import argparse
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from did.model import DIDClassifier  # noqa: E402
from did.did import get_device  # noqa: E402


def load_npz(p):
    z = np.load(p)
    d1 = torch.from_numpy(z["d1"].astype(np.float32))
    d2 = torch.from_numpy(z["d2"].astype(np.float32))
    if d1.shape[-1] != 256:
        d1 = torch.nn.functional.interpolate(d1[None], size=256, mode="bilinear",
                                             align_corners=False)[0]
        d2 = torch.nn.functional.interpolate(d2[None], size=256, mode="bilinear",
                                             align_corners=False)[0]
    return d1, d2


def ds_of(stem, default):
    return "sid_set" if stem.startswith(("sid_", "sid")) and default == "pooled" else \
        (default if default != "pooled" else "wildfake")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="DID feature cache dir")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ds", default="wildfake", help="namespace for the keys")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    thr = float(ck.get("threshold", 0.5))
    model = DIDClassifier(pretrained=False, backbone=ck.get("backbone", "resnet18")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    items = []
    for label, y in (("real", 0), ("fake", 1)):
        for p in sorted(glob.glob(os.path.join(args.cache, "test", "clean", label, "*.npz"))):
            items.append((p, y, label))
    print(f"{len(items)} clean test features  thr={thr:.3f}", flush=True)

    preds = {}
    B = 64
    with torch.no_grad():
        for i in range(0, len(items), B):
            chunk = items[i:i + B]
            loaded = [load_npz(p) for p, _, _ in chunk]
            d1 = torch.stack([a for a, _ in loaded]).to(device)
            d2 = torch.stack([b for _, b in loaded]).to(device)
            s = model.score(d1, d2).cpu().numpy()
            for j, (p, y, label) in enumerate(chunk):
                stem = os.path.basename(p)[:-4]
                key = f"{ds_of(stem, args.ds)}/{label}/{stem}"
                preds[key] = dict(label=int(y), score=float(s[j]), pred=int(s[j] > thr))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(threshold=thr, ckpt=args.ckpt, preds=preds), f, indent=2)
    acc = np.mean([p["pred"] == p["label"] for p in preds.values()])
    print(f"acc={acc:.4f}  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
