"""Compute DID reconstruction features (d1, d2) for an image folder tree and cache
them as .npz (float16).  Layout expected:  <root>/<split>/<label>/<name>.png

Usage:
  python extract_features.py --root data/wildfake --split train --out cache/wildfake
  python extract_features.py --root data/wildfake --split test  --out cache/wildfake --transforms all
"""
import argparse, os, glob, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

import random as _random

from did import make_reconstructor
from transforms import TRANSFORMS, random_transform

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff")


def to_tensor(im, res):
    im = im.convert("RGB").resize((200, 200), Image.Resampling.BICUBIC)
    im = im.resize((res, res), Image.Resampling.BICUBIC)   # shared 200->res upsample
    a = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--transforms", default="clean",
                    help="'clean', 'all', or comma list of transform names")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recon", default="sd15", help="sd15 | sana16")
    args = ap.parse_args()

    if args.transforms == "all":
        tnames = list(TRANSFORMS)
    else:
        tnames = args.transforms.split(",")

    rec = make_reconstructor(args.recon, res=args.res, steps=args.steps)
    print("recon", args.recon, "device", rec.device, "dtype", rec.dtype)

    for label in ("real", "fake"):
        paths = sorted(path for pattern in IMAGE_EXTENSIONS
                       for path in glob.glob(os.path.join(args.root, args.split, label, pattern)))
        if args.limit:
            paths = paths[: args.limit]
        for tname in tnames:
            is_rand = tname.startswith("randaug")
            tfn = (None if is_rand else TRANSFORMS[tname])
            outdir = os.path.join(args.out, args.split, tname, label)
            os.makedirs(outdir, exist_ok=True)
            todo = [p for p in paths
                    if not os.path.exists(os.path.join(outdir, os.path.basename(p)[:-4] + ".npz"))]
            print(f"{label}/{tname}: {len(todo)}/{len(paths)} to do", flush=True)

            def _load(p):
                im = Image.open(p).convert("RGB")
                if is_rand:
                    im = random_transform(im, _random.Random(hash((tname, p)) & 0xffffffff))
                else:
                    im = tfn(im)
                return to_tensor(im, args.res)

            pool = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8)))
            for i in range(0, len(todo), args.batch):
                chunk = todo[i: i + args.batch]
                xs = list(pool.map(_load, chunk))
                x = torch.stack(xs)
                t0 = time.time()
                d1, d2 = rec.did_features(x)
                dt = time.time() - t0
                for j, p in enumerate(chunk):
                    name = os.path.basename(p)[:-4]
                    np.savez(
                        os.path.join(outdir, name + ".npz"),
                        d1=d1[j].numpy().astype(np.float16),
                        d2=d2[j].numpy().astype(np.float16))
                print(f"  {i+len(chunk)}/{len(todo)}  ({dt/len(chunk):.2f}s/img)", flush=True)


if __name__ == "__main__":
    main()
