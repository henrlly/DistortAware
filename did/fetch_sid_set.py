"""Sample a balanced real-vs-synthetic subset of SID_Set into the folder layout
the rest of the pipeline expects:  data/sid_set/{train,test}/{real,fake}/*.png

SID_Set (https://huggingface.co/datasets/saberzl/SID_Set) is parquet-packed with
the image bytes inline.  label 0 = real (OpenImages V7), 1 = full synthetic
(Flux.1-dev), 2 = tampered (locally edited).  We map 0 -> real, 1 -> fake and
drop 2 by default (`--include-tampered` folds it into `fake`).

Parquet shards must already be in the HF cache (download them on the login node
with `slurm/dl_sid.py`; parsing needs more RAM than the login node allows, so run
this inside a job).  Images are canonicalised to 200x200 like `fetch_wildfake.py`.
"""
import argparse, glob, io, os, random

import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub.constants import HF_HUB_CACHE

CANON = 200
LABEL_DIR = {0: "real", 1: "fake", 2: "fake"}


def shard_paths(split):
    cache = os.environ.get("HF_HUB_CACHE") or HF_HUB_CACHE
    pat = os.path.join(cache, "datasets--saberzl--SID_Set", "snapshots", "*",
                       "data", f"{split}-*.parquet")
    return sorted(glob.glob(pat))


def iter_rows(paths, want_labels):
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
            for row in batch.to_pylist():
                if row["label"] in want_labels:
                    yield row["label"], row["image"]


def decode(img_field):
    b = img_field["bytes"] if isinstance(img_field, dict) else img_field
    if not b:
        return None
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sid_set")
    ap.add_argument("--split", default="validation", help="SID_Set split to draw from")
    ap.add_argument("--train", type=int, default=300, help="per class")
    ap.add_argument("--test", type=int, default=150, help="per class")
    ap.add_argument("--include-tampered", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    want = {0, 1, 2} if args.include_tampered else {0, 1}
    paths = shard_paths(args.split)
    if not paths:
        raise SystemExit("no SID_Set parquet shards in HF cache; run slurm/dl_sid.py first")
    print(f"{len(paths)} parquet shards")

    per_class = args.train + args.test
    buckets = {"real": [], "fake": []}
    for label, img_field in iter_rows(paths, want):
        d = LABEL_DIR[label]
        if len(buckets[d]) >= per_class:
            if all(len(v) >= per_class for v in buckets.values()):
                break
            continue
        im = decode(img_field)
        if im is None or min(im.size) < CANON - 8:
            continue
        if im.size != (CANON, CANON):
            im = im.resize((CANON, CANON), Image.Resampling.BICUBIC)
        buckets[d].append(im)
        if sum(len(v) for v in buckets.values()) % 200 == 0:
            print("  collected", {k: len(v) for k, v in buckets.items()}, flush=True)

    rng = random.Random(args.seed)
    for d, ims in buckets.items():
        rng.shuffle(ims)
        for split, lo, hi in (("train", 0, args.train), ("test", args.train, per_class)):
            outdir = os.path.join(args.out, split, d)
            os.makedirs(outdir, exist_ok=True)
            for i, im in enumerate(ims[lo:hi]):
                im.save(os.path.join(outdir, f"sid_{d}_{i:05d}.png"))
            print(f"wrote {outdir}: {len(ims[lo:hi])}", flush=True)


if __name__ == "__main__":
    main()
