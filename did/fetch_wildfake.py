"""Sample a balanced subset of WildFake via HTTP range reads (no full download).

Images are read in zip order starting from a random offset so the 4 MB read-ahead
buffer in HttpRangeFile amortizes across many small consecutive JPEGs.

WildFake stores every REAL image at 200x200 but FAKE images at 256x256; to stop a
classifier from cheating on that resampling signature we canonicalise BOTH classes
to 200x200 on disk (real: no-op, fake: bicubic downscale).  Train / test slices are
disjoint (opposite ends of the zip order).
"""
import io, os, random, argparse
from PIL import Image
from remote_zip import open_remote_zip, wildfake_url

CANON = 200

REAL_SOURCES = {
    "coco": "Images/Real/coco.zip",
    "imagenet": "Images/Real/imagenet.zip",
    "celebahq": "Images/Real/celebahq.zip",
    "afhq": "Images/Real/afhq.zip",
}
FAKE_SOURCES = {
    "ADM": "Images/Diffusion_based/ADM.zip",
    "DDIM": "Images/Diffusion_based/DDIM.zip",
    "DDPM": "Images/Diffusion_based/DDPM.zip",
    "VQDM": "Images/Diffusion_based/VQDM.zip",
}


def take(zf, names, want, out_dir, sname):
    os.makedirs(out_dir, exist_ok=True)
    got = 0
    for name in names:
        if got >= want:
            break
        try:
            im = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
        except Exception:
            continue
        if min(im.size) < CANON - 8:
            continue
        if im.size != (CANON, CANON):
            im = im.resize((CANON, CANON), Image.Resampling.BICUBIC)
        im.save(os.path.join(out_dir, f"{sname}_{got:05d}.png"))
        got += 1
        if got % 100 == 0:
            print(f"  {sname} -> {out_dir.split('/')[-2]}: {got}/{want}", flush=True)
    print(f"done {sname} {out_dir}: {got}", flush=True)
    return got


def sample_source(zip_path, sname, n_train, n_test, out_root):
    label = "real" if zip_path.startswith("Images/Real") else "fake"
    zf = open_remote_zip(wildfake_url(zip_path))
    names = [x for x in zf.namelist()
             if x.lower().endswith((".jpg", ".jpeg", ".png")) and not x.endswith("/")]
    n = len(names)
    rng = random.Random(hash(sname) & 0xffffffff)
    start = rng.randrange(0, max(1, n - (n_train + n_test) * 4))
    train_names = names[start: start + n_train * 4]
    test_names = names[start + n_train * 4: start + (n_train + n_test) * 4 + n_test * 4]
    take(zf, train_names, n_train, os.path.join(out_root, "train", label), sname)
    take(zf, test_names, n_test, os.path.join(out_root, "test", label), sname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wildfake")
    ap.add_argument("--train", type=int, default=350, help="per source, per class")
    ap.add_argument("--test", type=int, default=175)
    ap.add_argument("--only", default=None, help="comma list of source names")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    for sources in (REAL_SOURCES, FAKE_SOURCES):
        for sname, zpath in sources.items():
            if only and sname not in only:
                continue
            sample_source(zpath, sname, args.train, args.test, args.out)


if __name__ == "__main__":
    main()
