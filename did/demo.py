"""Visual demo: for a handful of images show the original, the first-order (DIRE)
and second-order DID error maps, and the detector's verdict.  Saves a panel PNG
you can screen-record for the demo video.

    python did/demo.py --images data/wildfake/test/real/coco_00000.png \
        data/wildfake/test/fake/ADM_00000.png --out results/did/demo.png
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from did import make_reconstructor, get_device
from model import DIDClassifier


def prep(path, res):
    im = Image.open(path).convert("RGB")
    im = im.resize((200, 200), Image.Resampling.BICUBIC).resize((res, res), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--ckpt", default="checkpoints/did.pt")
    ap.add_argument("--out", default="results/did/demo.png")
    ap.add_argument("--res", type=int, default=192)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    rec = make_reconstructor(ckpt.get("recon", "sd15"), res=args.res,
                             steps=args.steps, device=device)
    clf = DIDClassifier(pretrained=False,
                        backbone=ckpt.get("backbone", "resnet18")).to(device).eval()
    clf.load_state_dict(ckpt["model"])

    x = torch.stack([prep(p, args.res) for p in args.images])
    d1, d2 = rec.did_features(x)
    d1i = torch.nn.functional.interpolate(d1, size=256, mode="bilinear", align_corners=False)
    d2i = torch.nn.functional.interpolate(d2, size=256, mode="bilinear", align_corners=False)
    with torch.no_grad():
        scores = clf.score(d1i.to(device), d2i.to(device)).cpu().numpy()

    n = len(args.images)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, p in enumerate(args.images):
        img = x[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        verdict = "AIGC" if scores[i] > 0.5 else "REAL"
        axes[i, 0].set_title(f"{os.path.basename(p)}\nP(AIGC)={scores[i]:.3f} → {verdict}")
        axes[i, 1].imshow(d1[i].mean(0).numpy(), cmap="magma")
        axes[i, 1].set_title("first-order error |x−x'|")
        axes[i, 2].imshow(d2[i].mean(0).numpy(), cmap="coolwarm")
        axes[i, 2].set_title("second-order  |x−x'|−|x'−x''|")
        for a in axes[i]:
            a.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
