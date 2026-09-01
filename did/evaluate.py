"""Evaluate the trained DID classifier on clean and transformed test features.

Produces:
  - overall clean accuracy / AUC
  - robustness table (accuracy per transform)
  - error-analysis dump (worst false positives / false negatives on clean)
Writes results/did/metrics.json and results/robustness.csv
"""
import argparse, glob, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FeatureDataset
from model import DIDClassifier
from did import get_device


def run_split(model, cache, tname, device, thr=0.5):
    ds = FeatureDataset(cache, "test", (tname,))
    if len(ds) == 0:
        return None
    dl = DataLoader(ds, batch_size=64, num_workers=0)
    ys, ss, p1s, p2s, paths = [], [], [], [], [p for p, _ in ds.items]
    with torch.no_grad():
        for d1, d2, y in dl:
            l1, l2 = model(d1.to(device), d2.to(device))
            p1, p2 = torch.sigmoid(l1).cpu(), torch.sigmoid(l2).cpu()
            ss.append(0.5 * (p1 + p2)); p1s.append(p1); p2s.append(p2); ys.append(y)
    y = torch.cat(ys).numpy(); s = torch.cat(ss).numpy()
    p1 = torch.cat(p1s).numpy(); p2 = torch.cat(p2s).numpy()
    acc = float(((s > thr) == (y > 0.5)).mean())
    # paper's AND rule: real only if both heads say real; per-head thr t
    t = 1 - np.sqrt(0.5)
    fake_and = (p1 > (1 - t)) & (p2 > (1 - t))
    acc_and = float((fake_and == (y > 0.5)).mean())
    # AUC
    order = np.argsort(s)
    ys_sorted = y[order]
    npos = y.sum(); nneg = len(y) - npos
    auc = float((np.sum(np.where(ys_sorted == 1, np.arange(len(y)), 0)) - npos * (npos - 1) / 2)
                / (npos * nneg)) if npos and nneg else float("nan")
    real_acc = float(((s <= thr)[y == 0]).mean()) if (y == 0).any() else float("nan")
    fake_acc = float(((s > thr)[y == 1]).mean()) if (y == 1).any() else float("nan")
    return dict(transform=tname, n=len(y), acc=acc, acc_and=acc_and, auc=auc,
                real_acc=real_acc, fake_acc=fake_acc), (paths, y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", default="checkpoints/did.pt")
    ap.add_argument("--out", default="results/did")
    ap.add_argument("--backbone", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    thr = float(ckpt.get("threshold", 0.5))
    backbone = args.backbone or ckpt.get("backbone", "resnet18")
    model = DIDClassifier(pretrained=False, backbone=backbone).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"decision threshold = {thr:.3f}")

    transforms_present = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                                 for p in glob.glob(os.path.join(args.cache, "test", "*", "*", "*.npz"))})
    order = ["clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "blur0.5", "blur1.0",
             "blur2.0", "resize0.5", "resize0.25", "noise0.02", "noise0.05",
             "noise0.10", "jitter", "crop80"]
    transforms_present.sort(key=lambda x: order.index(x) if x in order else 99)

    rows = []
    clean_detail = None
    for tname in transforms_present:
        res = run_split(model, args.cache, tname, device, thr)
        if res is None:
            continue
        row, detail = res
        rows.append(row)
        if tname == "clean":
            clean_detail = detail
        print(f"{tname:12s} n={row['n']:4d} acc={row['acc']:.3f} "
              f"auc={row['auc']:.3f} real={row['real_acc']:.3f} fake={row['fake_acc']:.3f}",
              flush=True)

    with open(os.path.join(args.out, "robustness.csv"), "w") as f:
        f.write("transform,n,acc,acc_and,auc,real_acc,fake_acc\n")
        for r in rows:
            f.write(f"{r['transform']},{r['n']},{r['acc']:.4f},{r['acc_and']:.4f},"
                    f"{r['auc']:.4f},{r['real_acc']:.4f},{r['fake_acc']:.4f}\n")

    clean = next((r for r in rows if r["transform"] == "clean"), rows[0])
    transformed = [r for r in rows if r["transform"] != "clean"]
    summary = dict(
        clean_acc=clean["acc"], clean_auc=clean["auc"],
        mean_transformed_acc=float(np.mean([r["acc"] for r in transformed])) if transformed else None,
        min_transformed_acc=float(np.min([r["acc"] for r in transformed])) if transformed else None,
        per_transform={r["transform"]: r["acc"] for r in rows},
    )
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # error analysis on clean
    if clean_detail:
        paths, y, s = clean_detail
        fp = [(paths[i], s[i]) for i in range(len(y)) if y[i] == 0 and s[i] > thr]
        fn = [(paths[i], s[i]) for i in range(len(y)) if y[i] == 1 and s[i] <= thr]
        fp.sort(key=lambda t: -t[1]); fn.sort(key=lambda t: t[1])
        with open(os.path.join(args.out, "error_analysis.json"), "w") as f:
            json.dump(dict(
                n_false_pos=len(fp), n_false_neg=len(fn),
                worst_false_positives=[{"path": p, "score": float(sc)} for p, sc in fp[:15]],
                worst_false_negatives=[{"path": p, "score": float(sc)} for p, sc in fn[:15]],
            ), f, indent=2)
        print(f"\nclean: {len(fp)} false positives, {len(fn)} false negatives")

    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
