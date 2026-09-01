"""Evaluate the PatchHead detector: clean accuracy/AUC + the 14-transform
robustness suite, plus a full per-image prediction dump for compare.py.

Writes into <out>/:
  robustness.csv        acc / auc / real_acc / fake_acc per transform
  metrics.json          clean + mean/min transformed accuracy and 3-class summary
  preds_clean.json      {key: {label, score, pred}} for every clean test image
  error_analysis.json   worst false positives / negatives on clean
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ImageDataset
from metrics import roc_auc, threshold_metrics, multiclass_accuracy
from model import get_device, load_detector
from transforms import MAGNITUDE_NAMES, TYPE_NAMES

ORDER = ["clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "blur0.5", "blur1.0",
         "blur2.0", "resize0.5", "resize0.25", "noise0.02", "noise0.05",
         "noise0.10", "jitter", "crop80"]


def positive_score(image_logits, cls_logits):
    image_prob = torch.softmax(image_logits.float(), dim=1)[:, 1:].sum(dim=1)
    cls_prob = torch.softmax(cls_logits.float(), dim=1)[:, 1:].sum(dim=1)
    return 0.5 * (image_prob + cls_prob)


def _extend_keys(keys, batch_keys):
    """Append keys from either the default DataLoader collate or a scalar key."""
    if isinstance(batch_keys, torch.Tensor):
        batch_keys = batch_keys.detach().cpu().tolist()
    elif not isinstance(batch_keys, (list, tuple)):
        batch_keys = [batch_keys]
    keys.extend(batch_keys)


def _distortion_summary(true_values, predicted_values):
    if not len(true_values):
        return {}
    truth = np.concatenate(true_values)
    predicted = np.concatenate(predicted_values)
    type_count = len(TYPE_NAMES)
    type_truth = truth[:, :type_count] >= .5
    type_pred = predicted[:, :type_count] >= .5
    per_type = {}
    for index, name in enumerate(TYPE_NAMES):
        tp = int(np.sum(type_truth[:, index] & type_pred[:, index]))
        fp = int(np.sum(~type_truth[:, index] & type_pred[:, index]))
        fn = int(np.sum(type_truth[:, index] & ~type_pred[:, index]))
        per_type[name] = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        }
    magnitude_type = np.asarray([0, 1, 2, 3, 4, 4, 4, 5])
    magnitude_truth = truth[:, type_count:]
    magnitude_pred = predicted[:, type_count:]
    per_magnitude = {}
    for index, name in enumerate(MAGNITUDE_NAMES):
        present = type_truth[:, magnitude_type[index]]
        per_magnitude[name] = {
            "n": int(present.sum()),
            "mae": float(np.abs(magnitude_truth[present, index] - magnitude_pred[present, index]).mean())
            if present.any() else None,
        }
    return {"types": per_type, "magnitudes": per_magnitude}


@torch.no_grad()
def run_split(model, roots, tname, device, size, thr, limit, workers):
    ds = ImageDataset(roots, "test", tname, train_aug=False, size=size, limit=limit)
    dl = DataLoader(
        ds,
        batch_size=32,
        num_workers=workers,
        pin_memory=device != "cpu",
    )
    ys, ss, keys = [], [], []
    for x, y, k in dl:
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device != "cpu"):
            il, cl, _ = model(x.to(device))
        s = 0.5 * (torch.sigmoid(il.float()) + torch.sigmoid(cl.float()))
        ss.append(s.cpu())
        ys.append(y)
        keys += list(k)
    y = torch.cat(ys).numpy()
    s = torch.cat(ss).numpy()
    pred = s > thr
    acc = float((pred == (y > 0.5)).mean())
    npos, nneg = int(y.sum()), int((y == 0).sum())
    order = np.argsort(s)
    auc = float((np.sum(np.where(y[order] == 1, np.arange(len(y)), 0))
                 - npos * (npos - 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
    real_acc = float((~pred[y == 0]).mean()) if nneg else float("nan")
    fake_acc = float(pred[y == 1].mean()) if npos else float("nan")
    row = dict(transform=tname, n=len(y), acc=acc, auc=auc,
               real_acc=real_acc, fake_acc=fake_acc)
    return row, (keys, y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="wildfake")
    ap.add_argument("--ckpt", default="patchhead/checkpoints/patchhead.pt")
    ap.add_argument("--out", default="results/patchhead/current")
    ap.add_argument("--limit", type=int, default=150,
                    help="per-class cap for the transformed suite (clean uses all)")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader workers; use 0 for portable macOS/interactive evaluation",
    )
    args = ap.parse_args()
    if args.workers < 0:
        ap.error("--workers cannot be negative")
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    model, ckpt = load_detector(args.ckpt, device)
    distortion_mode = args.distortion_mode
    if distortion_mode == "auto":
        distortion_mode = "predicted" if ckpt.get("distortion_aware", False) else "off"
    if distortion_mode != "off" and not ckpt.get("distortion_aware", False):
        raise ValueError("predicted/oracle distortion mode requires a distortion-aware checkpoint")
    thr = float(ckpt.get("threshold", 0.5))
    calibration = ckpt.get("calibration", {})
    thresholds = {"fixed_0.5": 0.5, "calibration_balanced": thr}
    if calibration.get("target_fpr_threshold") is not None:
        thresholds["target_fpr"] = float(calibration["target_fpr_threshold"])
    size = int(ckpt.get("size", 256))
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    roots = [os.path.join(base, "wildfake"), os.path.join(base, "sid_set")] if args.ds == "pooled" else [os.path.join(base, args.ds)]
    print(f"ckpt {args.ckpt}  ds={args.ds}  thr={thr:.3f}  size={size} distortion={distortion_mode}", flush=True)

    rows, clean_detail, details_by_transform = [], None, {}
    for tname in ORDER:
        lim = 0 if tname == "clean" else args.limit
        row, detail = run_split(
            model, roots, tname, device, size, thr, lim, args.workers
        )
        rows.append(row)
        details_by_transform[tname] = detail
        if tname == "clean":
            clean_detail = detail
        print(f"{tname:12s} n={row['n']:4d} acc={row['acc']:.3f} auc={row['auc']:.3f} "
              f"real={row['real_acc']:.3f} fake={row['fake_acc']:.3f} "
              f"base={row['base_accuracy']:.3f}", flush=True)

    with open(os.path.join(args.out, "robustness.csv"), "w") as f:
        f.write("transform,n,acc,auc,real_acc,fake_acc,base_acc,mean_dynamic_threshold\n")
        for r in rows:
            f.write(f"{r['transform']},{r['n']},{r['acc']:.4f},{r['auc']:.4f},"
                    f"{r['real_acc']:.4f},{r['fake_acc']:.4f},{r['base_accuracy']:.4f},"
                    f"{r.get('mean_dynamic_threshold', float('nan')):.4f}\n")

    clean = rows[0]
    tf = [r for r in rows if r["transform"] != "clean"]
    threshold_metrics_by_name = {}
    for name, threshold in thresholds.items():
        threshold_metrics_by_name[name] = threshold_metrics((clean_detail[1] > 0).astype(int), clean_detail[2], threshold)
    threshold_metrics_by_transform = {
        row["transform"]: {
            name: threshold_metrics((details_by_transform[row["transform"]][1] > 0).astype(int), details_by_transform[row["transform"]][2], threshold)
            for name, threshold in thresholds.items()
        }
        for row in rows
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(clean_acc=clean["acc"], clean_auc=clean["auc"],
                       mean_transformed_acc=float(np.mean([r["acc"] for r in tf])),
                       min_transformed_acc=float(np.min([r["acc"] for r in tf])),
                       worst_transform=min(tf, key=lambda r: r["acc"])["transform"],
                       threshold=thr, thresholds=thresholds,
                       distortion_mode=distortion_mode,
                       distortion_estimation={r["transform"]: r["distortion"] for r in rows},
                       mean_dynamic_threshold={r["transform"]: r.get("mean_dynamic_threshold") for r in rows},
                       base_per_transform={r["transform"]: r["base_accuracy"] for r in rows},
                       clean_by_threshold=threshold_metrics_by_name,
                       by_transform_and_threshold=threshold_metrics_by_transform,
                       clean_three_class=multiclass_accuracy(clean_detail[1], clean_detail[3]),
                       per_transform={r["transform"]: r["acc"] for r in rows}), f, indent=2)

    keys, y, s, class_pred, base_s = clean_detail
    preds = {k: dict(label=int(y[i]), class_pred=int(class_pred[i]), score=float(s[i]), pred=int(s[i] >= thr),
                     base_score=float(base_s[i]))
             for i, k in enumerate(keys)}
    with open(os.path.join(args.out, "preds_clean.json"), "w") as f:
        json.dump(dict(threshold=thr, ds=args.ds, arch=ckpt.get("arch"),
                       preds=preds), f, indent=2)

    fp = sorted([(k, s[i]) for i, k in enumerate(keys) if y[i] == 0 and s[i] >= thr],
                key=lambda t: -t[1])
    fn = sorted([(k, s[i]) for i, k in enumerate(keys) if y[i] > 0 and s[i] < thr],
                key=lambda t: t[1])
    with open(os.path.join(args.out, "error_analysis.json"), "w") as f:
        json.dump(dict(n_false_pos=len(fp), n_false_neg=len(fn),
                       worst_false_positives=[{"key": k, "score": float(v)} for k, v in fp[:15]],
                       worst_false_negatives=[{"key": k, "score": float(v)} for k, v in fn[:15]]),
                  f, indent=2)
    print(f"\nclean: {len(fp)} FP, {len(fn)} FN   wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
