"""Train the PatchHead detector (LoRA + spatial head on a frozen DINOv3 L/16).

Same protocol as ../did/train.py so the two detectors are comparable:
  - model selection + decision-threshold calibration on a held-out slice of the
    TRAIN set (the test set is never touched during training),
  - threshold = balanced-accuracy-optimal on that slice, stored in the ckpt,
  - final binary score = mean of the patch-pooled and CLS-head probabilities
    for classes synthetic and tampered.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ImageDataset
from metrics import balanced_threshold, target_fpr_threshold, threshold_metrics
from model import PatchHeadDetector, get_device
from transforms import MAGNITUDE_NAMES, TYPE_NAMES

MAGNITUDE_TYPE_INDEX = torch.tensor([0, 1, 2, 3, 4, 4, 4, 5])


def resolve_roots(ds):
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    if ds == "pooled":
        return [os.path.join(base, "wildfake"), os.path.join(base, "sid_set")]
    return [os.path.join(base, ds)]


@torch.no_grad()
def infer_scores(model, loader, device, distortion_aware=False):
    model.eval()
    ys, ss = [], []
    for batch in loader:
        x, y = batch[:2]
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device != "cpu"):
            if distortion_aware:
                _, _, _, aux = model.forward_distortion_aware(x.to(device))
                s = aux["corrected_score"]
            else:
                il, cl, _ = model(x.to(device))
                s = 0.5 * (torch.softmax(il.float(), dim=1)[:, 1:].sum(dim=1) +
                           torch.softmax(cl.float(), dim=1)[:, 1:].sum(dim=1))
        ss.append(s.cpu())
        ys.append(y)
    return (torch.cat(ys).numpy().reshape(-1) > 0).astype(int), torch.cat(ss).numpy().reshape(-1)


def best_threshold(y, s):
    """Threshold maximising balanced accuracy on (y, s).  When several
    thresholds tie (common once val accuracy saturates) take the midpoint of the
    optimal plateau -- more stable on the test set than the first tying value."""
    return balanced_threshold(y, s)


def acc_at(y, s, t):
    return float(((s >= t) == (y > 0.5)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="wildfake", help="wildfake | sid_set | pooled")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lambda-cls", type=float, default=0.5)
    ap.add_argument("--lambda-patch", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--train-aug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default="patchhead/checkpoints/patchhead.pt")
    ap.add_argument("--init-checkpoint", help="Optional checkpoint used to initialize shared trainable weights")
    ap.add_argument("--manifest-dir", help="Data directory containing train.csv, validation.csv, calibration.csv, and test manifest")
    ap.add_argument("--calibration-manifest", help="Optional explicit calibration manifest")
    ap.add_argument("--target-fpr", type=float, default=.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask-crop-prob", type=float, default=.5)
    ap.add_argument("--mask-padding", type=float, default=.20)
    ap.add_argument("--distortion-aware", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lambda-distortion", type=float, default=.35)
    ap.add_argument("--distortion-pos-weight", type=float, default=5.0,
                    help="Compensate for sparse multi-label distortion targets")
    ap.add_argument("--lambda-threshold", type=float, default=.5)
    ap.add_argument("--oracle-mix-start", type=float, default=.75,
                    help="Initial fraction of known augmentation metadata used for conditioning")
    ap.add_argument("--oracle-mix-end", type=float, default=.15,
                    help="Final metadata fraction; inference always uses predictions only")
    args = ap.parse_args()

    device = get_device()
    print(f"device={device}  cuda_available={torch.cuda.is_available()}", flush=True)
    roots = resolve_roots(args.ds)
    tr = ImageDataset(roots, "train", "clean", train_aug=args.train_aug,
                      subset=None if args.manifest_dir else "train", size=args.size,
                      manifest=os.path.join(args.manifest_dir, "train.csv") if args.manifest_dir else None,
                      seed=args.seed, mask_crop_prob=args.mask_crop_prob,
                      mask_padding=args.mask_padding,
                      return_distortion=args.distortion_aware)
    va = ImageDataset(roots, "train", "clean", train_aug=False,
                      subset=None if args.manifest_dir else "val", size=args.size,
                        manifest=os.path.join(args.manifest_dir, "validation.csv") if args.manifest_dir else None,
                        return_distortion=args.distortion_aware)
    cal = ImageDataset(roots, "train", "clean", train_aug=False, size=args.size,
                       manifest=args.calibration_manifest or (os.path.join(args.manifest_dir, "calibration.csv") if args.manifest_dir else None),
                       return_distortion=args.distortion_aware) if (args.calibration_manifest or args.manifest_dir) else None
    test_manifest = None
    if args.manifest_dir:
        candidate = os.path.join(args.manifest_dir, "matched_test.csv")
        test_manifest = candidate if os.path.isfile(candidate) else os.path.join(args.manifest_dir, "test.csv")
    te = ImageDataset(roots, "test", "clean", train_aug=False, size=args.size,
                      manifest=test_manifest, return_distortion=args.distortion_aware)
    print(f"ds={args.ds}  train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)

    trl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=8,
                     drop_last=True, pin_memory=True)
    val = DataLoader(va, batch_size=32, num_workers=8, pin_memory=True)
    tel = DataLoader(te, batch_size=32, num_workers=8, pin_memory=True)
    call = DataLoader(cal, batch_size=32, num_workers=8, pin_memory=True) if cal is not None else None

    model = PatchHeadDetector(lora_r=args.lora_r).to(device)
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(init["model"], strict=False)
        allowed_missing = ("distortion_head.", "threshold_adapter.")
        frozen = {name for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        unexpected_trainable = [name for name in unexpected
                                if name not in ("mean", "std")]
        unexpected_missing = [name for name in missing
                              if name not in ("mean", "std")
                              and name not in frozen
                              and not name.startswith(allowed_missing)]
        if unexpected_trainable or unexpected_missing:
            raise ValueError(
                f"incompatible init checkpoint: unexpected={unexpected_trainable[:5]} "
                f"missing={unexpected_missing[:5]}"
            )
        print(f"initialized from {args.init_checkpoint}", flush=True)
    pc = model.param_counts()
    print(f"params: total={pc['total']/1e6:.1f}M  trainable={pc['trainable']/1e6:.3f}M "
          f"(lora {pc['lora']/1e6:.3f}M + head {pc['head']/1e6:.3f}M, "
          f"{pc['n_lora_adapters']} adapters)", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ce = torch.nn.CrossEntropyLoss()
    binary_bce = torch.nn.BCEWithLogitsLoss()

    best_val, best_state, best_t = -1.0, None, 0.5
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        oracle_mix = args.oracle_mix_start + (args.oracle_mix_end - args.oracle_mix_start) * ep / max(args.epochs - 1, 1)
        distortion_loss_total = threshold_loss_total = 0.0
        for batch in trl:
            x, y = batch[:2]
            distortion = batch[3] if args.distortion_aware else None
            x, y = x.to(device), y.long().to(device)
            if distortion is not None:
                distortion = distortion.to(device)
            with torch.autocast(device_type=device.split(":")[0],
                                dtype=torch.bfloat16, enabled=device != "cpu"):
                if args.distortion_aware:
                    il, cl, pl, aux = model.forward_distortion_aware(
                        x, oracle_distortion=distortion, oracle_mix=oracle_mix)
                else:
                    il, cl, pl = model(x)
                loss = ce(il, y) + args.lambda_cls * ce(cl, y)
                if args.lambda_patch:
                    yp = y.view(-1, 1, 1).expand(pl.shape[0], pl.shape[2], pl.shape[3])
                    loss = loss + args.lambda_patch * ce(pl, yp)
                if args.distortion_aware:
                    type_count = len(TYPE_NAMES)
                    type_target = distortion[:, :type_count]
                    magnitude_target = distortion[:, type_count:]
                    type_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        aux["type_logits"], type_target,
                        pos_weight=torch.full((type_count,), args.distortion_pos_weight,
                                              device=device))
                    magnitude_error = torch.nn.functional.smooth_l1_loss(
                        torch.sigmoid(aux["magnitude_logits"]), magnitude_target,
                        reduction="none")
                    magnitude_mask = type_target[:, MAGNITUDE_TYPE_INDEX.to(device)]
                    magnitude_loss = (magnitude_error * magnitude_mask).sum() / magnitude_mask.sum().clamp_min(1)
                    distortion_loss = type_loss + magnitude_loss
                    threshold_loss = binary_bce(aux["corrected_logit"], (y > 0).float())
                    loss = (loss + args.lambda_distortion * distortion_loss
                            + args.lambda_threshold * threshold_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            tot += loss.item()
            if args.distortion_aware:
                distortion_loss_total += distortion_loss.item()
                threshold_loss_total += threshold_loss.item()
        sched.step()

        yv, sv = infer_scores(model, val, device, args.distortion_aware)
        t, _ = best_threshold(yv, sv)
        va_acc = acc_at(yv, sv, t)
        extra = (f" dist={distortion_loss_total/len(trl):.3f} "
                 f"threshold={threshold_loss_total/len(trl):.3f} oracle_mix={oracle_mix:.2f}") if args.distortion_aware else ""
        print(f"ep{ep:02d} loss {tot/len(trl):.4f}{extra}  val_acc {va_acc:.4f} (t={t:.2f})  "
              "checkpoint selection uses validation only",
              flush=True)
        if va_acc >= best_val:
            best_val, best_t = va_acc, t
            best_state = trainable_state_dict(model)

    model.load_state_dict(best_state, strict=False)
    if call is not None:
        yc, sc = infer_scores(model, call, device, args.distortion_aware)
        best_t, calibration_balanced = balanced_threshold(yc, sc)
        fpr_t = target_fpr_threshold(yc, sc, args.target_fpr)
        calibration = {"balanced_threshold": best_t, "balanced_accuracy": calibration_balanced,
                       "target_fpr": args.target_fpr, "target_fpr_threshold": fpr_t,
                       "balanced_metrics": threshold_metrics(yc, sc, best_t),
                       "target_fpr_metrics": threshold_metrics(yc, sc, fpr_t)}
    else:
        calibration = {"balanced_threshold": best_t}
    torch.save({"model": best_state, "threshold": best_t, "val_acc": float(best_val),
                "calibration": calibration, "num_classes": 3,
                "lora_r": args.lora_r, "size": args.size, "ds": args.ds,
                "arch": "patchhead-dinov3-vitl16", "distortion_aware": args.distortion_aware,
                "distortion_types": list(TYPE_NAMES),
                "distortion_magnitudes": list(MAGNITUDE_NAMES)}, args.out)
    yt, st = infer_scores(model, tel, device, args.distortion_aware)
    print(f"\nSAVED {args.out}  val_acc={best_val:.4f}  threshold={best_t:.3f}")
    print(f"TEST  acc@0.5={acc_at(yt, st, 0.5):.4f}  acc@t={acc_at(yt, st, best_t):.4f}")


def trainable_state_dict(model):
    """Store only trainable tensors; reload the frozen DINOv3 backbone at eval."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    keep = {k: v.cpu().clone() for k, v in sd.items() if k in trainable}
    for b in ("mean", "std"):
        keep[b] = sd[b].cpu().clone()
    return keep


if __name__ == "__main__":
    main()
