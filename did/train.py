"""Train the two-head DID classifier on cached features.

Model selection and decision-threshold calibration use a held-out slice of the
TRAIN set (never the test set). The chosen threshold is stored in the checkpoint
and used by evaluate.py / infer.py.
"""
import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FeatureDataset
from model import DIDClassifier
from did import get_device


@torch.no_grad()
def infer_scores(model, loader, device):
    model.eval()
    ys, ss = [], []
    for d1, d2, y in loader:
        l1, l2 = model(d1.to(device), d2.to(device))
        s = 0.5 * (torch.sigmoid(l1) + torch.sigmoid(l2))
        ss.append(s.cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ss).numpy()


def best_threshold(y, s):
    """threshold maximising balanced accuracy on (y, s)."""
    best_t, best_b = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = s > t
        tpr = (pred[y == 1]).mean() if (y == 1).any() else 0
        tnr = (~pred[y == 0]).mean() if (y == 0).any() else 0
        b = 0.5 * (tpr + tnr)
        if b > best_b:
            best_b, best_t = b, t
    return float(best_t), float(best_b)


def acc_at(y, s, t):
    return float(((s > t) == (y > 0.5)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-transforms", default="clean")
    ap.add_argument("--out", default="checkpoints/did.pt")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--recon", default="sd15", help="reconstructor used for features (stamped into ckpt)")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    device = get_device()
    tnames = args.train_transforms.split(",")
    tr = FeatureDataset(args.cache, "train", tnames, augment=True, subset="train")
    va = FeatureDataset(args.cache, "train", tnames, augment=False, subset="val")
    te = FeatureDataset(args.cache, "test", ("clean",))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    trl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    val = DataLoader(va, batch_size=64, num_workers=0)
    tel = DataLoader(te, batch_size=64, num_workers=0)

    model = DIDClassifier(pretrained=not args.no_pretrained,
                          backbone=args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = torch.nn.BCEWithLogitsLoss()

    best_val, best_state, best_t = -1.0, None, 0.5
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for d1, d2, y in trl:
            d1, d2, y = d1.to(device), d2.to(device), y.to(device)
            l1, l2 = model(d1, d2)
            loss = lossf(l1, y) + lossf(l2, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()

        yv, sv = infer_scores(model, val, device)
        t, _ = best_threshold(yv, sv)
        va_acc = acc_at(yv, sv, t)
        yt, st = infer_scores(model, tel, device)
        te_acc05 = acc_at(yt, st, 0.5)
        te_acct = acc_at(yt, st, t)
        print(f"ep{ep:02d} loss {tot/len(trl):.4f}  val_acc {va_acc:.4f} (t={t:.2f})  "
              f"test_acc {te_acc05:.4f}@0.5  {te_acct:.4f}@t", flush=True)
        if va_acc >= best_val:
            best_val = va_acc
            best_t = t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save({"model": best_state, "threshold": best_t, "val_acc": float(best_val),
                "backbone": args.backbone, "recon": args.recon},
               args.out)
    # final report with the saved model
    model.load_state_dict(best_state)
    yt, st = infer_scores(model, tel, device)
    print(f"\nSAVED  val_acc={best_val:.4f}  threshold={best_t:.3f}")
    print(f"TEST   acc@0.5={acc_at(yt, st, 0.5):.4f}  acc@t={acc_at(yt, st, best_t):.4f}")


if __name__ == "__main__":
    main()
