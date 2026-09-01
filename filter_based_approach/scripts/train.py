"""Train the residual-aware segmentation/classification network on SID."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ai_detection.data import sid_raw_stream
from ai_detection.model import CLASS_NAMES, ResidualUNet, dice_loss, image_tensor, target_mask


def load_balanced(split: str, per_class: int, size: int):
    images, masks, labels, tampered = [], [], [], []
    counts = [0, 0, 0]
    for sample in sid_raw_stream(split, shuffle=False):
        label = int(sample["label"])
        if label not in range(3) or counts[label] >= per_class:
            continue
        images.append(image_tensor(sample["image"], size))
        masks.append(target_mask(sample["mask"], label, size))
        labels.append(label)
        tampered.append(label == 2)
        counts[label] += 1
        if sum(counts) % 50 == 0:
            print(f"loaded {split}: {counts}", flush=True)
        if min(counts) == per_class:
            break
    return torch.stack(images), torch.stack(masks), torch.tensor(labels), torch.tensor(tampered)


@torch.no_grad()
def evaluate(model, loader, device):
    confusion = torch.zeros(3, 3, dtype=torch.int64)
    intersection = union = 0.0
    for images, masks, labels, tampered in loader:
        mask_logits, class_logits = model(images.to(device))
        predictions = class_logits.argmax(1).cpu()
        for truth, prediction in zip(labels, predictions):
            confusion[truth, prediction] += 1
        predicted_masks = torch.sigmoid(mask_logits).cpu() >= 0.5
        targets = masks >= 0.5
        # IoU is reported on tampered samples; all-zero/all-one masks are trivial.
        selected = tampered
        intersection += float((predicted_masks[selected] & targets[selected]).sum())
        union += float((predicted_masks[selected] | targets[selected]).sum())
    return confusion, intersection / max(union, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-class", type=int, default=300)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("models/mask_classifier.pt"))
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    train = load_balanced("train", args.train_per_class, args.size)
    validation = load_balanced("validation", args.validation_per_class, args.size)
    train_loader = DataLoader(TensorDataset(*train), batch_size=12, shuffle=True)
    validation_loader = DataLoader(TensorDataset(*validation), batch_size=16)
    model = ResidualUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state, best_score = None, -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, masks, labels, _tampered in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            if torch.rand(()) < 0.5:
                images, masks = images.flip(-1), masks.flip(-1)
            mask_logits, class_logits = model(images)
            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, masks) + dice_loss(mask_logits, masks)
            class_loss = F.cross_entropy(class_logits, labels)
            loss = mask_loss + class_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(labels)
        model.eval()
        confusion, iou = evaluate(model, validation_loader, device)
        accuracy = float(confusion.diag().sum() / confusion.sum())
        score = accuracy + iou
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
        print(f"epoch={epoch:02d} loss={total/len(train[2]):.4f} val_accuracy={accuracy:.3f} tamper_iou={iou:.3f}")

    model.load_state_dict(best_state)
    confusion, iou = evaluate(model, validation_loader, device)
    print("validation confusion (rows=true):\n", confusion.numpy())
    print(f"validation tamper IoU: {iou:.3f}")
    torch.save({"model_state": model.cpu().state_dict(), "size": args.size, "class_names": CLASS_NAMES}, args.output)
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
