"""Fetch a balanced, non-overlapping SID evaluation manifest.

The SID validation split is used as the source. Existing training images are
excluded again by content hash, and the output contains equal real,
fully-synthetic, and tampered classes with masks where available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from manifest import ImageRecord, load_manifest, write_manifest


REPO = "saberzl/SID_Set"


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(args) -> None:
    from fetch import _load_hf_stream, _save_image

    root = Path(args.output_dir).resolve()
    manifest_path = root / "sid_eval_stratified.csv"
    if manifest_path.is_file() and not args.refresh:
        print(f"fetch cache hit: {manifest_path}")
        return

    train_manifest = root / "train.csv"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")
    train_hashes = {file_hash(record.image_path)
                    for record in load_manifest(train_manifest)
                    if record.source == "sid"}

    destination_root = root / "sid_eval_stratified"
    counts = {"real": 0, "full_synthetic": 0, "tampered": 0}
    wanted = {name: args.per_class for name in counts}
    records: list[ImageRecord] = []
    stream = _load_hf_stream(REPO, "validation", args.seed)
    for item in stream:
        raw = int(item.get("label", 0))
        category = {0: "real", 1: "full_synthetic", 2: "tampered"}.get(raw)
        if category not in wanted or counts[category] >= wanted[category]:
            continue
        image = item["image"]
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if image_hash in train_hashes:
            continue
        index = counts[category]
        counts[category] += 1
        source_id = str(item.get("img_id", item.get("id", index))).replace("/", "_")
        destination = destination_root / category / f"{index:05d}_{source_id}.jpg"
        _save_image(image, destination)
        if file_hash(destination) in train_hashes:
            destination.unlink()
            counts[category] -= 1
            continue
        mask_path = ""
        raw_mask = item.get("mask") or item.get("masks")
        if category == "tampered" and hasattr(raw_mask, "convert"):
            mask = destination_root / "masks" / f"{index:05d}_{source_id}.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            raw_mask.convert("L").save(mask)
            mask_path = str(mask.resolve())
        records.append(ImageRecord(str(destination.resolve()), 0 if category == "real" else 1 if category == "full_synthetic" else 2,
                                    source="sid", category=category,
                                    generator="" if category == "real" else category,
                                    group_id=f"sid_eval:{source_id}", mask_path=mask_path))
        if all(counts[name] >= wanted[name] for name in wanted):
            break

    if counts != wanted:
        raise RuntimeError(f"SID validation quota unavailable: {counts}, wanted {wanted}")
    write_manifest(manifest_path, records)
    (root / "sid_eval_stratified.json").write_text(
        json.dumps({"source": REPO, "split": "validation", "seed": args.seed,
                    "counts": counts, "manifest": str(manifest_path)}, indent=2),
        encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/matched_refactored"))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    fetch(parser.parse_args())
