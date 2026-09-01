"""Self-contained, cached dataset materialisation for PatchHead.

The fetch command owns the on-disk manifests used by training and evaluation.
Provider caches (Hugging Face, Kaggle, and ModelScope) are reused by their
respective clients; materialised images are additionally guarded by a config
fingerprint so ordinary reruns do not redownload or rewrite the dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import types
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from PIL import Image

from manifest import ImageRecord, manifest_fingerprint, split_records, split_train_only, write_manifest

WILDFake_ID = "hy2628982280/WildFake"
BASE_SOURCES = {
    "sid": ("hf", "saberzl/SID_Set", "train"),
    "hemg": ("hf", "Hemg/AI-Generated-vs-Real-Images-Datasets", "train"),
    "cifake": ("kaggle", "birdy654/cifake-real-and-ai-generated-synthetic-images", "train"),
}
WILDFake_TRAIN = {
    "imagenet": ("Images/Real/imagenet.zip", 0, "real", "imagenet"),
    "celebahq": ("Images/Real/celebahq.zip", 0, "real", "celebahq"),
    "afhq": ("Images/Real/afhq.zip", 0, "real", "afhq"),
    "ADM": ("Images/Diffusion_based/ADM.zip", 1, "full_synthetic", "ADM"),
    "DDIM": ("Images/Diffusion_based/DDIM.zip", 1, "full_synthetic", "DDIM"),
    "DDPM": ("Images/Diffusion_based/DDPM.zip", 1, "full_synthetic", "DDPM"),
    "VQDM": ("Images/Diffusion_based/VQDM.zip", 1, "full_synthetic", "VQDM"),
}
WILDFake_BENCHMARK = {
    "coco": ("label_csv_files/real_coco.csv", "Images/Real/coco.zip", 0, "real", "coco_val2017"),
    "dalle": ("label_csv_files/dalle3.csv", "Images/Diffusion_based/DALLE.zip", 1, "full_synthetic", "dalle_advanced"),
}


def _save_image(image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, format="JPEG", quality=95)


def _load_hf_stream(repo: str, split: str, seed: int):
    """Load an HF stream without importing libtorch on restricted login nodes."""
    try:
        import torch  # noqa: F401
    except Exception:
        # datasets' IterableDataset checks worker state through torch even when
        # no DataLoader is used. Login nodes may have insufficient virtual
        # memory to map libtorch; provide only the API that check requires.
        sys.modules.pop("torch", None)
        torch_stub = types.ModuleType("torch")
        data_stub = types.ModuleType("torch.utils.data")
        data_stub.get_worker_info = lambda: None
        utils_stub = types.ModuleType("torch.utils")
        utils_stub.data = data_stub
        torch_stub.utils = utils_stub
        torch_stub.Tensor = object
        sys.modules["torch"] = torch_stub
    from datasets import load_dataset
    return load_dataset(repo, split=split, streaming=True).shuffle(seed=seed, buffer_size=1000)


def _hf_records(root: Path, source: str, repo: str, split: str, quota: int, seed: int) -> list[ImageRecord]:
    ds = _load_hf_stream(repo, split, seed)
    wanted = {"real": quota, "full_synthetic": quota}
    if source == "sid":
        wanted["tampered"] = quota
    counts: Counter[str] = Counter()
    rows: list[ImageRecord] = []
    for item in ds:
        raw = int(item.get("label", 0))
        if source == "sid":
            category = {0: "real", 1: "full_synthetic", 2: "tampered"}.get(raw)
        else:
            category = "full_synthetic" if raw == 0 else "real" if raw == 1 else None
        if category not in wanted or counts[category] >= wanted[category]:
            continue
        index = counts[category]
        counts[category] += 1
        source_id = str(item.get("img_id", item.get("id", index))).replace("/", "_")
        destination = root / "images" / source / f"{category}_{index:06d}_{source_id}.jpg"
        _save_image(item["image"], destination)
        mask_path = ""
        raw_mask = item.get("mask") or item.get("masks")
        if source == "sid" and hasattr(raw_mask, "convert"):
            mask = root / "masks" / source / f"{category}_{index:06d}_{source_id}.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            raw_mask.convert("L").save(mask)
            mask_path = str(mask.resolve())
        output_label = {"real": 0, "full_synthetic": 1, "tampered": 2}[category]
        rows.append(ImageRecord(str(destination.resolve()), output_label,
                                source=source, category=category,
                                generator="" if category == "real" else category,
                                group_id=f"{source}:{source_id}", mask_path=mask_path))
        if all(counts[key] >= value for key, value in wanted.items() if value):
            break
    if any(counts[key] < value for key, value in wanted.items()):
        raise RuntimeError(f"{source} quota unavailable: {dict(counts)}")
    return rows


def _cifake_records(root: Path, quota: int, seed: int) -> list[ImageRecord]:
    import kagglehub

    downloaded = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
    files: dict[str, list[Path]] = {"real": [], "full_synthetic": []}
    for path in downloaded.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        split_names = {part.lower() for part in path.parts}
        if "train" not in split_names:
            continue
        category = "real" if path.parent.name.lower() == "real" else "full_synthetic" if path.parent.name.lower() in {"fake", "ai", "synthetic"} else None
        if category:
            files[category].append(path)
    rows = []
    for category in files:
        selected = sorted(files[category], key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).digest())[:quota]
        if len(selected) < quota:
            raise RuntimeError(f"CIFAKE {category}: found {len(selected)}, need {quota}")
        for index, path in enumerate(selected):
            destination = root / "images" / "cifake" / f"{category}_{index:06d}.jpg"
            with Image.open(path) as image:
                _save_image(image, destination)
            rows.append(ImageRecord(str(destination.resolve()), 0 if category == "real" else 1,
                                    source="cifake", category=category,
                                    generator="stable_diffusion_v1_4" if category != "real" else "",
                                    group_id=f"cifake:{path.name}"))
    return rows


def _modelscope_file(file_path: str) -> Path:
    from modelscope.hub.file_download import dataset_file_download
    return Path(dataset_file_download(dataset_id=WILDFake_ID, file_path=file_path))


def _archive_records(root: Path, source: str, archive_path: str, label: int, category: str,
                     generator: str, quota: int, seed: int) -> list[ImageRecord]:
    archive_path = _modelscope_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted((item.filename for item in archive.infolist()
                        if not item.is_dir() and Path(item.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}),
                       key=lambda name: hashlib.sha256(f"{seed}:{source}:{name}".encode()).digest())
        rows = []
        for index, name in enumerate(names[:quota]):
            destination = root / "images" / "wildfake" / source / f"{index:06d}.jpg"
            with archive.open(name) as handle, Image.open(handle) as image:
                _save_image(image, destination)
            rows.append(ImageRecord(str(destination.resolve()), label, source=f"wildfake_{source}",
                                    category=category, generator=generator,
                                    group_id=f"wildfake:{source}:{name}"))
    if len(rows) < quota:
        raise RuntimeError(f"WildFake {source}: found {len(rows)}, need {quota}")
    return rows


def _benchmark_records(root: Path, name: str, count: int, seed: int) -> list[ImageRecord]:
    metadata_path, archive_path, label, category, generator = WILDFake_BENCHMARK[name]
    metadata = _modelscope_file(metadata_path)
    selected = []
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if name == "coco":
                match = "/val2017/" in str(row.get("Image_path", "")) and int(row.get("IsFake", 1)) == 0
            else:
                match = str(row.get("Architecture", "")).upper() == "DALLE" and int(row.get("IsAdvanced", 0)) == 1 and int(row.get("IsFake", 0)) == 1
            if match:
                selected.append(row)
    selected = sorted(selected, key=lambda row: hashlib.sha256(f"{seed}:{name}:{row['Image_path']}".encode()).digest())[:count]
    archive = _modelscope_file(archive_path)
    with zipfile.ZipFile(archive) as zf:
        members = {item.filename.lstrip("./"): item for item in zf.infolist() if not item.is_dir()}
        rows = []
        for index, row in enumerate(selected):
            logical = str(row["Image_path"]).lstrip("./")
            member = members.get(logical)
            if member is None:
                suffix = "/".join(logical.split("/")[1:])
                matches = [item for item in members.values() if item.filename.lstrip("./").endswith(suffix)]
                if len(matches) != 1:
                    raise RuntimeError(f"Could not locate unique WildFake member: {logical}")
                member = matches[0]
            destination = root / "wildfake_benchmark" / f"{name}_{index:05d}.jpg"
            with zf.open(member) as handle, Image.open(handle) as image:
                _save_image(image, destination)
            rows.append(ImageRecord(str(destination.resolve()), label, source=f"wildfake_{name}",
                                    category=category, generator=generator,
                                    group_id=f"wildfake:{name}:{logical}"))
    if len(rows) < count:
        raise RuntimeError(f"WildFake benchmark {name}: found {len(rows)}, need {count}")
    return rows


def fetch(args) -> None:
    root = Path(args.output_dir).resolve()
    config = {"seed": args.seed, "base_quota": args.base_quota,
              "wildfake_quota": args.wildfake_quota, "benchmark_count": args.benchmark_count,
              "wildfake_train_sources": sorted(WILDFake_TRAIN)}
    config_path = root / "fetch_config.json"
    manifests = [root / name for name in ("train.csv", "validation.csv", "calibration.csv", "matched_test.csv", "wildfake_benchmark.csv", "sid_eval_200.csv")]
    if not args.refresh and config_path.is_file() and json.loads(config_path.read_text()) == config and all(path.is_file() for path in manifests):
        print(f"fetch cache hit: {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    base: list[ImageRecord] = []
    base += _hf_records(root, "sid", *BASE_SOURCES["sid"][1:], args.base_quota, args.seed)
    base += _hf_records(root, "hemg", *BASE_SOURCES["hemg"][1:], args.base_quota, args.seed)
    base += _cifake_records(root, args.base_quota, args.seed)
    extra = [_archive_records(root, name, archive, label, category, generator, args.wildfake_quota, args.seed)
             for name, (archive, label, category, generator) in WILDFake_TRAIN.items()]
    def group_by_bytes(records):
        return [replace(record, group_id=f"byte:{hashlib.sha256(Path(record.image_path).read_bytes()).hexdigest()}")
                for record in records]
    base = group_by_bytes(base)
    extra = [group_by_bytes(records) for records in extra]
    base_splits = split_records(base, seed=args.seed)
    extra_splits = split_train_only([record for group in extra for record in group], seed=args.seed + 1)
    write_manifest(root / "train.csv", base_splits["train"] + extra_splits["train"])
    write_manifest(root / "validation.csv", base_splits["validation"] + extra_splits["validation"])
    write_manifest(root / "calibration.csv", base_splits["calibration"] + extra_splits["calibration"])
    write_manifest(root / "matched_test.csv", base_splits["test"])
    benchmark = _benchmark_records(root, "coco", args.benchmark_count, args.seed) + _benchmark_records(root, "dalle", args.benchmark_count, args.seed)
    write_manifest(root / "wildfake_benchmark.csv", benchmark)
    sid = [record for record in base_splits["test"] if record.source == "sid"]
    selected = random.Random(args.seed).sample(sid, min(args.sid_count, len(sid)))
    write_manifest(root / "sid_eval_200.csv", selected)
    report = {"config": config, "counts": {name: len(load) for name, load in {
        "train": base_splits["train"] + extra_splits["train"],
        "validation": base_splits["validation"] + extra_splits["validation"],
        "calibration": base_splits["calibration"] + extra_splits["calibration"],
        "matched_test": base_splits["test"], "wildfake_benchmark": benchmark,
        "sid_eval_200": selected}.items()}, "sources": dict(Counter(record.source for record in base + [r for group in extra for r in group])),
              "manifest_fingerprint": manifest_fingerprint(benchmark)}
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (root / "dataset_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/matched_refactored"))
    parser.add_argument("--base-quota", type=int, default=1250)
    parser.add_argument("--wildfake-quota", type=int, default=1000)
    parser.add_argument("--benchmark-count", type=int, default=500)
    parser.add_argument("--sid-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    fetch(parser.parse_args())


if __name__ == "__main__":
    main()
