"""Streaming SID-Set access for PyTorch experiments."""

from __future__ import annotations

from datasets import load_dataset


def sid_raw_stream(split: str = "train", shuffle: bool = False, buffer_size: int = 512):
    """Return SID labels unchanged: 0 real, 1 synthetic, 2 tampered."""
    dataset = load_dataset("saberzl/SID_Set", split=split, streaming=True)
    if shuffle:
        dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)
    return dataset


def sid_stream(split: str = "train", shuffle: bool = True, buffer_size: int = 512):
    dataset = sid_raw_stream(split=split, shuffle=False)
    # Streaming shuffle must download/fill its buffer before yielding anything.
    # SID images and masks are large, so keep this modest on a laptop.
    if shuffle:
        dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)
    # Binary task: real versus fully synthetic. Keep tampered images for a later
    # explicitly documented experiment rather than silently changing the label.
    dataset = dataset.filter(lambda row: row["label"] in (0, 1))
    return dataset.map(lambda row: {**row, "label": int(row["label"] == 1)})
