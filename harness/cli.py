"""Independent dataset, training, and model-evaluation harness."""
from __future__ import annotations

import argparse

from . import evaluate, fetch, train
from .common import TRANSFORMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_parser = sub.add_parser("fetch")
    fetch.fetch_parser(fetch_parser)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--data-dir", required=True)
    eval_parser.add_argument("--manifest")
    eval_parser.add_argument("--models", default="physics,patchhead_baseline,patchhead_distortion_aware")
    eval_parser.add_argument("--baseline-checkpoint")
    eval_parser.add_argument("--aware-checkpoint")
    eval_parser.add_argument("--filter-checkpoint")
    eval_parser.add_argument("--did-checkpoint")
    eval_parser.add_argument("--did-reconstructor", default="sd15")
    eval_parser.add_argument("--did-resolution", type=int, default=256)
    eval_parser.add_argument("--did-steps", type=int, default=10)
    eval_parser.add_argument("--did-batch-size", type=int, default=32)
    eval_parser.add_argument("--transforms", default=",".join(TRANSFORMS))
    eval_parser.add_argument("--output-dir", default="results/harness/current")
    eval_parser.add_argument("--auto-proposals", action="store_true")
    train_parser = sub.add_parser("train-patchhead")
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--mode", choices=("baseline", "distortion_aware", "both"), default="both")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--bs", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--output-dir", default="runs/quick_training")
    train_parser.add_argument("--init-checkpoint")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch.fetch(args)
    elif args.command == "evaluate":
        evaluate.evaluate(args)
    else:
        train.train(args)


if __name__ == "__main__":
    main()
