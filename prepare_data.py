from __future__ import annotations

import argparse
from pathlib import Path

import configuration as cfg
from Step1_split_dataset import split_dataset
from to_tfrecord import build_tfrecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step1 split and optional TFRecord export.")
    parser.add_argument("--source", type=Path, default=cfg.SOURCE_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT_DATASET_DIR)
    parser.add_argument("--mode", choices=["copy", "move"], default=cfg.COPY_MODE)
    parser.add_argument("--skip-tfrecord", action="store_true")
    args = parser.parse_args()

    split_dataset(
        args.source, args.output, cfg.TRAIN_RATIO, cfg.VALID_RATIO,
        cfg.TEST_RATIO, cfg.RANDOM_SEED, args.mode,
    )
    if not args.skip_tfrecord:
        for split in ("train", "valid", "test"):
            output = cfg.TFRECORD_DIR / f"{split}.tfrecord"
            count = build_tfrecord(args.output / split, output, args.output / "class_names.json")
            print(f"{split}: {count} -> {output}")


if __name__ == "__main__":
    main()
