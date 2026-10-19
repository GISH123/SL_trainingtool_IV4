from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import configuration as cfg
from class_mapping import canonical_class_name, save_class_names
from image_utils import is_image_file, sha1_file


def _unique_destination(folder: Path, source: Path, digest: str) -> Path:
    candidate = folder / source.name
    if not candidate.exists():
        return candidate
    return folder / f"{source.stem}_{digest[:10]}{source.suffix.lower()}"


def _split_counts(total: int, train_ratio: float, valid_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    n_train = int(total * train_ratio)
    n_valid = int(total * valid_ratio)
    n_test = total - n_train - n_valid
    if total >= 3:
        if n_train == 0:
            n_train = 1
        if n_valid == 0:
            n_valid = 1
        n_test = total - n_train - n_valid
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
    return n_train, n_valid, n_test


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    mode: str,
) -> dict:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset folder not found: {source_dir}")
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0")
    if mode not in {"copy", "move"}:
        raise ValueError("mode must be copy or move")

    by_class: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    seen_hashes: dict[str, Path] = {}
    duplicate_rows = []

    for class_folder in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        canonical = canonical_class_name(class_folder.name)
        for path in sorted(class_folder.rglob("*")):
            if not path.is_file() or not is_image_file(path):
                continue
            digest = sha1_file(path)
            if digest in seen_hashes:
                duplicate_rows.append((str(path), str(seen_hashes[digest]), digest))
                continue
            seen_hashes[digest] = path
            by_class[canonical].append((path, digest))

    if not by_class:
        raise RuntimeError(f"No images found under class folders in {source_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "valid", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    class_names = save_class_names(by_class.keys(), output_dir / "class_names.json")
    rng = random.Random(seed)
    manifest_rows = []
    summary = {"classes": {}, "duplicates_skipped": len(duplicate_rows)}

    for class_name in class_names:
        items = list(by_class[class_name])
        rng.shuffle(items)
        n_train, n_valid, n_test = _split_counts(len(items), train_ratio, valid_ratio)
        ranges = {
            "train": items[:n_train],
            "valid": items[n_train:n_train + n_valid],
            "test": items[n_train + n_valid:n_train + n_valid + n_test],
        }
        summary["classes"][class_name] = {k: len(v) for k, v in ranges.items()}

        for split, split_items in ranges.items():
            dest_dir = output_dir / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for source, digest in split_items:
                destination = _unique_destination(dest_dir, source, digest)
                if mode == "copy":
                    shutil.copy2(source, destination)
                else:
                    shutil.move(str(source), str(destination))
                manifest_rows.append({
                    "split": split,
                    "class_name": class_name,
                    "source_path": str(source),
                    "output_path": str(destination),
                    "sha1": digest,
                })

    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_name", "source_path", "output_path", "sha1"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (output_dir / "duplicates_skipped.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["duplicate_path", "kept_path", "sha1"])
        writer.writerows(duplicate_rows)

    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split class-folder dataset into train/valid/test.")
    parser.add_argument("--source", type=Path, default=cfg.SOURCE_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT_DATASET_DIR)
    parser.add_argument("--train-ratio", type=float, default=cfg.TRAIN_RATIO)
    parser.add_argument("--valid-ratio", type=float, default=cfg.VALID_RATIO)
    parser.add_argument("--test-ratio", type=float, default=cfg.TEST_RATIO)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--mode", choices=["copy", "move"], default=cfg.COPY_MODE)
    args = parser.parse_args()

    summary = split_dataset(
        args.source, args.output, args.train_ratio, args.valid_ratio,
        args.test_ratio, args.seed, args.mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
