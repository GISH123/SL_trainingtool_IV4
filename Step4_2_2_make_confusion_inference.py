from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

import configuration as cfg
from class_mapping import class_index_map, load_class_names
from image_utils import list_image_files
from pydealer_detector import do_predict_one_label_original


def run_worker(class_folder: Path, model: Path, class_names_path: Path, output_csv: Path, smoke_max: int = 0):
    class_names = load_class_names(class_names_path)
    mapping = class_index_map(class_names)
    true_class = class_folder.name
    if true_class not in mapping:
        raise ValueError(f"Worker expects canonical class folder, got: {class_folder}")
    paths = list_image_files(class_folder)
    if smoke_max > 0:
        paths = paths[:smoke_max]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["image_path", "true_index", "true_class", "pred_index", "pred_class", "score", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[skip unreadable] {path}")
                continue
            pred_index, score = do_predict_one_label_original(image_bgr, model, class_names_path)
            writer.writerow({
                "image_path": str(path),
                "true_index": mapping[true_class],
                "true_class": true_class,
                "pred_index": pred_index,
                "pred_class": class_names[pred_index],
                "score": score,
                "correct": pred_index == mapping[true_class],
            })
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step4_2 worker: pydealerclient inference for one class.")
    parser.add_argument("--class-folder", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=cfg.LATEST_MODEL_DIR)
    parser.add_argument("--class-names", type=Path, default=cfg.CLASS_NAMES_PATH)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--smoke-max-per-class", type=int, default=cfg.SMOKE_MAX_PER_CLASS)
    args = parser.parse_args()
    count = run_worker(args.class_folder, args.model, args.class_names, args.output_csv, args.smoke_max_per_class)
    print(f"worker complete: {count} -> {args.output_csv}")


if __name__ == "__main__":
    main()
