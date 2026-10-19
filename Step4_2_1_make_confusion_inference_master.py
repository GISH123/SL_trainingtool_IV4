from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import configuration as cfg
from class_mapping import load_class_names
from report_utils import PredictionRecord, finalize_report


def _run_process(script: Path, class_folder: Path, model: Path, class_names: Path, output_csv: Path, smoke: int):
    command = [
        sys.executable, str(script),
        "--class-folder", str(class_folder),
        "--model", str(model),
        "--class-names", str(class_names),
        "--output-csv", str(output_csv),
        "--smoke-max-per-class", str(smoke),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Worker failed for {class_folder}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return class_folder.name, result.stdout


def _read_records(csv_path: Path) -> list[PredictionRecord]:
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append(PredictionRecord(
                image_path=row["image_path"],
                true_index=int(row["true_index"]),
                true_class=row["true_class"],
                pred_index=int(row["pred_index"]),
                pred_class=row["pred_class"],
                score=float(row["score"]),
                correct=str(row["correct"]).lower() in {"true", "1", "yes"},
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Step4_2 master: pydealerclient-path report.")
    parser.add_argument("--dataset", type=Path, default=cfg.TEST_DIR)
    parser.add_argument("--model", type=Path, default=cfg.LATEST_MODEL_DIR)
    parser.add_argument("--class-names", type=Path, default=cfg.CLASS_NAMES_PATH)
    parser.add_argument("--output", type=Path, default=cfg.REPORT_DIR / "step4_2_pydealer")
    parser.add_argument("--workers", type=int, default=cfg.STEP4_WORKERS)
    parser.add_argument("--smoke-max-per-class", type=int, default=cfg.SMOKE_MAX_PER_CLASS)
    args = parser.parse_args()

    class_names = load_class_names(args.class_names)
    partial_dir = args.output.parent / f".{args.output.name}_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("Step4_2_2_make_confusion_inference.py")
    tasks = []
    for class_name in class_names:
        folder = args.dataset / class_name
        if folder.exists():
            tasks.append((folder, partial_dir / f"{class_name}.csv"))

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_process, script, folder, args.model, args.class_names, csv_path, args.smoke_max_per_class)
            for folder, csv_path in tasks
        ]
        for future in as_completed(futures):
            name, stdout = future.result()
            print(f"[{name}] {stdout.strip()}")

    records = []
    for _, csv_path in tasks:
        records.extend(_read_records(csv_path))
    summary = finalize_report(records, class_names, args.output)
    print(summary)


if __name__ == "__main__":
    main()
