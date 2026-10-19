from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_predictions(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["image_path"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predictions from two Step4 report folders.")
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/step4_comparison.csv"))
    args = parser.parse_args()

    a = read_predictions(args.report_a / "predictions.csv")
    b = read_predictions(args.report_b / "predictions.csv")
    common = sorted(set(a) & set(b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "true_class", "a_pred", "a_score", "b_pred", "b_score", "same_prediction"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in common:
            writer.writerow({
                "image_path": path,
                "true_class": a[path]["true_class"],
                "a_pred": a[path]["pred_class"],
                "a_score": a[path]["score"],
                "b_pred": b[path]["pred_class"],
                "b_score": b[path]["score"],
                "same_prediction": a[path]["pred_index"] == b[path]["pred_index"],
            })
    print(f"Compared {len(common)} images -> {args.output}")


if __name__ == "__main__":
    main()
