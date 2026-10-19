from __future__ import annotations

import argparse
import csv
from pathlib import Path

import configuration as cfg
from image_utils import list_image_files
from inference_engine import InferenceEngine


def predict_single_image(path: Path, engine: InferenceEngine):
    index, score, probabilities = engine.predict_path_tf(path)
    return {
        "image_path": str(path),
        "pred_index": index,
        "pred_class": engine.class_name(index),
        "score": score,
        "probabilities": probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3 single-image/folder inference.")
    parser.add_argument("input", type=Path, help="Image file or folder")
    parser.add_argument("--model", type=Path, default=cfg.LATEST_MODEL_DIR)
    parser.add_argument("--class-names", type=Path, default=cfg.CLASS_NAMES_PATH)
    parser.add_argument("--output", type=Path, default=cfg.REPORT_DIR / "step3_predictions.csv")
    args = parser.parse_args()

    engine = InferenceEngine(args.model, args.class_names)
    paths = [args.input] if args.input.is_file() else list_image_files(args.input)
    rows = []
    for path in paths:
        result = predict_single_image(path, engine)
        rows.append({k: result[k] for k in ("image_path", "pred_index", "pred_class", "score")})
        print(f"{path} -> {result['pred_class']} ({result['score']:.6f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "pred_index", "pred_class", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
