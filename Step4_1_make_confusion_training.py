from __future__ import annotations

import argparse
from pathlib import Path

import configuration as cfg
from class_mapping import load_class_names
from inference_engine import InferenceEngine
from report_utils import PredictionRecord, collect_evaluation_samples, finalize_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Step4_1: direct SavedModel forward accuracy report.")
    parser.add_argument("--dataset", type=Path, default=cfg.TEST_DIR)
    parser.add_argument("--model", type=Path, default=cfg.LATEST_MODEL_DIR)
    parser.add_argument("--class-names", type=Path, default=cfg.CLASS_NAMES_PATH)
    parser.add_argument("--output", type=Path, default=cfg.REPORT_DIR / "step4_1_training")
    parser.add_argument("--smoke-max-per-class", type=int, default=cfg.SMOKE_MAX_PER_CLASS)
    args = parser.parse_args()

    class_names = load_class_names(args.class_names)
    engine = InferenceEngine(args.model, args.class_names)
    samples = collect_evaluation_samples(args.dataset, class_names, args.smoke_max_per_class)
    records = []
    for number, (path, true_index, true_class) in enumerate(samples, start=1):
        pred_index, score, _ = engine.predict_path_tf(path)
        records.append(PredictionRecord(
            str(path), true_index, true_class, pred_index,
            class_names[pred_index], score, pred_index == true_index,
        ))
        print(f"[{number}/{len(samples)}] true={true_class} pred={class_names[pred_index]} score={score:.6f}")
    summary = finalize_report(records, class_names, args.output)
    print(summary)


if __name__ == "__main__":
    main()
