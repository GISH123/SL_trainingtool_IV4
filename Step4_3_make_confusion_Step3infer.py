from __future__ import annotations

import argparse
from pathlib import Path

import configuration as cfg
from class_mapping import load_class_names
from inference_engine import InferenceEngine
from report_utils import PredictionRecord, collect_evaluation_samples, finalize_report
from Step3_test_single_image import predict_single_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Step4_3: call the Step3 inference function for every test image.")
    parser.add_argument("--dataset", type=Path, default=cfg.TEST_DIR)
    parser.add_argument("--model", type=Path, default=cfg.LATEST_MODEL_DIR)
    parser.add_argument("--class-names", type=Path, default=cfg.CLASS_NAMES_PATH)
    parser.add_argument("--output", type=Path, default=cfg.REPORT_DIR / "step4_3_step3infer")
    parser.add_argument("--smoke-max-per-class", type=int, default=cfg.SMOKE_MAX_PER_CLASS)
    args = parser.parse_args()

    class_names = load_class_names(args.class_names)
    engine = InferenceEngine(args.model, args.class_names)
    samples = collect_evaluation_samples(args.dataset, class_names, args.smoke_max_per_class)
    records = []
    for number, (path, true_index, true_class) in enumerate(samples, start=1):
        result = predict_single_image(path, engine)
        pred_index = result["pred_index"]
        records.append(PredictionRecord(
            str(path), true_index, true_class, pred_index,
            result["pred_class"], result["score"], pred_index == true_index,
        ))
        print(f"[{number}/{len(samples)}] true={true_class} pred={result['pred_class']} score={result['score']:.6f}")
    summary = finalize_report(records, class_names, args.output)
    print(summary)


if __name__ == "__main__":
    main()
