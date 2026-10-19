from __future__ import annotations

from pathlib import Path

import configuration as cfg
from pydealer_predict import get_engine


def do_predict_one_label_original(
    image_bgr,
    model_dir: Path = cfg.LATEST_MODEL_DIR,
    class_names_path: Path = cfg.CLASS_NAMES_PATH,
):
    """pydealerclient-style prediction path.

    Input is the original OpenCV BGR frame/crop. The function converts BGR->RGB,
    applies the exact shared 299x299 preprocessing, then returns (y_pred, y_prob)
    in the same scalar format expected by Step4_1.
    """
    engine = get_engine(model_dir, class_names_path)
    pred_index, score, _ = engine.predict_bgr(image_bgr)
    return pred_index, score
