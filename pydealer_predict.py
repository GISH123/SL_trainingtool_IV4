from __future__ import annotations

from pathlib import Path

import configuration as cfg
from inference_engine import InferenceEngine

_ENGINE: InferenceEngine | None = None
_ENGINE_KEY: tuple[str, str] | None = None


def get_engine(model_dir: Path = cfg.LATEST_MODEL_DIR, class_names_path: Path = cfg.CLASS_NAMES_PATH) -> InferenceEngine:
    global _ENGINE, _ENGINE_KEY
    key = (str(Path(model_dir).resolve()), str(Path(class_names_path).resolve()))
    if _ENGINE is None or _ENGINE_KEY != key:
        _ENGINE = InferenceEngine(model_dir, class_names_path)
        _ENGINE_KEY = key
    return _ENGINE


def predict_bgr(image_bgr, model_dir: Path = cfg.LATEST_MODEL_DIR, class_names_path: Path = cfg.CLASS_NAMES_PATH):
    return get_engine(model_dir, class_names_path).predict_bgr(image_bgr)
