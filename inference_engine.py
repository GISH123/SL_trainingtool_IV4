from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import configuration as cfg
from class_mapping import load_class_names
from image_utils import load_tf_image, preprocess_rgb_numpy, read_bgr
from model_io import load_saved_model_callable


class InferenceEngine:
    def __init__(self, model_dir: Path = cfg.LATEST_MODEL_DIR, class_names_path: Path = cfg.CLASS_NAMES_PATH):
        self.class_names = load_class_names(class_names_path)
        self.model = load_saved_model_callable(model_dir)

    @staticmethod
    def _decode_logits(logits) -> tuple[int, float, np.ndarray]:
        import tensorflow as tf

        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        index = int(np.argmax(probabilities))
        return index, float(probabilities[index]), probabilities

    def predict_rgb(self, image_rgb: np.ndarray) -> tuple[int, float, np.ndarray]:
        batch = np.expand_dims(preprocess_rgb_numpy(image_rgb), axis=0).astype(np.float32)
        logits = self.model(batch)
        return self._decode_logits(logits)

    def predict_bgr(self, image_bgr: np.ndarray) -> tuple[int, float, np.ndarray]:
        # pydealerclient/OpenCV path: OpenCV BGR must become RGB before model input.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.predict_rgb(image_rgb)

    def predict_path_cv2(self, path: Path) -> tuple[int, float, np.ndarray]:
        return self.predict_bgr(read_bgr(path))

    def predict_path_tf(self, path: Path) -> tuple[int, float, np.ndarray]:
        import tensorflow as tf

        image = load_tf_image(str(path), augment=False)
        logits = self.model(tf.expand_dims(image, axis=0))
        return self._decode_logits(logits)

    def class_name(self, index: int) -> str:
        return self.class_names[index]
