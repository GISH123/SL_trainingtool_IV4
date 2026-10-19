from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

import configuration as cfg


def is_image_file(path: Path) -> bool:
    return Path(path).suffix.lower() in cfg.VALID_IMAGE_EXTENSIONS


def list_image_files(folder: Path, recursive: bool = True) -> list[Path]:
    folder = Path(folder)
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in iterator if p.is_file() and is_image_file(p))


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resize_rgb_numpy(image_rgb: np.ndarray, height: int, width: int, mode: str) -> np.ndarray:
    if image_rgb is None or image_rgb.ndim != 3:
        raise ValueError("Expected an RGB image with shape [H,W,3].")
    if mode == "stretch":
        return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    if mode == "pad":
        src_h, src_w = image_rgb.shape[:2]
        scale = min(width / src_w, height / src_h)
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(image_rgb, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, 3), dtype=resized.dtype)
        x = (width - dst_w) // 2
        y = (height - dst_h) // 2
        canvas[y:y + dst_h, x:x + dst_w] = resized
        return canvas
    raise ValueError(f"Unsupported RESIZE_MODE: {mode}")


def normalize_numpy(image: np.ndarray, mode: str) -> np.ndarray:
    image = image.astype(np.float32)
    if mode == "minus_one_to_one":
        return image / 127.5 - 1.0
    if mode == "zero_to_one":
        return image / 255.0
    if mode == "none":
        return image
    raise ValueError(f"Unsupported NORMALIZE_MODE: {mode}")


def preprocess_rgb_numpy(image_rgb: np.ndarray) -> np.ndarray:
    resized = resize_rgb_numpy(
        image_rgb, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.RESIZE_MODE
    )
    return normalize_numpy(resized, cfg.NORMALIZE_MODE)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return image


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)


def preprocess_image_tensor(image, augment: bool = False):
    """Shared TensorFlow preprocessing for training, validation and inference."""
    import tensorflow as tf

    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    if cfg.RESIZE_MODE == "stretch":
        image = tf.image.resize(image, [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])
    elif cfg.RESIZE_MODE == "pad":
        image = tf.image.resize_with_pad(image, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    else:
        raise ValueError(f"Unsupported RESIZE_MODE: {cfg.RESIZE_MODE}")

    if augment:
        # No horizontal/vertical flips: they can change poker-card semantics.
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
        image = tf.clip_by_value(image, 0.0, 1.0)

    if cfg.NORMALIZE_MODE == "minus_one_to_one":
        image = image * 2.0 - 1.0
    elif cfg.NORMALIZE_MODE == "zero_to_one":
        pass
    elif cfg.NORMALIZE_MODE == "none":
        image = image * 255.0
    else:
        raise ValueError(f"Unsupported NORMALIZE_MODE: {cfg.NORMALIZE_MODE}")
    image.set_shape([cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.CHANNELS])
    return image


def load_tf_image(path, augment: bool = False):
    import tensorflow as tf

    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=cfg.CHANNELS, expand_animations=False)
    return preprocess_image_tensor(image, augment=augment)
