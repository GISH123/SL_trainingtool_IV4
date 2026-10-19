from __future__ import annotations

import configuration as cfg
from image_utils import preprocess_image_tensor

_FEATURES = None


def parse_tfrecord(serialized, augment: bool = False, include_path: bool = False):
    import tensorflow as tf

    features = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "class_name": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
    }
    item = tf.io.parse_single_example(serialized, features)
    image = tf.io.decode_image(item["image_raw"], channels=cfg.CHANNELS, expand_animations=False)
    image = preprocess_image_tensor(image, augment=augment)
    label = tf.cast(item["label"], tf.int32)
    if include_path:
        return image, label, item["path"]
    return image, label
