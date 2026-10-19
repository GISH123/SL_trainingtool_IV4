from __future__ import annotations

import argparse
from pathlib import Path

import configuration as cfg
from class_mapping import class_index_map, load_class_names
from image_utils import list_image_files


def _bytes_feature(value: bytes):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int_feature(value: int):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def build_tfrecord(split_dir: Path, output_path: Path, class_names_path: Path) -> int:
    import tensorflow as tf

    class_names = load_class_names(class_names_path)
    mapping = class_index_map(class_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for class_name in class_names:
            folder = split_dir / class_name
            if not folder.exists():
                continue
            for image_path in list_image_files(folder):
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image_raw": _bytes_feature(image_path.read_bytes()),
                    "label": _int_feature(mapping[class_name]),
                    "class_name": _bytes_feature(class_name.encode("utf-8")),
                    "path": _bytes_feature(str(image_path).encode("utf-8")),
                }))
                writer.write(example.SerializeToString())
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/valid/test TFRecords.")
    parser.add_argument("--dataset", type=Path, default=cfg.OUTPUT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=cfg.TFRECORD_DIR)
    parser.add_argument("--class-names", type=Path, default=cfg.CLASS_NAMES_PATH)
    args = parser.parse_args()

    for split in ("train", "valid", "test"):
        output = args.output / f"{split}.tfrecord"
        count = build_tfrecord(args.dataset / split, output, args.class_names)
        print(f"{split}: {count} -> {output}")


if __name__ == "__main__":
    main()
