from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import configuration as cfg
from class_mapping import class_index_map
from image_utils import list_image_files, load_tf_image, sha1_file


def collect_split_samples(split_dir: Path, class_names: list[str]) -> list[tuple[str, int]]:
    mapping = class_index_map(class_names)
    rows: list[tuple[str, int]] = []
    for class_name in class_names:
        folder = Path(split_dir) / class_name
        if not folder.exists():
            continue
        rows.extend((str(path), mapping[class_name]) for path in list_image_files(folder))
    return rows


def _hashes_in_splits(split_dirs: list[Path], class_names: list[str]) -> set[str]:
    hashes: set[str] = set()
    for split_dir in split_dirs:
        for class_name in class_names:
            folder = Path(split_dir) / class_name
            if not folder.exists():
                continue
            for path in list_image_files(folder):
                hashes.add(sha1_file(path))
    return hashes


def collect_hard_pool_samples(class_names: list[str]) -> list[tuple[str, int]]:
    if not cfg.USE_HARD_POOL or not cfg.HARD_POOL_DIR.exists():
        return []
    mapping = class_index_map(class_names)
    protected = _hashes_in_splits([cfg.VALID_DIR, cfg.TEST_DIR], class_names)
    rows = []
    seen = set()
    for class_name in class_names:
        folder = cfg.HARD_POOL_DIR / class_name
        if not folder.exists():
            continue
        for path in list_image_files(folder):
            digest = sha1_file(path)
            if digest in protected or digest in seen:
                continue
            seen.add(digest)
            rows.append((str(path), mapping[class_name]))
    return rows


def oversample(rows: list[tuple[str, int]], seed: int) -> list[tuple[str, int]]:
    if not cfg.OVERSAMPLE_TRAIN or not rows:
        return rows
    rng = random.Random(seed)
    grouped: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        grouped[row[1]].append(row)
    target = cfg.MAX_OVERSAMPLE_PER_CLASS or max(len(v) for v in grouped.values())
    result: list[tuple[str, int]] = []
    for original in grouped.values():
        if len(original) >= target:
            result.extend(original[:target])
        else:
            result.extend(original)
            result.extend(rng.choice(original) for _ in range(target - len(original)))
    rng.shuffle(result)
    return result


def make_dataset(rows: list[tuple[str, int]], training: bool):
    import tensorflow as tf

    if not rows:
        raise RuntimeError("Dataset has no images.")
    paths = [x[0] for x in rows]
    labels = [x[1] for x in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(rows), seed=cfg.RANDOM_SEED, reshuffle_each_iteration=True)

    def mapper(path, label):
        image = load_tf_image(path, augment=training and cfg.DATA_AUGMENTATION)
        return image, label

    ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.BATCH_SIZE, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)


def dataset_summary(rows: list[tuple[str, int]]) -> dict[int, int]:
    return dict(sorted(Counter(label for _, label in rows).items()))
