"""Dataset loading, splitting, augmentation. Expects the Kaggle Flower
Recognition layout: one folder per class under data/flowers/."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .utils import get_logger

logger = get_logger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DatasetSplit:
    train_paths: list[str]
    train_labels: list[int]
    val_paths: list[str]
    val_labels: list[int]
    test_paths: list[str]
    test_labels: list[int]
    classes: list[str]

    def class_distribution(self) -> dict[str, dict[str, int]]:
        return {
            "train": dict(Counter(self.classes[i] for i in self.train_labels)),
            "val": dict(Counter(self.classes[i] for i in self.val_labels)),
            "test": dict(Counter(self.classes[i] for i in self.test_labels)),
        }


def list_image_paths(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTS)


def discover_classes(data_dir: Path, expected: Sequence[str] | None = None) -> list[str]:
    found = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if expected is not None and set(found) != set(expected):
        logger.warning(
            "Class folders found %s differ from expected %s; using folders as-is.",
            found,
            list(expected),
        )
    return found


def make_split(
    data_dir: str | Path,
    classes: Sequence[str] | None,
    val_split: float,
    test_split: float,
    seed: int,
) -> DatasetSplit:
    """Per-class stratified split into train/val/test."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_names = discover_classes(data_dir, classes)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_p, train_y, val_p, val_y, test_p, test_y = [], [], [], [], [], []

    for cls in class_names:
        paths = [str(p) for p in list_image_paths(data_dir / cls)]
        if not paths:
            logger.warning("Class %s has no images, skipping", cls)
            continue
        labels = [class_to_idx[cls]] * len(paths)

        # Pull off the test set first, then split the rest into train/val.
        rest_p, te_p, rest_y, te_y = train_test_split(
            paths, labels, test_size=test_split, random_state=seed, stratify=labels
        )
        relative_val = val_split / (1.0 - test_split)
        tr_p, va_p, tr_y, va_y = train_test_split(
            rest_p, rest_y, test_size=relative_val,
            random_state=seed, stratify=rest_y,
        )
        train_p.extend(tr_p); train_y.extend(tr_y)
        val_p.extend(va_p); val_y.extend(va_y)
        test_p.extend(te_p); test_y.extend(te_y)

    logger.info("split sizes — train: %d val: %d test: %d (%d classes)",
                len(train_p), len(val_p), len(test_p), len(class_names))
    return DatasetSplit(train_p, train_y, val_p, val_y, test_p, test_y, list(class_names))


# ---------------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------------

def _decode_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


def _build_augmenter(aug_cfg) -> tf.keras.Sequential:
    """Train-only augmentation stack."""
    layers = [
        tf.keras.layers.RandomFlip("horizontal" if aug_cfg.horizontal_flip else "none"),
        tf.keras.layers.RandomRotation(aug_cfg.rotation_range / 360.0),
        tf.keras.layers.RandomZoom(aug_cfg.zoom_range),
        tf.keras.layers.RandomTranslation(aug_cfg.height_shift_range, aug_cfg.width_shift_range),
        tf.keras.layers.RandomBrightness(
            (aug_cfg.brightness_range[1] - aug_cfg.brightness_range[0]) / 2.0
        ),
    ]
    return tf.keras.Sequential(layers, name="augmenter")


def make_tf_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    image_size: int,
    batch_size: int,
    num_classes: int,
    shuffle: bool,
    augmenter: tf.keras.Sequential | None = None,
    rescale_01: bool = True,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 4096), seed=42, reshuffle_each_iteration=True)

    def _load(path, label):
        img = _decode_image(path, image_size)
        if rescale_01:
            img = img / 255.0
        return img, tf.one_hot(label, depth=num_classes)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(labels: Sequence[int], num_classes: int) -> dict[int, float]:
    """Inverse-frequency class weights, mean-normalised to ~1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}
