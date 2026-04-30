"""Run a trained model against the test split and dump metrics + figures
to reports/.

  python -m src.evaluate
  python -m src.evaluate --model models/baseline_cnn.keras
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import load_config, resolve_path
from .data import make_split, make_tf_dataset
from .utils import ensure_dir, get_logger, load_json, save_json, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split")
    parser.add_argument("--model", default=None, help="Path to .keras model. Defaults to inference.model_path")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-misclassified", type=int, default=16)
    return parser.parse_args()


def _plot_confusion_matrix(cm: np.ndarray, classes: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_misclassified(
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    classes: list[str],
    out: Path,
    max_n: int,
) -> None:
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        return
    # Show most confidently wrong predictions first — those are the interesting ones.
    wrong_idx = wrong_idx[np.argsort(-confidences[wrong_idx])][:max_n]
    cols = 4
    rows = int(np.ceil(len(wrong_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax, idx in zip(axes, wrong_idx):
        img = tf.io.decode_image(tf.io.read_file(paths[idx]), channels=3, expand_animations=False).numpy()
        ax.imshow(img); ax.axis("off")
        ax.set_title(
            f"true: {classes[y_true[idx]]}\npred: {classes[y_pred[idx]]} ({confidences[idx]:.2f})",
            fontsize=8,
        )
    for ax in axes[len(wrong_idx):]:
        ax.axis("off")
    fig.suptitle("Top misclassified test examples (most confidently wrong)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("evaluate", log_dir=resolve_path(cfg.paths.log_dir))
    set_global_seed(cfg.data.seed)

    model_path = resolve_path(args.model or cfg.inference.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train a model first.")

    logger.info("Loading model from %s", model_path)
    model = tf.keras.models.load_model(model_path)

    # Use the same deterministic split as training.
    split = make_split(
        data_dir=resolve_path(cfg.data.data_dir),
        classes=cfg.data.classes,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.data.seed,
    )
    classes = split.classes
    num_classes = len(classes)

    test_ds = make_tf_dataset(
        split.test_paths, split.test_labels,
        cfg.data.image_size, cfg.training.batch_size, num_classes,
        shuffle=False,
    )

    logger.info("Predicting on %d test images", len(split.test_paths))
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = np.array(split.test_labels)
    confidences = probs.max(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    metrics = {
        "accuracy": acc,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
    }
    logger.info("Test metrics: %s", metrics)

    reports_dir = ensure_dir(resolve_path(cfg.paths.reports_dir))
    figs_dir = ensure_dir(reports_dir / "figures")

    save_json(metrics, reports_dir / "test_metrics.json")

    per_class = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    save_json(per_class, reports_dir / "per_class_report.json")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with (reports_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true \\ pred"] + classes)
        for cname, row in zip(classes, cm):
            writer.writerow([cname] + row.tolist())

    _plot_confusion_matrix(cm, classes, figs_dir / "confusion_matrix.png")
    _plot_misclassified(
        split.test_paths, y_true, y_pred, confidences,
        classes, figs_dir / "misclassified_grid.png", args.max_misclassified,
    )

    logger.info("Reports written to %s", reports_dir)


if __name__ == "__main__":
    main()
