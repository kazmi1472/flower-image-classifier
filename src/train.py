"""Training script.

  python -m src.train --model transfer
  python -m src.train --model baseline
  python -m src.train --model transfer --quick

Transfer model uses the standard two-phase schedule (head-only, then
fine-tune the top N backbone layers at a smaller LR). Saves the best
checkpoint to models/<name>.keras and copies it to models/best_model.keras
so the API/CLI pick it up by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import tensorflow as tf

from .config import load_config, resolve_path
from .data import (
    compute_class_weights,
    make_split,
    make_tf_dataset,
    _build_augmenter,
)
from .models import build_baseline_cnn, build_transfer_model, unfreeze_top
from .utils import ensure_dir, get_logger, save_json, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a flower-recognition model")
    parser.add_argument(
        "--model",
        choices=["baseline", "transfer"],
        default="transfer",
        help="Which model to train",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs (baseline only)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Sanity check: 2 epochs, small batches")
    return parser.parse_args()


def _callbacks(cfg, monitor: str, ckpt_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=cfg.training.early_stopping_patience,
            mode="max" if "acc" in monitor else "min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=cfg.training.reduce_lr_patience,
            factor=0.3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=monitor,
            mode="max" if "acc" in monitor else "min",
            save_best_only=True,
            verbose=1,
        ),
    ]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger = get_logger("train", log_dir=resolve_path(cfg.paths.log_dir))
    set_global_seed(cfg.data.seed)

    batch_size = args.batch_size or cfg.training.batch_size
    if args.quick:
        batch_size = min(batch_size, 16)

    models_dir = ensure_dir(resolve_path(cfg.paths.models_dir))
    reports_dir = ensure_dir(resolve_path(cfg.paths.reports_dir))

    # ---- data ----------------------------------------------------------------
    split = make_split(
        data_dir=resolve_path(cfg.data.data_dir),
        classes=cfg.data.classes,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.data.seed,
    )
    save_json({i: c for i, c in enumerate(split.classes)}, models_dir / "labels.json")
    save_json(split.class_distribution(), reports_dir / "class_distribution.json")

    augmenter = _build_augmenter(cfg.augmentation)
    num_classes = len(split.classes)

    train_ds = make_tf_dataset(
        split.train_paths, split.train_labels,
        cfg.data.image_size, batch_size, num_classes,
        shuffle=True, augmenter=augmenter,
    )
    val_ds = make_tf_dataset(
        split.val_paths, split.val_labels,
        cfg.data.image_size, batch_size, num_classes,
        shuffle=False,
    )

    class_weights = None
    if cfg.training.use_class_weights and not args.no_class_weights:
        class_weights = compute_class_weights(split.train_labels, num_classes)
        logger.info("Class weights: %s", class_weights)

    # ---- model ---------------------------------------------------------------
    if args.model == "baseline":
        model = build_baseline_cnn(cfg.data.image_size, num_classes, dropout=cfg.model.dropout)
        epochs = args.epochs or cfg.training.epochs_baseline
        if args.quick:
            epochs = 2
        ckpt_path = models_dir / "baseline_cnn.keras"

        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.training.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("Baseline CNN — params: %s", model.count_params())

        t0 = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=_callbacks(cfg, "val_accuracy", ckpt_path),
        )
        logger.info("Baseline trained in %.1fs", time.time() - t0)
        save_json(history.history, models_dir / "baseline_cnn_history.json")
        final_ckpt = ckpt_path

    else:  # transfer
        model, base = build_transfer_model(
            cfg.data.image_size,
            num_classes,
            backbone=cfg.model.transfer_backbone,
            dense_units=cfg.model.dense_units,
            dropout=cfg.model.dropout,
        )
        ckpt_path = models_dir / f"transfer_{cfg.model.transfer_backbone}.keras"

        # Phase 1: head-only
        head_epochs = 2 if args.quick else cfg.training.epochs_transfer_head
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.training.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("Phase 1 — head-only training for %d epochs", head_epochs)
        t0 = time.time()
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=head_epochs,
            class_weight=class_weights,
            callbacks=_callbacks(cfg, "val_accuracy", ckpt_path),
        )

        # Phase 2: fine-tune top N layers at small LR
        ft_epochs = 2 if args.quick else cfg.training.epochs_transfer_finetune
        unfreeze_top(base, cfg.training.finetune_unfreeze_layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.training.finetune_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("Phase 2 — fine-tuning top %d backbone layers", cfg.training.finetune_unfreeze_layers)
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=head_epochs + ft_epochs,
            initial_epoch=len(h1.history.get("loss", [])),
            class_weight=class_weights,
            callbacks=_callbacks(cfg, "val_accuracy", ckpt_path),
        )
        logger.info("Transfer trained in %.1fs", time.time() - t0)

        merged = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in h1.history}
        save_json(merged, models_dir / f"transfer_{cfg.model.transfer_backbone}_history.json")
        final_ckpt = ckpt_path

    # ---- promote as the canonical inference model ---------------------------
    best_path = models_dir / "best_model.keras"
    shutil.copy2(final_ckpt, best_path)
    logger.info("Saved best model to %s", best_path)


if __name__ == "__main__":
    main()
