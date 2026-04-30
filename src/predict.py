"""Inference: a tiny predictor class plus a CLI.

  python -m src.predict path/to/image.jpg
  python -m src.predict path/to/folder --top-k 3
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import load_config, resolve_path
from .utils import load_json

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FlowerPredictor:
    """Loads the model + label map once. Reused by the CLI and the API."""

    def __init__(self, model_path, labels_path, image_size: int):
        model_path = Path(model_path)
        labels_path = Path(labels_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.image_size = int(image_size)
        # The transfer model wraps mobilenet_v2.preprocess_input in a Lambda layer.
        # Keras 3 can't resolve the function by name on its own, so pass it explicitly.
        # safe_mode=False allows the anonymous `lambda t: t * 255.0` rescaler.
        self.model = tf.keras.models.load_model(
            model_path,
            safe_mode=False,
            custom_objects={
                "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
            },
        )
        # JSON keys are strings — normalise to int.
        self.labels: dict[int, str] = {int(k): v for k, v in load_json(labels_path).items()}

    @classmethod
    def from_config(cls, config_path=None) -> "FlowerPredictor":
        cfg = load_config(config_path)
        return cls(
            model_path=resolve_path(cfg.inference.model_path),
            labels_path=resolve_path(cfg.inference.labels_path),
            image_size=cfg.data.image_size,
        )

    def _to_array(self, source) -> np.ndarray:
        """source: file path, raw bytes, or a PIL.Image. Returns HWC float32 in [0,1]."""
        if isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, Image.Image):
            img = source
        else:
            raise TypeError(f"Unsupported source type: {type(source).__name__}")
        img = img.convert("RGB").resize((self.image_size, self.image_size))
        return np.asarray(img, dtype=np.float32) / 255.0

    def predict(self, source, top_k: int = 1) -> dict:
        """Run a single prediction. Returns predicted_label, confidence, top_k."""
        arr = self._to_array(source)
        probs = self.model.predict(arr[None, ...], verbose=0)[0]
        order = np.argsort(-probs)[:top_k]
        return {
            "predicted_label": self.labels[int(order[0])],
            "confidence": float(probs[order[0]]),
            "top_k": [
                {"label": self.labels[int(i)], "probability": float(probs[i])}
                for i in order
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict flower class for an image or folder")
    parser.add_argument("source", help="Image file or directory")
    parser.add_argument("--config", default=None, help="Path to YAML config (optional)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictor = FlowerPredictor.from_config(args.config)
    src = Path(args.source)
    paths = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in VALID_EXTS)
    if not paths:
        raise SystemExit(f"No images found at {src}")

    for p in paths:
        result = predictor.predict(p, top_k=args.top_k)
        line = ", ".join(f"{t['label']} ({t['probability']:.3f})" for t in result["top_k"])
        print(f"{p.name}: {line}")


if __name__ == "__main__":
    main()
