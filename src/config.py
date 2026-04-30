"""YAML config loader. Returns a nested SimpleNamespace so callers can do
cfg.training.batch_size instead of dict["training"]["batch_size"]."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return _to_namespace(yaml.safe_load(fh))


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str | Path) -> Path:
    """Anchor a relative path to the project root."""
    p = Path(p)
    return p if p.is_absolute() else project_root() / p
