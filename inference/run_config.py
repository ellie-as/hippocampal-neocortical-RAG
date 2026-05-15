"""Shared runtime configuration for inference figure scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent
DEFAULT_CONFIG_PATH = INFERENCE_DIR / "inference_config.json"


DEFAULTS: dict[str, Any] = {
    "spatial_model_dir": str(INFERENCE_DIR / "outputs_graph"),
    "family_model_dir": str(INFERENCE_DIR / "outputs_tree"),
    "cache_dir": str(INFERENCE_DIR / "data" / "rel_inf_cache"),
    "figures_dir": str(REPO_ROOT / "figures"),
    "data_dir": str(INFERENCE_DIR / "data"),
    "seed": 321,
}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load inference runtime config.

    Resolution order:
    1. Explicit ``config_path`` argument
    2. ``INFERENCE_CONFIG`` environment variable
    3. ``inference/inference_config.json`` if present
    4. Built-in repository defaults
    """
    selected = config_path or os.environ.get("INFERENCE_CONFIG")
    path = Path(selected).expanduser() if selected else DEFAULT_CONFIG_PATH

    config = dict(DEFAULTS)
    if path.exists():
        loaded = json.loads(path.read_text())
        config.update({k: v for k, v in loaded.items() if v is not None})

    for key in ("spatial_model_dir", "family_model_dir", "cache_dir", "figures_dir", "data_dir"):
        config[key] = _resolve_path(config[key])
    config["seed"] = int(config.get("seed", DEFAULTS["seed"]))
    config["config_path"] = path
    return config
