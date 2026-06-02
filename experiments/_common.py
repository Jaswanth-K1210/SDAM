"""Shared experiment helpers: config loading and data acquisition."""

from __future__ import annotations

import json
import os

import numpy as np
import yaml

from data.clevr_loader import load_features, synthetic_features

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "base.yaml")


def load_config(path: str = _CONFIG_PATH) -> dict:
    """Load the YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_features(cfg: dict):
    """Return labeled features, falling back to synthetic data with a warning.

    Returns ``(features_dict, used_synthetic_bool)``.
    """
    data_cfg = cfg["data"]
    seed_dim = cfg["model"]["seed_dim"]
    try:
        feats = load_features(data_cfg["clevr_root"])
        return feats, False
    except FileNotFoundError:
        if not data_cfg.get("use_synthetic_fallback", True):
            raise
        print(
            "[data] CLEVR features not found -> falling back to SYNTHETIC features. "
            "(Set data.use_synthetic_fallback=false to forbid this.)"
        )
        feats = synthetic_features(
            n_per_category=data_cfg["n_per_category"], seed_dim=seed_dim, device="cpu"
        )
        return feats, True


def ensure_results_dir(cfg: dict) -> str:
    out = cfg["output"]["results_dir"]
    os.makedirs(out, exist_ok=True)
    return out


def _to_native(o):
    """json.dump default hook: coerce NumPy scalars/arrays to native Python.

    NumPy types (``np.bool_``, ``np.float32``, ``np.int64``, ``np.ndarray``)
    are not JSON-serializable. In NumPy 2.x ``np.bool_`` even reports its class
    name as ``"bool"``, producing the misleading "Object of type bool is not
    JSON serializable" error.
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def dump_json(obj, path: str) -> None:
    """Write ``obj`` to ``path`` as JSON, tolerating NumPy scalars/arrays."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_native)
