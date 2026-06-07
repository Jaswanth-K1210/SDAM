"""Feature extraction cache — the A100-time fix.

Re-running 6,000-image DINOv2 extraction every iteration wastes GPU time. We
cache the (N, 384) feature matrix to disk, keyed on a hash of
(model_name, ordered image filenames, preprocessing signature). Order matters:
features are index-aligned to the factor arrays, so the key must reflect the
exact ordered list, not a sorted one.

Pure/torch-free so it is unit-testable with a dummy extractor.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


def feature_cache_key(model_name: str, image_filenames: list[str], preprocessing_sig: str) -> str:
    """Deterministic 16-hex key over (model, ordered basenames, preprocessing)."""
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(b"\x00")
    h.update(preprocessing_sig.encode())
    h.update(b"\x00")
    h.update(str(len(image_filenames)).encode())
    for fn in image_filenames:
        h.update(b"\x00")
        h.update(os.path.basename(fn).encode())
    return h.hexdigest()[:16]


def cached_features(
    cache_dir: str,
    model_name: str,
    image_paths: list[str],
    preprocessing_sig: str,
    extract_fn,
) -> np.ndarray:
    """Return the (N, D) feature matrix, computing+caching on miss, loading on hit.

    extract_fn(image_paths) -> np.ndarray is called only on a cache miss.
    """
    key = feature_cache_key(model_name, image_paths, preprocessing_sig)
    path = os.path.join(cache_dir, f"feats_{model_name}_{key}.npz")
    if os.path.exists(path):
        return np.load(path)["features"]
    feats = np.asarray(extract_fn(image_paths), dtype=np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, features=feats)
    return feats
