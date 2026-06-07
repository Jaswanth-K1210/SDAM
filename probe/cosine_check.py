"""Pairwise cosine distribution — the FIRST thing run_probe prints.

Gates the gate: if mean-centered cosines are still tightly clustered
(std < ~0.05), DINOv2 did NOT fix the common-mode on CLEVR renders and every
downstream metric is computed in a degenerate space. We report raw AND centered
so the common-mode (raw mean ~1) and its removal (centered mean ~0) are both
visible.
"""
from __future__ import annotations

import numpy as np


def _cosines(M: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return np.sum(Mn[i] * Mn[j], axis=1)


def pairwise_cosine_stats(X: np.ndarray, n_sample: int = 2000, seed: int = 0) -> dict:
    """Sample random pairs; cosine on RAW and on MEAN-CENTERED features.

    Returns mean/std/p05/p95 for each, plus 40-bin histograms over [-1, 1]
    (counts, edges) for plotting.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    i = rng.integers(0, n, size=n_sample)
    j = rng.integers(0, n, size=n_sample)
    mask = i != j
    i, j = i[mask], j[mask]

    raw = _cosines(X, i, j)
    centered = _cosines(X - X.mean(axis=0, keepdims=True), i, j)

    h_raw_c, h_raw_e = np.histogram(raw, bins=40, range=(-1.0, 1.0))
    h_cen_c, h_cen_e = np.histogram(centered, bins=40, range=(-1.0, 1.0))

    return {
        "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
        "raw_p05": float(np.percentile(raw, 5)), "raw_p95": float(np.percentile(raw, 95)),
        "centered_mean": float(centered.mean()), "centered_std": float(centered.std()),
        "centered_p05": float(np.percentile(centered, 5)),
        "centered_p95": float(np.percentile(centered, 95)),
        "histogram_raw": (h_raw_c.tolist(), h_raw_e.tolist()),
        "histogram_centered": (h_cen_c.tolist(), h_cen_e.tolist()),
    }
