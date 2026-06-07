"""Variance gate for the feasibility probe — the DUAL-bar decision.

For each factor we derive a 'seed' direction and measure its concentration on
the CENTERED covariance, then the combined variance of the QR-orthonormalized
3-direction span.

Dual bar (reviewer-locked, both must hold):
  - combined > variance_combined_bar        (the span the pipeline uses must
                                             carry variance), AND
  - min(per_factor_concentration) > floor    (each factor must be a non-trivial
                                             axis; random direction ~= 1.0)

A high combined with one factor at concentration ~1 means that factor is dead
and it's really a 2-factor experiment — the per-factor floor catches this.
"""
from __future__ import annotations

import numpy as np

from probe.variance import covariance, concentration, combined_subspace_variance, regression_direction


def factor_seed_direction(
    X: np.ndarray, factor_values: np.ndarray, is_categorical: bool, ridge_lambda: float = 1.0
) -> np.ndarray:
    """L2-normalized 'seed' direction for one factor.

    - categorical (shape): first principal component of the class-mean vectors
      (the axis of maximal between-class spread).
    - continuous (count/layout): ridge regression_direction (stable under
      DINOv2 feature collinearity).
    """
    X = np.asarray(X, dtype=np.float64)
    factor_values = np.asarray(factor_values).reshape(-1)
    if is_categorical:
        classes = np.unique(factor_values)
        means = np.stack([X[factor_values == c].mean(axis=0) for c in classes])  # (k, D)
        means_c = means - means.mean(axis=0, keepdims=True)
        # top right singular vector = first PC of the class-mean spread
        _, _, Vt = np.linalg.svd(means_c, full_matrices=False)
        d = Vt[0]
        return d / (np.linalg.norm(d) + 1e-12)
    return regression_direction(X, factor_values, ridge_lambda=ridge_lambda)


def variance_gate(X: np.ndarray, factors: dict, is_categorical: dict, ridge_lambda: float = 1.0) -> dict:
    """Per-factor concentration + combined orthonormalized-span variance.

    Returns {'per_factor': {name: concentration}, 'combined': float,
             'seed_dirs': (k, D)}.
    """
    X = np.asarray(X, dtype=np.float64)
    Sigma_c = covariance(X)
    per_factor = {}
    dirs = []
    for name, vals in factors.items():
        d = factor_seed_direction(X, vals, is_categorical[name], ridge_lambda=ridge_lambda)
        per_factor[name] = concentration(Sigma_c, d)
        dirs.append(d)
    dirs = np.stack(dirs)
    combined = combined_subspace_variance(Sigma_c, dirs)
    return {"per_factor": per_factor, "combined": combined, "seed_dirs": dirs}


def variance_gate_passes(result: dict, combined_bar: float = 0.10, concentration_floor: float = 2.0) -> bool:
    """Dual-bar decision: combined span carries variance AND no dead factor."""
    combined_ok = result["combined"] > combined_bar
    floor_ok = min(result["per_factor"].values()) > concentration_floor
    return bool(combined_ok and floor_ok)
