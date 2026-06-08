"""FactoredSDAM — factored-storage variant (Milestone 3).

Tests whether factoring the dominant (objectness/shape) axis into a separate
scalar channel improves Hopfield recall over storing the raw pattern. For each
centered pattern x and unit seed s:

    c = x · s                  # shape coefficient (scalar)
    r = x − c·s                # residual (shape removed; orthogonal to s)
    stored vector = [r, w·c]   # (D+1)-dim, w = channel scale (see spec §2.1)

Read reconstructs  x̂ = r_hat + c_hat·s  where c_hat is recalled FROM MEMORY
(the retrieved channel), not from the query.

The pure factoring functions are torch-free (unit-tested in tests/test_factored.py).
Only FactoredSDAM touches torch, via the UNMODIFIED sdam.hopfield.HopfieldLayer
(lazy import). sdam/model.py, seeds.py, hopfield.py are not edited.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Pure factoring math (numpy; no torch)
# --------------------------------------------------------------------------- #
def factor_pattern(X: np.ndarray, seed: np.ndarray):
    """Split X into (residual, coefficient) along unit `seed`.

    Returns (r, c). For (D,) input returns ((D,), scalar); for (N, D) returns
    ((N, D), (N,)). residual is orthogonal to seed; r + c·seed == X.
    """
    X = np.asarray(X, dtype=np.float64)
    s = np.asarray(seed, dtype=np.float64).reshape(-1)
    single = X.ndim == 1
    Xb = X.reshape(1, -1) if single else X
    c = Xb @ s                          # (N,)
    r = Xb - np.outer(c, s)             # (N, D)
    if single:
        return r[0], float(c[0])
    return r, c


def variance_matched_w(residuals: np.ndarray, coeffs: np.ndarray) -> float:
    """Channel scale making the c-channel variance equal the MEAN per-dimension
    variance of the residual: w = sqrt(mean_d Var(r_d) / Var(c)).

    Neutral choice — shape contributes to addressing like one residual dimension.
    Pre-registered (spec §2.1); never tuned.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    mean_dim_var = float(residuals.var(axis=0).mean())
    c_var = float(coeffs.var())
    return float(np.sqrt(mean_dim_var / (c_var + 1e-12)))


def augment(residual: np.ndarray, c, w: float) -> np.ndarray:
    """Append the scaled coefficient channel: [residual, w·c] -> (.., D+1)."""
    residual = np.asarray(residual, dtype=np.float64)
    single = residual.ndim == 1
    R = residual.reshape(1, -1) if single else residual
    c_arr = np.atleast_1d(np.asarray(c, dtype=np.float64)).reshape(-1)
    aug = np.concatenate([R, (w * c_arr).reshape(-1, 1)], axis=1)
    return aug[0] if single else aug


def reconstruct(aug: np.ndarray, seed: np.ndarray, w: float) -> np.ndarray:
    """Inverse of augment∘factor: x̂ = r_hat + (wc/w)·seed. c comes from `aug`."""
    aug = np.asarray(aug, dtype=np.float64)
    s = np.asarray(seed, dtype=np.float64).reshape(-1)
    single = aug.ndim == 1
    A = aug.reshape(1, -1) if single else aug
    r_hat = A[:, :-1]
    c = A[:, -1] / w
    x = r_hat + np.outer(c, s)
    return x[0] if single else x


def crossover_point(rates, gains):
    """Corruption rate where the seeded gain (seeded − baseline) crosses 0 downward.

    Linearly interpolated. Returns None if the gain is non-negative everywhere
    (no crossover); returns rates[0] if it is already negative at the first rate.
    """
    rates = list(rates)
    gains = list(gains)
    if all(g >= 0 for g in gains):
        return None
    if gains[0] < 0:
        return float(rates[0])
    for i in range(len(rates) - 1):
        if gains[i] >= 0 and gains[i + 1] < 0:
            r0, r1 = rates[i], rates[i + 1]
            g0, g1 = gains[i], gains[i + 1]
            return float(r0 + (r1 - r0) * g0 / (g0 - g1))
    return None


# --------------------------------------------------------------------------- #
# Torch wrapper around the UNMODIFIED HopfieldLayer (lazy import; Colab/GPU)
# --------------------------------------------------------------------------- #
class FactoredSDAM:
    """Factored-storage associative memory: store [residual, w·c] in a Modern
    Hopfield memory; reconstruct with the coefficient recalled from memory."""

    def __init__(self, seed: np.ndarray, w: float, beta: float = 16.0, max_iter: int = 3):
        import torch

        from sdam.hopfield import HopfieldLayer

        self._torch = torch
        self.seed = np.asarray(seed, dtype=np.float64).reshape(-1)
        self.dim = self.seed.shape[0]
        self.w = float(w)
        self.mem = HopfieldLayer(self.dim + 1, beta=beta, max_iter=max_iter)

    def reset(self) -> None:
        self.mem.reset()

    @property
    def n_stored(self) -> int:
        return self.mem.n_stored

    def store(self, X: np.ndarray) -> None:
        r, c = factor_pattern(X, self.seed)
        aug = np.atleast_2d(augment(r, c, self.w))
        self.mem.store(self._torch.tensor(aug, dtype=self._torch.float32))

    def read(self, X: np.ndarray) -> np.ndarray:
        r, c = factor_pattern(X, self.seed)
        key = np.atleast_2d(augment(r, c, self.w))
        aug_hat = self.mem.retrieve(self._torch.tensor(key, dtype=self._torch.float32))
        aug_hat = aug_hat.detach().cpu().numpy()
        x_hat = reconstruct(aug_hat, self.seed, self.w)
        single = np.asarray(X).ndim == 1
        return x_hat[0] if (single and x_hat.ndim == 2) else x_hat
