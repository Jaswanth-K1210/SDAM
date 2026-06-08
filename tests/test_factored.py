"""Known-answer tests for the pure units of FactoredSDAM (sdam/factored.py).

Pure numpy — no torch — so the factoring math is verified before any GPU run.
Each test prints the actual computed value next to the expected one.
The torch-level Hopfield round-trip is a separate, skipped test (Colab only).
"""
import importlib

import numpy as np
import pytest

from sdam.factored import (
    factor_pattern, augment, reconstruct, variance_matched_w, crossover_point,
)

RNG = np.random.default_rng(0)


def _unit(d, seed=1):
    v = np.random.default_rng(seed).normal(size=d)
    return v / np.linalg.norm(v)


# 1 — round-trip identity: r + c·s == x
def test_factor_roundtrip_identity():
    D = 64
    s = _unit(D)
    X = RNG.normal(size=(50, D))
    r, c = factor_pattern(X, s)
    recon = r + np.outer(c, s)
    err = np.abs(recon - X).max()
    print(f"\n[roundtrip] max|r + c·s − x| = {err:.2e} (expect ~0)")
    assert err < 1e-10
    # residual is orthogonal to the seed (shape removed)
    resid_dot = np.abs(r @ s).max()
    print(f"[roundtrip] max|residual·seed| = {resid_dot:.2e} (expect ~0)")
    assert resid_dot < 1e-10


# 2 — augmented-vector construction: (N, D+1), last col = w·c
def test_augment_shape_and_values():
    D = 16
    s = _unit(D)
    X = RNG.normal(size=(7, D))
    r, c = factor_pattern(X, s)
    w = 0.4
    aug = augment(r, c, w)
    print(f"\n[augment] shape = {aug.shape} (expect (7, {D+1}))")
    assert aug.shape == (7, D + 1)
    assert np.allclose(aug[:, :D], r)
    assert np.allclose(aug[:, D], w * c)
    print(f"[augment] last col == w·c ? {np.allclose(aug[:, D], w * c)}")


# 3 — variance-matched w: w = sqrt(mean_dim_var(residual) / var(c))
def test_variance_matched_w_known():
    # residuals ~ N(0, 4) over D dims -> per-dim var ~4, mean ~4
    # coeffs    ~ N(0, 25)            -> var ~25  => w = sqrt(4/25) = 0.4
    D, N = 30, 200000
    residuals = RNG.normal(scale=2.0, size=(N, D))   # var 4
    coeffs = RNG.normal(scale=5.0, size=N)           # var 25
    w = variance_matched_w(residuals, coeffs)
    print(f"\n[var-matched w] computed = {w:.4f} (expect ~0.40)")
    assert abs(w - 0.4) < 0.02
    # by construction the c-channel variance now matches mean residual-dim variance
    cchan_var = (w * coeffs).var()
    mean_dim_var = residuals.var(axis=0).mean()
    print(f"[var-matched w] c-channel var {cchan_var:.3f} vs mean dim var {mean_dim_var:.3f}")
    assert abs(cchan_var - mean_dim_var) / mean_dim_var < 0.02


# 4 — clean reconstruction ≈ identity through the augmented representation
def test_reconstruct_identity():
    D = 48
    s = _unit(D, seed=3)
    X = RNG.normal(size=(20, D))
    r, c = factor_pattern(X, s)
    w = 0.7
    aug = augment(r, c, w)
    x_hat = reconstruct(aug, s, w)
    err = np.abs(x_hat - X).max()
    print(f"\n[reconstruct] max|reconstruct(augment(factor(x))) − x| = {err:.2e} (expect ~0)")
    assert err < 1e-10


# 5 — crossover-point metric (corruption where seeded gain crosses 0 down)
def test_crossover_point():
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # gains positive then negative, crossing between 0.5 (+0.01) and 0.6 (-0.01) -> ~0.55
    gains = [0.05, 0.04, 0.03, 0.02, 0.01, -0.01, -0.04, -0.07, -0.10]
    xp = crossover_point(rates, gains)
    print(f"\n[crossover] computed = {xp:.3f} (expect ~0.55)")
    assert abs(xp - 0.55) < 1e-6
    # never crosses (all >= 0) -> None
    assert crossover_point(rates, [0.05] * 9) is None
    print("[crossover] all-positive -> None ✓")
    # negative from the start -> at first rate
    assert crossover_point(rates, [-0.02] * 9) == 0.1
    print("[crossover] all-negative -> 0.1 ✓")


# torch-level Hopfield round-trip — Colab only
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_factored_sdam_clean_recall():
    from sdam.factored import FactoredSDAM
    D = 64
    s = _unit(D)
    X = RNG.normal(size=(5, D)).astype(np.float32)
    w = 1.0
    m = FactoredSDAM(s, w, beta=32.0)
    m.reset()
    m.store(X)
    x_hat = m.read(X)
    err = float(np.abs(x_hat - X).max())
    assert err < 0.1  # clean recall reconstructs stored patterns
