"""Tests for probe/variance_gate.py — seed directions + the dual-bar gate."""
import numpy as np

from probe.variance_gate import factor_seed_direction, variance_gate, variance_gate_passes

RNG = np.random.default_rng(23)


def test_seed_direction_categorical_recovers_separation():
    """3 classes whose means separate along e0 -> seed direction aligns with e0."""
    D = 30
    n = 6000
    y = RNG.integers(0, 3, size=n)
    X = RNG.normal(size=(n, D)) * 0.2
    X[:, 0] += (y - 1) * 3.0   # class means at -3, 0, +3 on axis 0
    d = factor_seed_direction(X, y, is_categorical=True)
    e0 = np.zeros(D); e0[0] = 1.0
    align = abs(float(d @ e0))
    print(f"\n[seed cat] |cos(seed, e0)| = {align:.4f} (expect ~1)")
    assert align > 0.95


def test_seed_direction_continuous():
    """Continuous factor linear in e7 -> ridge seed direction aligns with e7."""
    D = 30
    n = 6000
    X = RNG.normal(size=(n, D))
    y = 4.0 * X[:, 7] + 0.1 * RNG.normal(size=n)
    d = factor_seed_direction(X, y, is_categorical=False)
    e7 = np.zeros(D); e7[7] = 1.0
    align = abs(float(d @ e7))
    print(f"\n[seed cont] |cos(seed, e7)| = {align:.4f} (expect ~1)")
    assert align > 0.95


def _structured(n=6000, D=30):
    ys = RNG.integers(0, 3, size=n)
    yc = RNG.normal(size=n)
    yl = RNG.normal(size=n)
    X = RNG.normal(size=(n, D)) * 0.1
    X[:, 0] += (ys - 1) * 3.0   # shape loud on axis 0
    X[:, 1] += yc * 3.0         # count loud on axis 1
    X[:, 2] += yl * 3.0         # layout loud on axis 2
    return X, ys, yc, yl


def test_gate_passes_on_structured():
    X, ys, yc, yl = _structured()
    factors = {"shape": ys, "count": yc, "layout": yl}
    is_cat = {"shape": True, "count": False, "layout": False}
    res = variance_gate(X, factors, is_cat)
    pf = {k: round(v, 2) for k, v in res["per_factor"].items()}
    print(f"\n[gate pass] per_factor concentration = {pf}")
    print(f"[gate pass] combined = {res['combined']:.4f} (expect > 0.10)")
    assert res["combined"] > 0.10
    assert min(res["per_factor"].values()) > 2.0
    assert variance_gate_passes(res, combined_bar=0.10, concentration_floor=2.0)


def test_gate_fails_one_dead_factor():
    """2 loud factors + 1 factor with NO axis in the features (dead).
    Combined may still exceed 0.10 (from the 2 loud), but the dead factor's
    concentration ~1 fails the per-factor floor -> gate FAIL."""
    X, ys, yc, _ = _structured()
    yl_dead = RNG.normal(size=X.shape[0])   # unrelated to any feature axis
    factors = {"shape": ys, "count": yc, "layout": yl_dead}
    is_cat = {"shape": True, "count": False, "layout": False}
    res = variance_gate(X, factors, is_cat)
    pf = {k: round(v, 2) for k, v in res["per_factor"].items()}
    print(f"\n[gate fail] per_factor concentration = {pf}")
    print(f"[gate fail] combined = {res['combined']:.4f}")
    print(f"[gate fail] layout (dead) concentration = {res['per_factor']['layout']:.2f} (expect ~1)")
    assert res["per_factor"]["layout"] < 2.0          # dead factor detected
    assert not variance_gate_passes(res, combined_bar=0.10, concentration_floor=2.0)
