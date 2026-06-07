"""Tests for probe/cosine_check.py."""
import numpy as np

from probe.cosine_check import pairwise_cosine_stats

RNG = np.random.default_rng(11)


def test_isotropic_has_spread():
    """Isotropic Gaussian -> centered cosine mean ~0 and a real spread.

    NOTE: analytic std of pairwise cosine for isotropic features is ~1/sqrt(D).
    At D=384 that is ~0.051 (NOT >0.1, which the original prose suggested — that
    would be mathematically impossible at D=384). The degenerate-space gate sits
    at std < 0.05, so isotropic-384 (~0.051) is the healthy baseline just above
    it. We assert the correct value.
    """
    D = 384
    X = RNG.normal(size=(3000, D))
    s = pairwise_cosine_stats(X, n_sample=4000, seed=0)
    print(f"\n[isotropic D=384] centered_mean={s['centered_mean']:.4f} (expect ~0)")
    print(f"[isotropic D=384] centered_std={s['centered_std']:.4f} (analytic ~1/sqrt(384)=0.051)")
    assert abs(s["centered_mean"]) < 0.03
    assert s["centered_std"] > 0.04


def test_common_mode_detected():
    """Big constant offset -> raw cosine mean near 1 (everything looks similar),
    centered cosine mean near 0 (artifact removed by centering)."""
    D = 64
    X = RNG.normal(size=(3000, D)) * 0.1
    X[:, 0] += 10.0  # massive common-mode
    s = pairwise_cosine_stats(X, n_sample=4000, seed=0)
    print(f"\n[common-mode] raw_mean={s['raw_mean']:.4f} (expect ~1)")
    print(f"[common-mode] centered_mean={s['centered_mean']:.4f} (expect ~0)")
    assert s["raw_mean"] > 0.95
    assert abs(s["centered_mean"]) < 0.1
