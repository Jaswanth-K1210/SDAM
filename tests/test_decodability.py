"""Known-answer tests for probe/decodability.py.

Plant factors that ARE decodable and factors that are PURE NOISE, then assert
the probe separates them: real >> permutation for signal, real ~= permutation
for noise. Also verify tertile binning balances classes and reports sane
boundaries.
"""
import numpy as np
import pytest

from probe.decodability import balanced_tertiles, decodability

RNG = np.random.default_rng(7)


def _split(n, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(n * frac)
    return perm[:k], perm[k:]


def test_balanced_tertiles_are_balanced():
    """Continuous uniform -> three nearly-equal classes; boundaries monotone."""
    y = RNG.uniform(0, 100, size=9000)
    labels, bounds = balanced_tertiles(y)
    counts = np.bincount(labels)
    print(f"\n[tertiles] class counts = {counts.tolist()} (expect ~3000 each)")
    print(f"[tertiles] boundaries = {bounds.round(2).tolist()} (expect ~33, ~67)")
    assert counts.min() > 2800 and counts.max() < 3200
    assert bounds[0] < bounds[1]


def test_balanced_tertiles_skew_detected_via_boundaries():
    """Heavily skewed target: binning still balances by RANK, but boundaries
    reveal the skew (both cut points near the low end)."""
    # 90% of mass near 0, a long thin tail to 100
    y = np.concatenate([RNG.uniform(0, 1, size=8100), RNG.uniform(1, 100, size=900)])
    labels, bounds = balanced_tertiles(y)
    counts = np.bincount(labels)
    print(f"\n[tertiles skew] class counts = {counts.tolist()} (rank-balanced -> ~3000 each)")
    print(f"[tertiles skew] boundaries = {bounds.round(3).tolist()} (both should sit low, < 1.5)")
    assert counts.min() > 2800  # still rank-balanced
    assert bounds[1] < 1.5      # boundary logging reveals the skew


def test_decodable_categorical_factor():
    """Shape-like: 3 classes living on separable means -> high balanced acc,
    permutation collapses to chance."""
    D = 50
    n = 3000
    centers = np.zeros((3, D))
    centers[0, 0] = 3.0
    centers[1, 1] = 3.0
    centers[2, 2] = 3.0
    y = RNG.integers(0, 3, size=n)
    X = centers[y] + RNG.normal(size=(n, D))
    tr, te = _split(n, seed=1)
    res = decodability(X, y, "shape", is_categorical=True, train_idx=tr, test_idx=te)
    print(f"\n[decodable cat] balanced_acc={res.balanced_acc:.3f} (expect > 0.9)")
    print(f"[decodable cat] permutation={res.permutation_acc:.3f} (expect ~0.33)")
    print(f"[decodable cat] margin={res.margin:.3f}  best_C={res.best_C}")
    assert res.balanced_acc > 0.9
    assert abs(res.permutation_acc - 0.333) < 0.08
    assert res.passed()


def test_noise_factor_fails():
    """Label independent of features -> real acc ~= permutation ~= chance.
    The probe must NOT report this as decodable (false-positive guard)."""
    D = 50
    n = 3000
    X = RNG.normal(size=(n, D))
    y = RNG.integers(0, 3, size=n)  # random labels, no relation to X
    tr, te = _split(n, seed=2)
    res = decodability(X, y, "noise", is_categorical=True, train_idx=tr, test_idx=te)
    print(f"\n[noise] balanced_acc={res.balanced_acc:.3f} (expect ~0.33)")
    print(f"[noise] permutation={res.permutation_acc:.3f} (expect ~0.33)")
    print(f"[noise] margin={res.margin:.3f} (expect ~0)")
    assert res.balanced_acc < 0.45
    assert abs(res.margin) < 0.12
    assert not res.passed()


def test_decodable_continuous_factor_with_tertiles():
    """Count/layout-like: continuous target linear in one axis. Tertile-binned
    decodability is high, AND continuous Spearman/R2 are populated and strong."""
    D = 40
    n = 3000
    X = RNG.normal(size=(n, D))
    y_cont = 5.0 * X[:, 10] + 0.3 * RNG.normal(size=n)  # continuous, ordinal
    tr, te = _split(n, seed=3)
    res = decodability(X, y_cont, "count", is_categorical=False, train_idx=tr, test_idx=te)
    print(f"\n[decodable cont] balanced_acc={res.balanced_acc:.3f} (expect > 0.8)")
    print(f"[decodable cont] permutation={res.permutation_acc:.3f} (expect ~0.33)")
    print(f"[decodable cont] spearman_rho={res.spearman_rho:.3f} (expect > 0.9)")
    print(f"[decodable cont] multivariate_r2={res.multivariate_r2:.3f} (expect > 0.9)")
    print(f"[decodable cont] tertile_boundaries={[round(b,2) for b in res.tertile_boundaries]}")
    assert res.balanced_acc > 0.8
    assert res.spearman_rho > 0.9
    assert res.multivariate_r2 > 0.9
    assert res.tertile_boundaries is not None
    assert res.passed()


def test_imbalanced_labels_use_balanced_accuracy():
    """When one class dominates 90%, raw accuracy would be misleadingly high
    by always predicting majority. balanced_accuracy must stay near chance for
    a noise factor even under heavy imbalance."""
    D = 30
    n = 4000
    X = RNG.normal(size=(n, D))
    # 90% class 0, 5% class 1, 5% class 2, labels unrelated to X
    y = np.zeros(n, dtype=int)
    y[: int(0.05 * n)] = 1
    y[int(0.05 * n) : int(0.10 * n)] = 2
    RNG.shuffle(y)
    tr, te = _split(n, seed=4)
    res = decodability(X, y, "imbalanced_noise", is_categorical=True, train_idx=tr, test_idx=te)
    print(f"\n[imbalanced noise] balanced_acc={res.balanced_acc:.3f} (expect ~0.33, NOT ~0.9)")
    print(f"[imbalanced noise] class_counts={res.class_counts}")
    assert res.balanced_acc < 0.45  # balanced metric resists majority-class trick
    assert not res.passed()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
