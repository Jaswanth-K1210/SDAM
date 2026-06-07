"""Known-answer tests for probe/variance.py.

Each test plants a KNOWN structure and asserts the math recovers it. The
expected values are derived analytically and printed alongside the computed
value so the test itself can be verified (a test can pass against a wrong
expected value — printing the number guards against that).
"""
import numpy as np
import pytest

from probe.variance import (
    second_moment,
    covariance,
    variance_share,
    concentration,
    combined_subspace_variance,
    regression_direction,
    multivariate_r2,
)

RNG = np.random.default_rng(42)


def test_second_moment_equals_cov_plus_mean_outer():
    """second_moment = covariance + mu mu^T (the defining relationship)."""
    X = RNG.normal(size=(2000, 8)) + np.array([3.0, 0, 0, 0, 0, 0, 0, 0])
    M = second_moment(X)
    Sigma = covariance(X)
    mu = X.mean(axis=0)
    reconstructed = Sigma + np.outer(mu, mu)
    err = np.abs(M - reconstructed).max()
    print(f"\n[second_moment decomposition] max|M - (Sigma + mu mu^T)| = {err:.2e} (expect ~0)")
    assert err < 1e-10


def test_variance_share_planted_fraction():
    """Plant 80% of variance on axis 0, 20% spread on the rest. Recover 0.8."""
    D = 10
    n = 50000
    std = np.ones(D)
    # variance on axis0 = 0.8 * total; remaining 0.2 split over 9 axes
    var0 = 0.8
    var_rest = 0.2 / (D - 1)
    std[0] = np.sqrt(var0)
    std[1:] = np.sqrt(var_rest)
    X = RNG.normal(size=(n, D)) * std
    Sigma = covariance(X)
    e0 = np.zeros(D); e0[0] = 1.0
    share = variance_share(Sigma, e0)
    print(f"\n[variance_share planted] planted=0.800  computed={share:.4f} (expect ~0.80)")
    assert abs(share - 0.80) < 0.02


def test_concentration_random_vs_dominant():
    """Random direction -> ~1. Dominant axis -> >> 1. Test both ends."""
    D = 50
    n = 40000
    std = np.ones(D)
    std[0] = np.sqrt(10.0)  # axis 0 carries 10x variance
    X = RNG.normal(size=(n, D)) * std
    Sigma = covariance(X)

    # Dominant axis
    e0 = np.zeros(D); e0[0] = 1.0
    conc_dom = concentration(Sigma, e0)

    # Random direction (averaged over several)
    randoms = []
    for _ in range(20):
        u = RNG.normal(size=D)
        randoms.append(concentration(Sigma, u))
    conc_rand = np.mean(randoms)

    print(f"\n[concentration] dominant axis = {conc_dom:.2f} (expect >> 1)")
    print(f"[concentration] random dir mean = {conc_rand:.2f} (expect ~1)")
    assert conc_dom > 5.0
    assert 0.5 < conc_rand < 2.0


def test_raw_vs_centered_common_mode():
    """THE key test: plant a big common-mode on one axis.

    raw second-moment share on that axis should be > 0.9 (common-mode inflates).
    centered covariance share should be < 0.1 (true variance is small).
    This demonstrates exactly the cosine~0.68 inflation we are correcting for.
    """
    D = 20
    n = 60000
    # Small isotropic variance everywhere, but a HUGE constant offset on axis 0.
    X = RNG.normal(size=(n, D)) * 0.1
    X[:, 0] += 10.0  # common-mode: every sample ~+10 on axis 0

    e0 = np.zeros(D); e0[0] = 1.0
    M = second_moment(X)       # raw
    Sigma = covariance(X)      # centered

    share_raw = variance_share(M, e0)
    share_centered = variance_share(Sigma, e0)

    print(f"\n[raw vs centered] share_raw (axis0) = {share_raw:.4f} (expect > 0.9)")
    print(f"[raw vs centered] share_centered (axis0) = {share_centered:.4f} (expect < 0.1)")
    assert share_raw > 0.9
    assert share_centered < 0.1


def test_combined_subspace_variance_planted():
    """Plant variance on axes 0,1,2; pass those as seed dirs. Recover their sum."""
    D = 30
    n = 50000
    std = np.ones(D) * 0.1
    std[0] = std[1] = std[2] = 1.0  # three loud axes
    X = RNG.normal(size=(n, D)) * std
    Sigma = covariance(X)

    # total variance on axes 0,1,2 vs everything
    var_loud = 3 * 1.0
    var_quiet = 27 * 0.01
    expected = var_loud / (var_loud + var_quiet)

    dirs = np.eye(D)[:3]  # e0, e1, e2
    combined = combined_subspace_variance(Sigma, dirs)
    print(f"\n[combined subspace] planted={expected:.4f}  computed={combined:.4f}")
    assert abs(combined - expected) < 0.02


def test_combined_subspace_random_baseline():
    """Random 3-D subspace in D dims captures ~3/D of isotropic variance."""
    D = 100
    n = 40000
    X = RNG.normal(size=(n, D))  # isotropic
    Sigma = covariance(X)
    dirs = RNG.normal(size=(3, D))
    combined = combined_subspace_variance(Sigma, dirs)
    expected = 3.0 / D
    print(f"\n[combined random baseline] 3/D={expected:.4f}  computed={combined:.4f} (D=100 -> ~0.03)")
    assert abs(combined - expected) < 0.01


def test_regression_direction_planted():
    """y is a linear function of axis 5 -> recovered direction aligns with e5."""
    D = 40
    n = 20000
    X = RNG.normal(size=(n, D))
    y = 3.0 * X[:, 5] + 0.01 * RNG.normal(size=n)
    u = regression_direction(X, y, ridge_lambda=1.0)
    e5 = np.zeros(D); e5[5] = 1.0
    alignment = abs(float(u @ e5))
    print(f"\n[regression direction] |cos(recovered, e5)| = {alignment:.4f} (expect ~1.0)")
    assert alignment > 0.98


def test_regression_direction_ridge_stable_under_collinearity():
    """With collinear features, ridge direction stays stable; this just asserts
    it runs and returns a unit vector without blowing up."""
    D = 30
    n = 5000
    base = RNG.normal(size=(n, 1))
    # axes 0..4 are near-copies of base (collinear)
    X = RNG.normal(size=(n, D)) * 0.01
    X[:, :5] += base
    y = base.reshape(-1) * 2.0 + 0.01 * RNG.normal(size=n)
    u = regression_direction(X, y, ridge_lambda=1.0)
    norm = np.linalg.norm(u)
    print(f"\n[ridge stability] ||u|| = {norm:.4f} (expect 1.0), finite={np.all(np.isfinite(u))}")
    assert abs(norm - 1.0) < 1e-6
    assert np.all(np.isfinite(u))


def test_multivariate_r2_centering_invariant():
    """R^2 must be identical whether or not we add a constant offset to X.

    This proves R^2 is the WRONG place to look for the raw-vs-centered
    contrast — that contrast lives in variance_share. Here we shift X by a
    large constant and assert R^2 is unchanged.
    """
    D = 15
    n = 10000
    X = RNG.normal(size=(n, D))
    y = X[:, 3] - 2.0 * X[:, 7] + 0.1 * RNG.normal(size=n)

    r2_a = multivariate_r2(X, y, ridge_lambda=1.0)
    X_shifted = X + 100.0  # massive common-mode
    r2_b = multivariate_r2(X_shifted, y, ridge_lambda=1.0)

    print(f"\n[R2 centering-invariance] r2_original={r2_a:.4f}  r2_shifted={r2_b:.4f} (expect equal)")
    assert abs(r2_a - r2_b) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
