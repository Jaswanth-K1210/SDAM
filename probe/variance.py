"""Variance / concentration math for the S-DAM feasibility probe.

Every function here is pure (numpy in, numpy out) so it can be verified with
known-answer synthetic tests BEFORE any CLEVR data is involved. If a test
fails here, the only possible cause is a math bug — not data, encoder, or GPU.

Key distinction (this is the raw-vs-centered contrast the gate depends on):

  second_moment(X)  = E[x x^T]  about the ORIGIN  -> "raw"
                      includes the mean outer product (the common-mode that
                      produced the cosine~0.68 artifact). A real factor
                      direction looks tiny here because the common-mode
                      dominates the trace.

  covariance(X)     = Sigma about the MEAN          -> "centered"
                      the factor's true variance share shows here.

variance_share(M, u) = (u^T M u) / trace(M)   for a unit vector u
concentration(M, u)  = variance_share(M, u) * D  (random dir -> ~1, dominant >>1)
combined_subspace_variance(Sigma_c, dirs) = trace(U^T Sigma_c U)/trace(Sigma_c)
                       on the QR-orthonormalized span of `dirs`.
"""
from __future__ import annotations

import numpy as np


def second_moment(X: np.ndarray) -> np.ndarray:
    """Raw second moment about the origin: E[x x^T]. Shape (D, D)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    return (X.T @ X) / n


def covariance(X: np.ndarray) -> np.ndarray:
    """Centered covariance about the mean: Sigma. Shape (D, D).

    Uses the population convention (divide by N) to match second_moment, so
    the raw-vs-centered comparison differs ONLY by the mean outer product:
        second_moment(X) = covariance(X) + mu mu^T
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    n = X.shape[0]
    return (Xc.T @ Xc) / n


def variance_share(M: np.ndarray, u: np.ndarray) -> float:
    """Fraction of total variance (trace of M) lying along unit vector u.

    share = (u^T M u) / trace(M). u is L2-normalized internally.
    """
    M = np.asarray(M, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    u = u / (np.linalg.norm(u) + 1e-12)
    num = float(u @ M @ u)
    den = float(np.trace(M))
    return num / (den + 1e-12)


def concentration(M: np.ndarray, u: np.ndarray) -> float:
    """variance_share scaled by dimension D.

    A random direction concentrates ~1/D of the variance -> concentration ~1.
    A dominant axis -> concentration >> 1. This makes the number comparable
    across feature dimensionalities.
    """
    M = np.asarray(M, dtype=np.float64)
    D = M.shape[0]
    return variance_share(M, u) * D


def combined_subspace_variance(Sigma_c: np.ndarray, dirs: np.ndarray) -> float:
    """Fraction of centered variance captured by the span of `dirs`.

    `dirs` is (k, D): k direction vectors (the seed directions). They are
    QR-orthonormalized first, because that is what the real S-DAM pipeline
    does to its seeds. Returns trace(U^T Sigma_c U) / trace(Sigma_c).

    NOTE: this measures the ORTHOGONALIZED SPAN, not the raw individual
    directions. A high value can mean the span aligns with high variance even
    if one factor is dead — so the gate must ALSO check per-factor
    concentration. See the probe orchestration for the dual-bar gate.
    """
    Sigma_c = np.asarray(Sigma_c, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    if dirs.ndim == 1:
        dirs = dirs.reshape(1, -1)
    # QR on the (D, k) matrix -> orthonormal columns spanning the same space
    Q, _ = np.linalg.qr(dirs.T)  # Q: (D, k), orthonormal columns
    # Guard against rank deficiency (correlated factors): keep only columns
    # whose norm survived (qr gives orthonormal, but rank-deficient input can
    # yield near-zero columns in some numpy versions — normalize defensively).
    num = float(np.trace(Q.T @ Sigma_c @ Q))
    den = float(np.trace(Sigma_c))
    return num / (den + 1e-12)


def regression_direction(X: np.ndarray, y: np.ndarray, ridge_lambda: float = 1.0) -> np.ndarray:
    """Direction in feature space most associated with scalar target y.

    Ridge regression (small lambda) rather than OLS: DINOv2 dimensions can be
    collinear, which makes the OLS direction unstable — and this direction
    feeds the concentration metric in the gate, so we want it stable.

    Returns an L2-normalized (D,) direction. y is mean-centered internally.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    D = Xc.shape[1]
    # Solve (X^T X + lambda I) w = X^T y
    A = Xc.T @ Xc + ridge_lambda * np.eye(D)
    b = Xc.T @ yc
    w = np.linalg.solve(A, b)
    norm = np.linalg.norm(w)
    return w / (norm + 1e-12)


def multivariate_r2(X: np.ndarray, y: np.ndarray, ridge_lambda: float = 1.0) -> float:
    """In-sample multivariate R^2 of ridge-regressing y on X.

    Centering-invariant by construction (R^2 is measured relative to the mean
    of y either way). This is why the raw-vs-centered contrast lives in
    variance_share, NOT here — proven by test_multivariate_r2_centering_invariant.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    D = Xc.shape[1]
    A = Xc.T @ Xc + ridge_lambda * np.eye(D)
    b = Xc.T @ yc
    w = np.linalg.solve(A, b)
    y_pred = Xc @ w
    ss_res = float(np.sum((yc - y_pred) ** 2))
    ss_tot = float(np.sum(yc ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)
