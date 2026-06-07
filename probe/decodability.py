"""Decodability probe for the S-DAM feasibility gate.

For each Spelke factor we produce ONE comparable number: held-out balanced
3-class accuracy (chance ~= 0.33, gate bar 0.70). This makes shape (native
3-class) and count/layout (continuous -> tertile-binned) directly comparable.

Design decisions (locked with reviewer before coding):
  - StandardScaler fit on TRAIN only, applied to test. DINOv2 is O(1)-scaled
    but we don't economize on the gate that decides whether to build the
    pipeline.
  - C is CV-tuned (3-fold) over {0.01, 0.1, 1, 10} on the TRAIN set. Guards
    against false NEGATIVES (a bad fixed C under-fitting a decodable factor) —
    a false negative here kills the project.
  - Permutation control: shuffle train labels, refit, score on real test.
    Guards against false POSITIVES (384-d probe fitting noise). Real must beat
    permutation by a wide margin.
  - balanced_accuracy_score, not raw accuracy (CLEVR labels are imbalanced).
  - For count/layout we ALSO keep continuous Spearman rho + multivariate R^2,
    so binning doesn't throw away ordinal signal — the gate uses the
    comparable accuracy, but we don't lose the real signal.
  - Tertile boundaries are LOGGED so a degenerate/skewed binning can't hide
    behind a healthy-looking balanced_accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from probe.variance import multivariate_r2


def balanced_tertiles(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin a continuous/ordinal target into 3 rank-balanced classes.

    Returns (labels in {0,1,2}, boundary_values). Uses rank-based tertiles so
    classes are as balanced as ties allow. boundary_values are the two cut
    points in the ORIGINAL units (for logging / degeneracy detection).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ranks = y.argsort().argsort()  # 0..n-1 rank positions
    n = len(y)
    # tertile cut ranks
    c1, c2 = n / 3.0, 2.0 * n / 3.0
    labels = np.zeros(n, dtype=int)
    labels[ranks >= c1] = 1
    labels[ranks >= c2] = 2
    # boundary values in original units: value at the rank cut points
    order = np.argsort(y)
    b1 = float(y[order[int(c1)]]) if int(c1) < n else float(y.max())
    b2 = float(y[order[int(c2)]]) if int(c2) < n else float(y.max())
    return labels, np.array([b1, b2])


@dataclass
class DecodabilityResult:
    factor: str
    balanced_acc: float
    permutation_acc: float
    margin: float                 # balanced_acc - permutation_acc
    best_C: float
    class_counts: dict
    tertile_boundaries: Optional[list] = None  # None for native-categorical (shape)
    spearman_rho: Optional[float] = None       # continuous factors only
    multivariate_r2: Optional[float] = None     # continuous factors only
    chance: float = 1.0 / 3.0

    def passed(self, bar: float = 0.70, min_margin: float = 0.15) -> bool:
        return self.balanced_acc >= bar and self.margin >= min_margin


def _cv_tuned_logistic(Xtr: np.ndarray, ytr: np.ndarray, seed: int = 42) -> tuple[LogisticRegression, float]:
    """3-fold CV over C on the TRAIN set. Returns (fitted best estimator, best_C)."""
    n_classes = len(np.unique(ytr))
    # ensure at least 2 samples per class per fold; fall back to fewer folds
    min_count = np.min(np.bincount(ytr))
    n_splits = int(min(3, max(2, min_count)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000),
        param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(Xtr, ytr)
    return grid.best_estimator_, float(grid.best_params_["C"])


def decodability(
    X: np.ndarray,
    y_raw: np.ndarray,
    factor: str,
    is_categorical: bool,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
) -> DecodabilityResult:
    """Compute held-out balanced 3-class decodability for one factor.

    X:  (n, D) features
    y_raw: (n,) factor values. For shape these are class ids {0,1,2}; for
           count/layout these are continuous/ordinal and get tertile-binned.
    is_categorical: True for shape (native 3-class), False for count/layout.
    train_idx / test_idx: explicit split (caller controls it for comparability).
    """
    X = np.asarray(X, dtype=np.float64)
    y_raw = np.asarray(y_raw).reshape(-1)

    tertile_boundaries = None
    spearman_rho = None
    mv_r2 = None

    if is_categorical:
        y = y_raw.astype(int)
    else:
        y, boundaries = balanced_tertiles(y_raw)
        tertile_boundaries = boundaries.tolist()
        # continuous signal preserved alongside (computed on full set)
        # Spearman of the FIRST PC of X vs y_raw is not meaningful; instead use
        # multivariate R^2 (ridge) as the multivariate continuous signal, and
        # Spearman between ridge prediction and y_raw for a rank correlation.
        from probe.variance import regression_direction
        u = regression_direction(X[train_idx], y_raw[train_idx], ridge_lambda=1.0)
        proj_all = X @ u
        spearman_rho = float(spearmanr(proj_all[test_idx], y_raw[test_idx]).statistic)
        mv_r2 = float(multivariate_r2(X[train_idx], y_raw[train_idx], ridge_lambda=1.0))

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    # StandardScaler fit on TRAIN only
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Real probe: CV-tuned logistic
    clf, best_C = _cv_tuned_logistic(Xtr_s, ytr, seed=seed)
    pred = clf.predict(Xte_s)
    bal_acc = float(balanced_accuracy_score(yte, pred))

    # Permutation control: shuffle TRAIN labels, refit at the same best_C, score real test
    rng = np.random.default_rng(seed)
    ytr_perm = ytr.copy()
    rng.shuffle(ytr_perm)
    clf_perm = LogisticRegression(C=best_C, max_iter=2000)
    clf_perm.fit(Xtr_s, ytr_perm)
    perm_acc = float(balanced_accuracy_score(yte, clf_perm.predict(Xte_s)))

    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}

    return DecodabilityResult(
        factor=factor,
        balanced_acc=bal_acc,
        permutation_acc=perm_acc,
        margin=bal_acc - perm_acc,
        best_C=best_C,
        class_counts=counts,
        tertile_boundaries=tertile_boundaries,
        spearman_rho=spearman_rho,
        multivariate_r2=mv_r2,
    )
