# CLEVR Factor Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a go/no-go probe that measures whether DINOv2 features on CLEVR linearly encode the three Spelke factors (shape/count/layout) and whether those factor directions are *dominant* axes of feature variance — gating all downstream S-DAM pipeline work.

**Architecture:** Pure, unit-tested metric functions (`sdam/probe_metrics.py`) are the heart — they are where a probe can quietly lie, so they get known-answer synthetic tests. A CLEVR scene loader (`data/clevr_scene_loader.py`) turns images + scene graphs into DINOv2 features + factor labels. An orchestration script (`experiments/clevr_factor_probe.py`) runs both encoders × both variance spaces, applies **pre-committed gate thresholds**, and emits JSON + figure + a GREEN/YELLOW/RED verdict.

**Tech Stack:** PyTorch, timm (ViT-S/16), torch.hub `facebookresearch/dinov2` (ViT-S/14), NumPy, scikit-learn (LogisticRegression, Ridge, balanced_accuracy_score), scipy (spearmanr), matplotlib, pytest.

---

## Pre-committed gate thresholds (LOCKED — written before any numbers exist)

These are decided now, on purpose, so a marginal result cannot be rationalized into a green light:

- **Decodability:** per-factor **balanced 3-class accuracy > 0.70** on held-out test (chance ≈ 0.33). For count/layout (continuous) the target is balanced-binned into tertiles so the 0.70 bar means the same thing for all three. Real accuracy must also exceed its **permutation control** by a wide margin.
- **Combined centered variance-explained:** the three orthonormalized seed directions must span **> 10%** of centered test-feature variance (`trace(UᵀΣ_cU)/trace(Σ_c) > 0.10`; a random 3-D subspace expects ≈ 3/384 ≈ 0.8%).
- **Verdict:**
  - 🟢 GREEN → both bars cleared → build the pipeline (separate plan).
  - 🟡 YELLOW → decodable (>0.70) but combined centered variance ≤ 10% → problem #1 is fatal on CLEVR; pivot encoder or write the null.
  - 🔴 RED → any factor decodability ≤ 0.70 or not above permutation → encoder/mapping problem.

---

## File Structure

- **Create** `sdam/probe_metrics.py` — pure metric functions (decodability, permutation, multivariate R², variance share/concentration, combined variance, seed-direction derivation, pairwise-cosine stats). No I/O, no torch model loading. Fully unit-tested.
- **Create** `tests/test_probe_metrics.py` — known-answer tests for every function in `probe_metrics.py`.
- **Create** `data/clevr_scene_loader.py` — CLEVR download/locate, scene-graph → factor labels, DINOv2/ViT feature extraction, caching, train/test split.
- **Create** `tests/test_clevr_factors.py` — pure scene-graph → label tests on synthetic scene dicts.
- **Create** `experiments/clevr_factor_probe.py` — orchestration + gate + JSON/PNG output.
- **Modify** none of `sdam/seeds.py`, `sdam/hopfield.py`, `sdam/model.py`.

Conventions to follow (from existing code): `set_all_seeds(42)` at top of experiments; results to `results/`; JSON via `experiments._common.dump_json`; every experiment JSON carries `"passed"` and `"synthetic"` keys.

---

## Task 1: Pairwise-cosine spread diagnostic

Catches the "DINOv2 trained on photos may still cluster tight on CLEVR renders" risk before anything else.

**Files:**
- Create: `sdam/probe_metrics.py`
- Test: `tests/test_probe_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from sdam.probe_metrics import pairwise_cosine_stats

def test_pairwise_cosine_spread_tight_vs_spread():
    rng = np.random.default_rng(0)
    # Tight cluster: all vectors near one direction -> high mean cosine, low std
    base = rng.normal(size=(1, 64))
    tight = base + 0.01 * rng.normal(size=(200, 64))
    s_tight = pairwise_cosine_stats(tight, sample=2000, seed=0)
    assert s_tight["mean"] > 0.95 and s_tight["std"] < 0.05

    # Spread: isotropic gaussian -> mean cosine near 0, larger std
    spread = rng.normal(size=(200, 64))
    s_spread = pairwise_cosine_stats(spread, sample=2000, seed=0)
    assert abs(s_spread["mean"]) < 0.15 and s_spread["std"] > s_tight["std"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe_metrics.py::test_pairwise_cosine_spread_tight_vs_spread -v`
Expected: FAIL — `ImportError: cannot import name 'pairwise_cosine_stats'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Pure metric functions for the CLEVR factor probe.

No torch, no I/O. Everything operates on NumPy float arrays so the math can be
unit-tested with known-answer synthetic data. This module is where a probe can
quietly lie, so every function here has a known-answer test.
"""
from __future__ import annotations

import numpy as np


def _l2norm(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def pairwise_cosine_stats(X: np.ndarray, sample: int = 20000, seed: int = 0) -> dict:
    """Mean/std/percentiles of off-diagonal pairwise cosine over a random sample of pairs.

    A tight cluster (the CIFAR/ImageNet failure mode) gives mean ~1, std ~0.
    Healthy spread gives a wide distribution with dynamic range.
    """
    rng = np.random.default_rng(seed)
    Xn = _l2norm(np.asarray(X, dtype=np.float64))
    n = Xn.shape[0]
    i = rng.integers(0, n, size=sample)
    j = rng.integers(0, n, size=sample)
    mask = i != j
    cos = np.sum(Xn[i[mask]] * Xn[j[mask]], axis=1)
    return {
        "mean": float(cos.mean()),
        "std": float(cos.std()),
        "p05": float(np.percentile(cos, 5)),
        "p95": float(np.percentile(cos, 95)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe_metrics.py::test_pairwise_cosine_spread_tight_vs_spread -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sdam/probe_metrics.py tests/test_probe_metrics.py
git commit -m "feat(probe): pairwise cosine spread diagnostic"
```

---

## Task 2: Multivariate R² (features ~ factor), centering-invariant

The hard-to-fool "how much feature variance does the factor account for" number.

**Files:**
- Modify: `sdam/probe_metrics.py`
- Test: `tests/test_probe_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from sdam.probe_metrics import multivariate_r2, onehot_design, scalar_design

def test_multivariate_r2_known_fraction():
    rng = np.random.default_rng(1)
    n, D = 4000, 20
    y = rng.normal(size=n)
    # Feature = factor signal along dim 0 + independent noise on dims 1..D-1.
    # Construct so the factor explains a known ~50% of total variance.
    signal = np.zeros((n, D)); signal[:, 0] = 3.0 * y          # var ~ 9
    noise = rng.normal(size=(n, D)); noise[:, 0] = 0.0         # var ~ (D-1)=19 over other dims
    # scale noise so total noise variance ~= signal variance (~9)
    noise *= np.sqrt(9.0 / (D - 1))
    X = signal + noise
    tr, te = slice(0, 3000), slice(3000, n)
    r2 = multivariate_r2(X[tr], y[tr], X[te], y[te], scalar_design)
    assert 0.40 < r2 < 0.60   # ~50%, the planted fraction

def test_multivariate_r2_centering_invariant():
    rng = np.random.default_rng(2)
    n, D = 2000, 10
    y = rng.normal(size=n)
    X = np.outer(y, rng.normal(size=D)) + rng.normal(size=(n, D))
    tr, te = slice(0, 1500), slice(1500, n)
    r2_raw = multivariate_r2(X[tr], y[tr], X[te], y[te], scalar_design)
    Xc = X - X[tr].mean(0)   # center using train mean
    r2_cen = multivariate_r2(Xc[tr], y[tr], Xc[te], y[te], scalar_design)
    assert abs(r2_raw - r2_cen) < 1e-9   # invariant to centering

def test_onehot_design_shape():
    y = np.array([0, 1, 2, 1, 0])
    Phi = onehot_design(y, n_classes=3)
    assert Phi.shape == (5, 3) and np.allclose(Phi.sum(1), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe_metrics.py -k multivariate_r2 -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def scalar_design(y: np.ndarray) -> np.ndarray:
    """Design matrix [1, y] for a continuous scalar factor."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return np.column_stack([np.ones_like(y), y])


def onehot_design(y: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot design (no separate intercept; columns sum to 1) for a categorical factor."""
    y = np.asarray(y).astype(int).reshape(-1)
    Phi = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    Phi[np.arange(y.shape[0]), y] = 1.0
    return Phi


def multivariate_r2(X_tr, y_tr, X_te, y_te, design_fn) -> float:
    """Fraction of TEST feature variance predictable from the factor (linear, fit on train).

    Baseline prediction is the train feature mean (so SS_tot is about the train mean).
    Centering-invariant: subtracting a constant from X shifts X_hat and mu equally.
    """
    X_tr = np.asarray(X_tr, dtype=np.float64); X_te = np.asarray(X_te, dtype=np.float64)
    Phi_tr = design_fn(y_tr); Phi_te = design_fn(y_te)
    B, *_ = np.linalg.lstsq(Phi_tr, X_tr, rcond=None)   # (k, D)
    X_hat = Phi_te @ B
    mu = X_tr.mean(0)
    ss_res = float(((X_te - X_hat) ** 2).sum())
    ss_tot = float(((X_te - mu) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe_metrics.py -k multivariate_r2 -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add sdam/probe_metrics.py tests/test_probe_metrics.py
git commit -m "feat(probe): centering-invariant multivariate R2 (features~factor)"
```

---

## Task 3: Variance share + concentration (raw vs centered) and combined subspace variance

This is the raw-vs-centered contrast: about-origin second moment (includes the common-mode) vs about-mean covariance.

**Files:**
- Modify: `sdam/probe_metrics.py`
- Test: `tests/test_probe_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from sdam.probe_metrics import (
    variance_share, concentration_factor, combined_subspace_variance, second_moment,
)

def test_variance_share_random_direction_is_one_over_D():
    rng = np.random.default_rng(3)
    n, D = 5000, 50
    X = rng.normal(size=(n, D))                  # isotropic, centered ~0
    Sigma = np.cov(X, rowvar=False)
    u = rng.normal(size=D); u /= np.linalg.norm(u)
    conc = concentration_factor(u, Sigma)
    assert 0.6 < conc < 1.6                       # ~1 for a random direction

def test_variance_share_dominant_direction():
    rng = np.random.default_rng(4)
    n, D = 5000, 50
    d = rng.normal(size=D); d /= np.linalg.norm(d)
    # Strong variance along d, weak isotropic elsewhere.
    X = np.outer(rng.normal(scale=5.0, size=n), d) + 0.2 * rng.normal(size=(n, D))
    Sigma = np.cov(X, rowvar=False)
    conc = concentration_factor(d, Sigma)
    assert conc > 10.0                            # dominant axis

def test_raw_vs_centered_common_mode():
    rng = np.random.default_rng(5)
    n, D = 5000, 30
    mean_dir = np.zeros(D); mean_dir[0] = 10.0    # big common-mode on dim 0
    X = mean_dir + rng.normal(size=(n, D))        # tight-ish cluster, mean far from origin
    e0 = np.zeros(D); e0[0] = 1.0
    M = second_moment(X)                          # about origin (raw)
    Sigma = np.cov(X, rowvar=False)               # about mean (centered)
    share_raw = variance_share(e0, M)
    share_cen = variance_share(e0, Sigma)
    assert share_raw > 0.9                         # common-mode dominates raw
    assert share_cen < 0.1                         # vanishes once centered

def test_combined_subspace_variance_orthonormal():
    rng = np.random.default_rng(6)
    n, D = 4000, 40
    U = np.linalg.qr(rng.normal(size=(D, 3)))[0]   # (D,3) orthonormal
    coeffs = rng.normal(scale=[6, 5, 4], size=(n, 3))
    X = coeffs @ U.T + 0.1 * rng.normal(size=(n, D))
    Sigma = np.cov(X, rowvar=False)
    frac = combined_subspace_variance(U, Sigma)
    assert frac > 0.9                              # the 3 dirs hold almost all variance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe_metrics.py -k "variance or concentration or common_mode" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def second_moment(X: np.ndarray) -> np.ndarray:
    """About-origin second-moment matrix E[x xᵀ] (the 'raw' covariance, includes the mean)."""
    X = np.asarray(X, dtype=np.float64)
    return (X.T @ X) / X.shape[0]


def variance_share(u: np.ndarray, S: np.ndarray) -> float:
    """Fraction of total variance (trace S) lying along unit direction u: uᵀSu / trace(S)."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    u = u / max(np.linalg.norm(u), 1e-12)
    return float((u @ S @ u) / max(np.trace(S), 1e-12))


def concentration_factor(u: np.ndarray, S: np.ndarray) -> float:
    """variance_share(u, S) * D. Random direction -> ~1; dominant axis -> >>1."""
    return variance_share(u, S) * S.shape[0]


def combined_subspace_variance(U: np.ndarray, S: np.ndarray) -> float:
    """Fraction of total variance in the span of orthonormal columns of U: tr(UᵀSU)/tr(S)."""
    U = np.asarray(U, dtype=np.float64)
    return float(np.trace(U.T @ S @ U) / max(np.trace(S), 1e-12))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe_metrics.py -k "variance or concentration or common_mode" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add sdam/probe_metrics.py tests/test_probe_metrics.py
git commit -m "feat(probe): variance share, concentration, combined subspace (raw vs centered)"
```

---

## Task 4: Decodability (balanced 3-class) + permutation control

**Files:**
- Modify: `sdam/probe_metrics.py`
- Test: `tests/test_probe_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from sdam.probe_metrics import balanced_tertiles, decodability_balanced_acc

def test_balanced_tertiles_three_balanced_bins():
    y = np.arange(300, dtype=float)
    b = balanced_tertiles(y)
    assert set(np.unique(b)) == {0, 1, 2}
    counts = np.bincount(b)
    assert counts.max() - counts.min() <= 1     # balanced

def test_decodability_separable_high_permutation_chance():
    rng = np.random.default_rng(7)
    n, D = 1200, 16
    cls = rng.integers(0, 3, size=n)
    centers = rng.normal(size=(3, D)) * 4.0
    X = centers[cls] + rng.normal(size=(n, D))
    tr, te = slice(0, 900), slice(900, n)
    real = decodability_balanced_acc(X[tr], cls[tr], X[te], cls[te], n_classes=3, permute=False, seed=0)
    perm = decodability_balanced_acc(X[tr], cls[tr], X[te], cls[te], n_classes=3, permute=True, seed=0)
    assert real > 0.90        # separable -> high
    assert perm < 0.50        # shuffled labels -> near chance (0.33)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe_metrics.py -k "tertiles or decodability" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def balanced_tertiles(y: np.ndarray) -> np.ndarray:
    """Map a continuous scalar to 3 (near-)balanced classes by rank tertiles."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ranks = y.argsort().argsort()
    edges = [len(y) / 3.0, 2 * len(y) / 3.0]
    return np.digitize(ranks, edges).astype(int)


def decodability_balanced_acc(X_tr, y_tr, X_te, y_te, n_classes, permute=False, seed=0) -> float:
    """Balanced held-out accuracy of an L2-logistic probe. If permute, shuffle TRAIN labels
    (the chance-level control). Returns balanced accuracy in [0,1]."""
    X_tr = np.asarray(X_tr, dtype=np.float64); X_te = np.asarray(X_te, dtype=np.float64)
    y_tr = np.asarray(y_tr).astype(int); y_te = np.asarray(y_te).astype(int)
    if permute:
        y_tr = np.random.default_rng(seed).permutation(y_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="auto")
    clf.fit(X_tr, y_tr)
    return float(balanced_accuracy_score(y_te, clf.predict(X_te)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe_metrics.py -k "tertiles or decodability" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sdam/probe_metrics.py tests/test_probe_metrics.py
git commit -m "feat(probe): balanced 3-class decodability + permutation control"
```

---

## Task 5: Seed-direction derivation (regression weight / LDA leading)

**Files:**
- Modify: `sdam/probe_metrics.py`
- Test: `tests/test_probe_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from sdam.probe_metrics import regression_direction, lda_leading_direction

def test_regression_direction_recovers_planted_axis():
    rng = np.random.default_rng(8)
    n, D = 3000, 24
    d = rng.normal(size=D); d /= np.linalg.norm(d)
    y = rng.normal(size=n)
    X = np.outer(y, d) + 0.1 * rng.normal(size=(n, D))
    u = regression_direction(X, y)
    assert abs(abs(float(u @ d)) - 1.0) < 0.05      # recovered up to sign

def test_lda_leading_recovers_class_axis():
    rng = np.random.default_rng(9)
    n, D = 3000, 24
    d = rng.normal(size=D); d /= np.linalg.norm(d)
    cls = rng.integers(0, 3, size=n)
    X = np.outer((cls - 1.0), d) * 3.0 + rng.normal(size=(n, D))
    u = lda_leading_direction(X, cls, n_classes=3)
    assert abs(float(u @ d)) > 0.9                   # aligned with the class-separating axis
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe_metrics.py -k "regression_direction or lda" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def regression_direction(X, y) -> np.ndarray:
    """Unit direction of the OLS weight vector for scalar target y (count / layout seed)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Xc = X - X.mean(0)
    w, *_ = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)
    return w / max(np.linalg.norm(w), 1e-12)


def lda_leading_direction(X, y, n_classes) -> np.ndarray:
    """Leading Fisher-LDA discriminant direction for categorical y (shape seed)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    lda = LinearDiscriminantAnalysis(n_components=min(n_classes - 1, X.shape[1]))
    lda.fit(X, y)
    w = lda.scalings_[:, 0]
    return w / max(np.linalg.norm(w), 1e-12)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe_metrics.py -k "regression_direction or lda" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sdam/probe_metrics.py tests/test_probe_metrics.py
git commit -m "feat(probe): seed-direction derivation (regression weight, LDA leading)"
```

---

## Task 6: CLEVR scene-graph → factor labels (pure function)

**Files:**
- Create: `data/clevr_scene_loader.py`
- Test: `tests/test_clevr_factors.py`

- [ ] **Step 1: Write the failing test**

```python
from data.clevr_scene_loader import scene_to_factors, DOMINANT_SHAPE_CLASSES

def test_scene_to_factors_count_layout_shape():
    scene = {"objects": [
        {"shape": "cube",     "pixel_coords": [40, 100, 8]},
        {"shape": "cube",     "pixel_coords": [60, 110, 8]},
        {"shape": "sphere",   "pixel_coords": [400, 120, 8]},
    ]}
    f = scene_to_factors(scene, image_width=480)
    assert f["count"] == 3
    assert f["shape"] == DOMINANT_SHAPE_CLASSES.index("cube")   # plurality shape
    # mean x = (40+60+400)/3 = 166.67 -> normalized to [-1,1]: 2*166.67/480 - 1 ≈ -0.306
    assert abs(f["layout"] - (2 * (500/3) / 480 - 1)) < 1e-6

def test_scene_to_factors_dominant_shape_tiebreak_is_deterministic():
    scene = {"objects": [
        {"shape": "cube",   "pixel_coords": [10, 10, 1]},
        {"shape": "sphere", "pixel_coords": [20, 20, 1]},
    ]}
    f1 = scene_to_factors(scene, image_width=480)
    f2 = scene_to_factors(scene, image_width=480)
    assert f1["shape"] == f2["shape"]   # deterministic tiebreak
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clevr_factors.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
"""CLEVR images + scene graphs -> DINOv2/ViT features + Spelke factor labels.

Factor mapping (Milestone 1, 3 systems; Agentness out of scope):
  Objectness -> dominant object shape (cube/sphere/cylinder)   [proxy]
  Numerosity -> object count
  Geometry   -> layout = mean object x normalized to [-1, 1]
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np

DOMINANT_SHAPE_CLASSES = ["cube", "sphere", "cylinder"]


def scene_to_factors(scene: dict, image_width: int) -> dict:
    """Pure: CLEVR scene-graph dict -> {count, layout, shape}. Deterministic tiebreak."""
    objs = scene["objects"]
    count = len(objs)
    xs = [float(o["pixel_coords"][0]) for o in objs]
    mean_x = sum(xs) / max(len(xs), 1)
    layout = 2.0 * mean_x / image_width - 1.0
    counts = Counter(o["shape"] for o in objs)
    # plurality; tie broken by DOMINANT_SHAPE_CLASSES order (deterministic)
    best = max(DOMINANT_SHAPE_CLASSES, key=lambda s: (counts.get(s, 0), -DOMINANT_SHAPE_CLASSES.index(s)))
    return {"count": count, "layout": layout, "shape": DOMINANT_SHAPE_CLASSES.index(best)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clevr_factors.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add data/clevr_scene_loader.py tests/test_clevr_factors.py
git commit -m "feat(probe): CLEVR scene-graph -> Spelke factor labels"
```

---

## Task 7: Encoders + CLEVR feature extraction (integration; Colab-run)

This task touches the network and large downloads — it is exercised on Colab/A100, not in CI. Keep functions importable and side-effect-free until called.

**Files:**
- Modify: `data/clevr_scene_loader.py`
- Test: `tests/test_clevr_factors.py` (import-only smoke test; no download)

- [ ] **Step 1: Write the failing test**

```python
import data.clevr_scene_loader as L

def test_encoder_registry_lists_both():
    assert set(L.ENCODERS) == {"dinov2_vits14", "imagenet_vits16"}
    assert L.ENCODERS["dinov2_vits14"]["dim"] == 384
    assert L.ENCODERS["imagenet_vits16"]["dim"] == 384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clevr_factors.py::test_encoder_registry_lists_both -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'ENCODERS'`.

- [ ] **Step 3: Write minimal implementation**

```python
import torch

ENCODERS = {
    "dinov2_vits14": {"dim": 384, "img_size": 224},   # torch.hub facebookresearch/dinov2
    "imagenet_vits16": {"dim": 384, "img_size": 224},  # timm vit_small_patch16_224 (comparison)
}


def load_encoder(name: str, device: str):
    """Return a frozen, eval-mode encoder returning a (B,384) CLS embedding."""
    if name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif name == "imagenet_vits16":
        import timm
        m = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    else:
        raise ValueError(f"unknown encoder {name!r}")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def extract_features(encoder_name, clevr_root, n_scenes, cache_path, device="cpu", seed=42):
    """Locate CLEVR images+scenes under clevr_root, extract CLS features for n_scenes images,
    compute factor labels, cache {features, count, layout, shape, encoder} to cache_path.

    Requires CLEVR on disk: <root>/scenes/CLEVR_*_scenes.json and <root>/images/<split>/.
    Raises FileNotFoundError otherwise (caller decides how to obtain CLEVR)."""
    import json
    from PIL import Image
    import torchvision.transforms as T

    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    scenes_files = [f for f in os.listdir(os.path.join(clevr_root, "scenes"))] \
        if os.path.isdir(os.path.join(clevr_root, "scenes")) else []
    if not scenes_files:
        raise FileNotFoundError(
            f"CLEVR scene graphs not found under {clevr_root}/scenes. "
            "Download CLEVR v1.0 (images + scenes) first."
        )
    scenes_path = os.path.join(clevr_root, "scenes", sorted(scenes_files)[0])
    with open(scenes_path) as f:
        scenes = json.load(f)["scenes"][:n_scenes]

    spec = ENCODERS[encoder_name]
    enc = load_encoder(encoder_name, device)
    tfm = T.Compose([
        T.Resize(spec["img_size"]), T.CenterCrop(spec["img_size"]), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    split_dir = os.path.join(clevr_root, "images")
    feats, count, layout, shape = [], [], [], []
    import torch as _t
    for sc in scenes:
        img_path = _find_image(split_dir, sc["image_filename"])
        img = Image.open(img_path).convert("RGB")
        with _t.no_grad():
            x = tfm(img).unsqueeze(0).to(device)
            emb = enc(x).reshape(1, -1)[:, : spec["dim"]].cpu().squeeze(0)
        f = scene_to_factors(sc, image_width=img.width)
        feats.append(emb); count.append(f["count"]); layout.append(f["layout"]); shape.append(f["shape"])
    out = {
        "features": _t.stack(feats), "count": _t.tensor(count),
        "layout": _t.tensor(layout, dtype=_t.float32), "shape": _t.tensor(shape),
        "encoder": encoder_name,
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    _t.save(out, cache_path)
    return out


def _find_image(images_root, filename):
    import glob
    hits = glob.glob(os.path.join(images_root, "**", filename), recursive=True)
    if not hits:
        raise FileNotFoundError(f"CLEVR image {filename} not found under {images_root}")
    return hits[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clevr_factors.py::test_encoder_registry_lists_both -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/clevr_scene_loader.py tests/test_clevr_factors.py
git commit -m "feat(probe): DINOv2 + ViT encoders and CLEVR feature extraction"
```

---

## Task 8: Probe orchestration + locked gate + outputs

**Files:**
- Create: `experiments/clevr_factor_probe.py`
- Test: `tests/test_probe_metrics.py` (gate-logic unit test, no data)

- [ ] **Step 1: Write the failing test**

```python
from sdam.probe_metrics import probe_verdict

def test_probe_verdict_gate_logic():
    # all decodable + combined variance > 0.10 -> GREEN
    assert probe_verdict({"shape": 0.8, "count": 0.75, "layout": 0.72}, 0.12)["verdict"] == "GREEN"
    # decodable but low variance -> YELLOW
    assert probe_verdict({"shape": 0.8, "count": 0.75, "layout": 0.72}, 0.04)["verdict"] == "YELLOW"
    # a factor below 0.70 -> RED
    assert probe_verdict({"shape": 0.8, "count": 0.60, "layout": 0.72}, 0.12)["verdict"] == "RED"
    assert probe_verdict({"shape": 0.8, "count": 0.75, "layout": 0.72}, 0.12)["passed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe_metrics.py::test_probe_verdict_gate_logic -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

First add the gate function to `sdam/probe_metrics.py`:

```python
DECODABILITY_BAR = 0.70           # per-factor balanced 3-class accuracy
COMBINED_VARIANCE_BAR = 0.10      # centered combined variance share of the 3 seed dirs

def probe_verdict(decodability: dict, combined_centered_variance: float) -> dict:
    """Apply the LOCKED gate. decodability: {factor: balanced_acc}."""
    all_decodable = all(v > DECODABILITY_BAR for v in decodability.values())
    var_ok = combined_centered_variance > COMBINED_VARIANCE_BAR
    if all_decodable and var_ok:
        verdict = "GREEN"
    elif all_decodable and not var_ok:
        verdict = "YELLOW"
    else:
        verdict = "RED"
    return {"verdict": verdict, "passed": verdict == "GREEN",
            "decodability": decodability,
            "combined_centered_variance": combined_centered_variance}
```

Then the orchestration script `experiments/clevr_factor_probe.py`:

```python
"""CLEVR factor probe (go/no-go). Gates all S-DAM pipeline work.

Runs both encoders (DINOv2 primary, ImageNet ViT comparison) x both variance spaces.
Outputs results/clevr_factor_probe.json (+ .png). Verdict GREEN/YELLOW/RED via the
LOCKED thresholds in sdam.probe_metrics. No threshold-hacking.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments._common import ensure_results_dir, dump_json, load_config
from sdam.utils import set_all_seeds
from sdam.probe_metrics import (
    pairwise_cosine_stats, multivariate_r2, scalar_design, onehot_design,
    variance_share, concentration_factor, combined_subspace_variance, second_moment,
    balanced_tertiles, decodability_balanced_acc, regression_direction,
    lda_leading_direction, probe_verdict,
)
from data.clevr_scene_loader import extract_features, DOMINANT_SHAPE_CLASSES

CLEVR_ROOT = os.environ.get("CLEVR_ROOT", "data/CLEVR_v1.0")
N_SCENES = int(os.environ.get("CLEVR_N_SCENES", "5000"))
PRIMARY_ENCODER = "dinov2_vits14"


def _split(n, frac=0.8, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(frac * n)
    return idx[:k], idx[k:]


def _factor_labels(data, factor):
    y = data[factor].numpy()
    if factor == "shape":
        return y.astype(int), 3, "lda"
    return y.astype(float), 3, "reg"   # count/layout -> tertiles for the 3-class bar


def run() -> dict:
    set_all_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"experiment": "clevr_factor_probe", "synthetic": False, "encoders": {}}

    for enc in ["dinov2_vits14", "imagenet_vits16"]:
        cache = f"data/clevr_feats_{enc}.pt"
        data = extract_features(enc, CLEVR_ROOT, N_SCENES, cache, device=device)
        X = data["features"].numpy().astype(np.float64)
        tr, te = _split(X.shape[0])
        mu = X[tr].mean(0)
        Xc = X - mu                                   # centered with TRAIN mean
        spread = pairwise_cosine_stats(X)
        Sigma_c = np.cov(Xc[te], rowvar=False)
        M_raw = second_moment(X[te])

        enc_res = {"pairwise_cosine": spread, "factors": {}}
        seed_dirs = []
        decod = {}
        for factor in ["shape", "count", "layout"]:
            y_raw, n_cls, kind = _factor_labels(data, factor)
            # 3-class decodability target
            y_cls = y_raw.astype(int) if factor == "shape" else balanced_tertiles(y_raw)
            real = decodability_balanced_acc(X[tr], y_cls[tr], X[te], y_cls[te], 3, permute=False, seed=0)
            perm = decodability_balanced_acc(X[tr], y_cls[tr], X[te], y_cls[te], 3, permute=True, seed=0)
            decod[factor] = real
            # multivariate R² (centering-invariant)
            if factor == "shape":
                r2 = multivariate_r2(X[tr], y_raw[tr], X[te], y_raw[te],
                                     lambda v: onehot_design(v, 3))
                u = lda_leading_direction(X[tr], y_raw[tr].astype(int), 3)
            else:
                r2 = multivariate_r2(X[tr], y_raw[tr], X[te], y_raw[te], scalar_design)
                u = regression_direction(X[tr], y_raw[tr])
            seed_dirs.append(u)
            enc_res["factors"][factor] = {
                "decodability": real, "permutation": perm, "multivariate_r2": r2,
                "concentration_raw": concentration_factor(u, M_raw),
                "concentration_centered": concentration_factor(u, Sigma_c),
                "variance_share_centered": variance_share(u, Sigma_c),
            }
        U = np.linalg.qr(np.column_stack(seed_dirs))[0]
        combined_cen = combined_subspace_variance(U, Sigma_c)
        combined_raw = combined_subspace_variance(U, M_raw)
        verdict = probe_verdict(decod, combined_cen)
        enc_res["combined_centered_variance"] = combined_cen
        enc_res["combined_raw_variance"] = combined_raw
        enc_res["verdict"] = verdict
        enc_res["seed_angles_deg"] = _angles(seed_dirs)
        results["encoders"][enc] = enc_res

    primary = results["encoders"][PRIMARY_ENCODER]
    results["verdict"] = primary["verdict"]["verdict"]
    results["passed"] = primary["verdict"]["passed"]

    _print_report(results)
    out_dir = ensure_results_dir(load_config())
    dump_json(results, os.path.join(out_dir, "clevr_factor_probe.json"))
    _plot(results, os.path.join(out_dir, "clevr_factor_probe.png"))
    return results


def _angles(dirs):
    out = {}
    names = ["shape", "count", "layout"]
    for a in range(3):
        for b in range(a + 1, 3):
            c = float(np.clip(np.dot(dirs[a], dirs[b]), -1, 1))
            out[f"{names[a]}-{names[b]}"] = float(np.degrees(np.arccos(abs(c))))
    return out


def _print_report(results):
    print("=" * 64)
    print("CLEVR FACTOR PROBE — go/no-go")
    for enc, r in results["encoders"].items():
        pc = r["pairwise_cosine"]
        print(f"\n[{enc}]  pairwise cos mean={pc['mean']:.3f} std={pc['std']:.3f} "
              f"(tight cluster if mean~1, std~0)")
        for f, m in r["factors"].items():
            print(f"  {f:7s} decod={m['decodability']:.3f} (perm {m['permutation']:.3f})  "
                  f"R2={m['multivariate_r2']:.3f}  conc_cen={m['concentration_centered']:.1f}x")
        print(f"  combined centered variance = {r['combined_centered_variance']:.3f} "
              f"(bar 0.10)  -> {r['verdict']['verdict']}")
    print("=" * 64)
    print(f"PRIMARY ({PRIMARY_ENCODER}) VERDICT: {results['verdict']}  passed={results['passed']}")
    print("=" * 64)


def _plot(results, path):
    encs = list(results["encoders"])
    factors = ["shape", "count", "layout"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(factors)); w = 0.35
    for k, enc in enumerate(encs):
        r = results["encoders"][enc]
        dec = [r["factors"][f]["decodability"] for f in factors]
        perm = [r["factors"][f]["permutation"] for f in factors]
        axes[0].bar(x + (k - 0.5) * w, dec, w, label=f"{enc} real")
        axes[0].plot(x + (k - 0.5) * w, perm, "k_", ms=14)
    axes[0].axhline(0.70, color="red", ls="--", lw=0.8, label="bar 0.70")
    axes[0].set_xticks(x); axes[0].set_xticklabels(factors)
    axes[0].set_title("Decodability (bars) vs permutation (ticks)"); axes[0].legend(fontsize=7)
    for k, enc in enumerate(encs):
        r = results["encoders"][enc]
        conc = [r["factors"][f]["concentration_centered"] for f in factors]
        axes[1].bar(x + (k - 0.5) * w, conc, w, label=enc)
    axes[1].axhline(1.0, color="gray", ls="--", lw=0.8, label="random (1x)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(factors)
    axes[1].set_title("Centered concentration factor (>>1 = dominant axis)"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe_metrics.py::test_probe_verdict_gate_logic -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite**

Run: `pytest tests/test_probe_metrics.py tests/test_clevr_factors.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/clevr_factor_probe.py sdam/probe_metrics.py tests/test_probe_metrics.py
git commit -m "feat(probe): orchestration, locked go/no-go gate, JSON + figure outputs"
```

---

## Task 9: Colab cell for the probe (run on A100)

**Files:**
- Modify: `notebooks/colab_runner.ipynb` (add a probe cell + CLEVR download cell)

- [ ] **Step 1: Add a CLEVR download cell** (before the probe cell)

```python
# CLEVR v1.0 (images + scene graphs). ~18GB; downloads once.
import os
if not os.path.exists("data/CLEVR_v1.0/scenes"):
    !mkdir -p data
    !wget -q https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O data/CLEVR_v1.0.zip
    !cd data && unzip -q CLEVR_v1.0.zip && rm CLEVR_v1.0.zip
print("CLEVR ready:", os.path.exists("data/CLEVR_v1.0/scenes"))
```

- [ ] **Step 2: Add the probe cell**

```python
!pip install -q scikit-learn
!CLEVR_ROOT=data/CLEVR_v1.0 CLEVR_N_SCENES=5000 python experiments/clevr_factor_probe.py
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_runner.ipynb
git commit -m "feat(probe): Colab cells for CLEVR download + factor probe"
```

---

## Self-Review

**Spec coverage:**
- Pairwise-cosine spread check (your addition) → Task 1. ✓
- Decodability + permutation → Task 4; reported per factor in Task 8. ✓
- Multivariate R² (centering-invariant) → Task 2. ✓
- Variance share/concentration raw vs centered → Task 3. ✓
- Combined centered variance + locked 10% bar → Task 3 (metric) + Task 8 (`probe_verdict`). ✓
- Per-factor decodability 70% bar (incl. balanced binning for regression factors) → Task 4 (`balanced_tertiles`) + Task 8 gate. ✓
- Seed directions (regression / LDA) → Task 5. ✓
- Factor independence (angles) → Task 8 `_angles`. ✓
- 3 factors, Agentness out of scope → Task 6. ✓
- DINOv2 primary + ImageNet comparison → Task 7 + Task 8 loop. ✓
- JSON with `passed`/`synthetic`, PNG → Task 8. ✓

**Placeholder scan:** none — all steps contain runnable code/commands.

**Type consistency:** `scene_to_factors` returns `{count,layout,shape}` used consistently in Tasks 6–8; `probe_verdict(decodability_dict, combined_variance)` signature matches Task 8 call; `ENCODERS` dim 384 consistent with `multivariate_r2`/feature shapes.

**Deferred (NOT in this plan, by design):** the full 3-system memory pipeline (Phase 1/2/3 on test split) — gets its own plan only if the probe returns GREEN.
