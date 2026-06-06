# CLEVR Factor Probe + 3-System S-DAM — Design Spec

**Date:** 2026-06-07
**Status:** Approved design (Milestone 1)
**Author:** Jaswanth K. + Claude

---

## 0. Why this exists (the honest framing)

The original S-DAM experiments "passed" on synthetic data only because that data was
built around the seed directions (`centers = canonical_spelke_directions`) — the test was
rigged to match the seeds. On real CIFAR-10 features every experiment fails, for three
compounding reasons:

1. **Negligible seed subspace.** 4 seeds in 512-d remove ≈0.8% of each vector, so the
   stored residual ≈ the raw input. Spelke/random/zeroed seeds give *identical* numbers.
2. **Feature-geometry artifact.** Post-ReLU ImageNet features sit in the positive orthant
   → any two images have cosine ≈0.68. Phase 2's "interference" is just baseline similarity.
3. **Domain mismatch.** CIFAR's classes (airplane/cat/…) have nothing to do with Spelke's
   core-knowledge systems, so seeds aligned to those systems are no more relevant than
   random directions.

This spec defines a scientifically honest test of the *real* claim, on a *real*
Spelke-structured dataset, with a cheap go/no-go probe gating the expensive pipeline.

**The claim under test:** seeds aligned to a dataset's generative factors improve
associative memory more than random seeds — measured on **held-out** data so it is not
circular.

**Outcome policy:** A positive result *or* a sharply-diagnosed null are both acceptable and
publishable. No threshold-hacking. Report exactly what happens.

---

## 1. Scope decisions (locked)

- **Dataset:** CLEVR v1.0 (images + **scene-graph** annotations; questions not used).
- **Systems tested: 3, not 4.** CLEVR has no agents, so **Agentness is explicitly out of
  scope** and noted as a known limitation. The `cylinder = agent` heuristic from the old
  loader is abandoned as indefensible.
- **Factor → CLEVR operationalization:**
  | Spelke system | CLEVR factor | Probe target | Note |
  |---|---|---|---|
  | Objectness | dominant object **shape** (cube/sphere/cylinder) | 3-class | ⚠️ proxy — shape is object *identity*, adjacent to but not identical to Spelke Objectness (cohesion/permanence). Flagged in write-up. |
  | Numerosity | object **count** (3–10) | regression | clean |
  | Geometry | **layout** (mean object x-position, left↔right balance) | regression | clean |
- **Encoder: DINOv2 ViT-S/14 (self-supervised), 384-d, frozen.** ImageNet-supervised
  ViT-S/16 CLS discards count/layout; DINOv2 preserves scene structure. ImageNet ViT-S/16
  is retained **only as a documented comparison** in the probe (demonstrates encoder
  dependence — itself a publishable finding).
- **Feature preprocessing:** mean-center (train mean only) + L2-normalize. **No whitening**
  (it would erase the structure the seeds must capture).
- **Model architecture unchanged.** `SpelkeSeedLayer`/`HopfieldLayer`/`SDAM` untouched;
  `n_systems=3` already supported. Derived seeds injected via the existing
  `m.ssl.seeds = nn.Parameter(...)` path.

---

## 2. Step 1 — The probe (go/no-go). Build this FIRST.

Single script `experiments/clevr_factor_probe.py` → one JSON + one figure.
Runs on ~3–5k CLEVR scenes, **80/20 train/test split by image** (no leakage).
Run for **both** encoders (DINOv2 primary, ImageNet ViT comparison) and **both** feature
spaces (raw L2-norm; centered-then-L2-norm, train mean only).

### 2.1 Per-image factor labels (each reduced to ONE target → one seed direction)
- **count:** integer #objects. Ridge regression.
- **layout:** mean object x normalized to [−1, 1]. Ridge regression.
- **shape:** scene **dominant-shape** 3-class label. L2-logistic classification. Single
  seed direction = **leading Fisher-LDA discriminant** (top generalized eigenvector).
  *Open alternative:* scalar "sphere-fraction" composition score for a cleaner single axis;
  default is dominant-class+LDA unless probe shows it is unstable.

### 2.2 Metric 1 — Decodability (linear, held-out)
Fit on train (L2 strength via train-only CV), evaluate on test:
- count, layout: **held-out R²** + **Spearman ρ**.
- shape: **balanced accuracy** + **macro-F1** (not raw accuracy — labels imbalanced).
- **Permutation control:** refit on shuffled labels, same pipeline; factor counts as
  decodable only if real ≫ permutation. Catches a 384-d probe fitting noise.

### 2.3 Metric 2 — Variance-explained (is the factor a DOMINANT axis? — the fatal question)
Computed in **centered** space (raw reported only to show common-mode inflation):

**(a) Multivariate R² of features-on-factor** (the hard-to-fool number):
```
R²_multi = 1 − ( Σ_i ||x_i − x̂_i||² ) / ( Σ_i ||x_i − μ||² )
```
where x̂_i = least-squares prediction of the *feature vector* from the factor label
(fit on train, evaluated on test). Fraction of feature variance the factor accounts for.
With permutation control.

**(b) Variance share along the derived seed direction u** (unit):
```
share = (uᵀ Σ u) / trace(Σ),     Σ = Cov(centered test features)
concentration = share × D          (D = 384)
```
A random direction expects share = 1/D → concentration ≈ 1. concentration ≫ 1 (>~5–10×)
means the factor is a genuinely dominant axis.

**Why both:** (a) measures "factor → feature variance"; (b) measures "variance along the
prediction direction." A factor can be decodable (good Metric 1) yet low-variance (b≈1) —
the exact trap where the full pipeline later shows nothing. The regression direction ≠ the
max-variance direction, so projecting onto u alone would lie; reporting (a) and (b)
together does not.

### 2.4 Metric 3 — Factor independence (sanity)
- Correlation matrix among the 3 scalar factor targets.
- Pairwise angles between the 3 derived seed directions. Near-collinear seeds are not
  independent "systems" and would be distorted by later orthonormalization — flag now.

### 2.5 Go/no-go logic (provisional thresholds; actual numbers inform the call)
- 🟢 **GREEN** → build §3 pipeline: all 3 factors decodable (real ≫ permutation) AND ≥1
  factor with concentration ≫ random (>~5×) / multivariate R² >~5% in centered space.
- 🟡 **YELLOW** → decodable but low-variance: pivot encoder (try mean-pooled DINOv2 patch
  tokens, CLIP) or write the honest null with the variance numbers as the explanation.
- 🔴 **RED** → not decodable above permutation: encoder cannot test the hypothesis on CLEVR.

### 2.6 Deliverables
- `experiments/clevr_factor_probe.py`
- `results/clevr_factor_probe.json` — all metrics, both encoders, raw+centered, +permutation
  controls, +`"verdict": "GREEN"|"YELLOW"|"RED"`, +`"passed": bool`.
- `results/clevr_factor_probe.png` — decodability vs permutation per factor; concentration
  vs random baseline.

---

## 3. Step 2 — Full 3-system pipeline (build ONLY on GREEN)

Held-out throughout: derive seeds on **train**, evaluate memory on **test**.

### 3.1 New modules
- `data/clevr_scene_loader.py` — CLEVR images + scene graphs → DINOv2 features + factor
  labels; cached; train/test split.
- `sdam/factor_seeds.py` — fit linear probes on train → 3 factor directions → orthonormalize
  (QR) → (3, D) seed matrix. Reports inter-factor angles + held-out probe quality.

### 3.2 Three-way comparison (every experiment)
- **factor** seeds (derived, held-out) — the aligned condition.
- **random** orthonormal seeds.
- **zeroed** seeds (pure Hopfield baseline).

### 3.3 Experiments (on test split, mean-centered features)
- **Phase 1** retrieval under corruption: factor-seeds vs random vs zeroed.
- **Phase 2** interference: categories = held-out factor labels (e.g. dominant shape);
  same- vs cross-category cosine. NOT circular because seeds are train-derived, evaluated on
  held-out test items.
- **Phase 3** capacity: capacity vs #patterns, three-way.

### 3.4 Honest verdict
PASS = (probe GREEN) AND (factor-seeds beat random-seeds on held-out Phase 1 retrieval
and/or Phase 2 separation, by a margin exceeding noise). Otherwise report the null with the
probe + variance numbers as the mechanistic explanation.

---

## 4. Out of scope (Milestone 2, separate spec)
- Agentness via motion (Heider-Simmel / video encoder) — requires a shared video-feature
  space or a two-memory redesign; deferred.
- Any change to `sdam/seeds.py`, `sdam/hopfield.py`, `sdam/model.py`.

---

## 5. Risks
- **R1 (primary):** even with DINOv2, factors may be decodable but low-variance → pipeline
  shows little. *Mitigation:* the probe's concentration factor catches this before the
  pipeline is built.
- **R2:** 3 factors may be correlated (count↔layout) → not independent systems.
  *Mitigation:* Metric 3 measures it; orthogonalization reported.
- **R3:** shape→Objectness proxy is contestable. *Mitigation:* flagged explicitly; framed as
  proxy, not clean instantiation.
- **R4:** CLEVR download/scene-graph acquisition. *Mitigation:* requirement is images +
  scene graphs; lightest source (val split / HF mirror / official zip) chosen at impl time.
