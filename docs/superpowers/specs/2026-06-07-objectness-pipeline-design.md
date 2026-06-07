# Objectness-Only S-DAM Pipeline + Dissociation Writeup — Design Spec

**Date:** 2026-06-07
**Status:** Approved design (Milestone 2, post-probe)
**Depends on:** `2026-06-07-clevr-factor-probe-design.md` (probe returned YELLOW)

---

## 0. What the probe established (real numbers, origin/main:results_archive/probe/probe_results.json)

| metric | value | bar | result |
|---|---|---|---|
| centered cosine std | 0.283 (raw mean 0.807) | >0.05 | space healthy (common-mode removed) |
| shape decodability | 0.851 (margin 0.534) | ≥0.70 | ✅ |
| count decodability | 0.853 (margin 0.523, R²=0.952) | ≥0.70 | ✅ |
| layout decodability | 0.641 (margin 0.299, R²=0.746) | ≥0.70 | ❌ |
| concentration shape / count / layout | 89.21 / 1.03 / 0.17 | >2.0 | only shape |
| combined span | 0.235 | >0.10 | ✅ (carried by shape) |

**Verdict: YELLOW.** Only **objectness (shape)** is both decodable AND a dominant variance axis.
**count** is the key finding: linearly recoverable (R²=0.95) yet a random-level direction (1.03×).

## 1. Two contributions (and their honest scope)

**C1 — Dissociation (headline):** *decodable ≠ usable as a memory prior.* A factor must be a
**dominant variance axis**, not merely linearly decodable, to be exploitable by residual-coding
associative memory. count is the existence proof (R²=0.95, concentration 1.03×). The dual-bar
gate (`combined>0.10 AND per-factor concentration>2.0`) is the instrument that exposes it.

**C2 — Objectness existence proof:** on the one factor that clears both bars (shape, 89×), a
3-way test of whether the S-DAM residual mechanism delivers. **SCOPE (state explicitly in
paper):** this is a *single-factor existence proof*, NOT validation of the four-system Spelke
architecture. We claim "the mechanism works when the prior aligns with a dominant axis," nothing
more. Numerosity/geometry are reported as characterized boundary cases, not as system claims.

## 2. PRE-REGISTERED PREDICTIONS (locked BEFORE running — the anti-Rorschach)

shape is *both* the seed direction and the Phase-2 category label, so projecting it out is
subtle and could go either way. We commit to these interpretations **now**:

**Mechanism design recap:** S-DAM stores `residual = x − proj_shape(x)` and reconstructs
`read(A) = retrieved_residual + proj_shape(A)`. The shape axis (89×) is the dominant common
direction that makes patterns interfere; removing it from *storage* should decorrelate stored
residuals; adding `proj_shape(A)` back at *read* should restore the category signal.

**P1 — Phase 1 (retrieval under corruption) is the PRIMARY test.**
- PREDICT: objectness-seed accuracy > random-seed ≳ zeroed, **widening as corruption/load rises**,
  because decorrelated residuals reduce cross-talk on the dominant axis.
- MECHANISM WORKS iff: objectness-seed strictly beats both baselines at moderate-to-high
  corruption (≥0.3) by a margin exceeding run-to-run noise (≥3 seeds, report std).
- MECHANISM FAILS (honest, publishable null) iff: objectness-seed ≈ random ≈ zeroed. Reported as
  "even a dominant, decodable prior does not improve S-DAM retrieval."

**P2 — Phase 2 (interference, category = dominant shape) is the CONFOUND CHECK, not pass/fail.**
We measure same-/cross-category cosine **before** residual encoding (raw centered features =
the zeroed baseline) and **after** (objectness-seed), and pre-commit the reading:
- EXPECTED (mechanism intact): same-category separation is **preserved** after residual
  encoding — because `read()` adds `proj_shape(A)` back, restoring the category signal. The seed's
  value shows up as cleaner retrieval/lower cross-category interference, not as a change in raw
  separation.
- CONFOUND (NOT a success): if same-category similarity **collapses** under the objectness-seed
  relative to zeroed, that means residual coding removed the category axis and read() failed to
  restore it (retrieval failure). We will report this as a confound of using the seed axis as the
  category, **not** as evidence the mechanism works or fails on interference.
- This before/after table is reported regardless of outcome, so the seed's effect is *understood*,
  not read as pass/fail.

## 3. Why count/layout fail — stated hypothesis (boundary section)

Measurement is "low concentration"; the paper must offer a *why*. Pre-stated hypothesis:

> DINOv2's self-supervised objective (view-invariant self-distillation) organizes feature
> **variance** around appearance / object identity (shape, texture, color). Numerosity and spatial
> layout are still *encoded* — linearly recoverable (count R²=0.95) — but as **low-variance,
> distributed** codes spread across many dimensions, because counting and spatial extent are
> neither what dominates appearance variance nor what the SSL objective rewards. Linear
> decodability only requires the signal exist in *some* linear combination; concentration requires
> it *dominate* the variance. SSL appearance features satisfy the former for count, not the latter.

**Falsifiable corollary (report if cheap):** a *k*-dimensional count subspace (top-k ridge /
PLS directions) should capture more variance than the single count direction but still fall well
short of shape — confirming "distributed, not concentrated." A count-*supervised* encoder would be
expected to concentrate it (left as future work, not run here).

## 4. Pipeline design

### 4.1 Features (with caching — the A100-time fix)
- Reuse `probe/encoder.py` DINOv2 ViT-S/14, 384-d, mean-centered (train mean only), L2 as needed.
- **Cache** the (N, 384) feature matrix + factor arrays to disk, keyed on a hash of
  `(model_name, sorted image filenames, preprocessing signature)`. Skip re-extraction on cache hit.
  (The probe did not cache — this is the fix.)

### 4.2 Seed (held-out)
- objectness seed = the shape direction from `probe/variance_gate.factor_seed_direction`
  (first PC of class-mean spread), fit on **train** only.
- Three-way: **objectness-seed** vs **random orthonormal** vs **zeroed** (pure Hopfield).
- Injected via the existing `m.ssl.seeds = nn.Parameter(...)` path; `sdam/` core untouched.

### 4.3 Experiments (held-out test split, mean-centered features)
- **Phase 1:** retrieval accuracy vs corruption rate (0.1–0.9), ≥3 seeds, mean±std, 3-way.
- **Phase 2:** same-/cross-shape interference, with the **before/after** table from P2.

### 4.4 Files
- `experiments/objectness_pipeline.py` (orchestration; reuses probe + sdam modules)
- new caching helper in `probe/encoder.py` (or `data/feature_cache.py`)
- tests for: cache key determinism, seed-injection wiring, Phase-1/2 metric functions
  (pure, known-answer where possible)
- Colab cell appended to `notebooks/probe_runner.ipynb` (or a new `objectness_runner.ipynb`)

## 5. Honest verdict logic
- **Mechanism POSITIVE:** P1 satisfied (objectness-seed beats both baselines at ≥0.3 corruption
  beyond noise) AND P2 shows no confound collapse.
- **Mechanism NULL (publishable):** P1 shows no advantage. Paper: "a dominant, decodable prior
  still does not improve S-DAM" — strengthens C1.
- **Confounded:** P2 collapse — reported as a measurement confound, P1 still adjudicates.
- No threshold-hacking; predictions above are locked.

## 6. Out of scope
- Numerosity/geometry pipelines (boundary cases, characterized via the probe only).
- The four-system architecture claim. Agentness. Motion/Heider-Simmel. Encoder pivots.
- Any change to `sdam/seeds.py`, `sdam/hopfield.py`, `sdam/model.py`.

## 7. Risks
- **R1:** Phase-2 confound (seed == category axis) — mitigated by the pre-registered before/after
  reading in §2-P2.
- **R2:** objectness pipeline may be null even at 89× — mitigated: null is publishable (C1).
- **R3:** single-factor result over-claimed — mitigated by explicit scope statement in §1/C2.
