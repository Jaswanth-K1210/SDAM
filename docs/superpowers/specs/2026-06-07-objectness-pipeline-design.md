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
3-way test of whether the S-DAM residual mechanism delivers.

**Headline claim (exact wording for the paper — do not inflate):** the contribution is
(i) *a method* for testing whether core-knowledge seeding helps a residual-coding memory
(the dual-bar feasibility gate), (ii) *a single-factor proof-of-mechanism* on objectness, and
(iii) *a dissociation* showing linear decodability is **necessary but not sufficient** for a
factor to be usable as a memory prior. It does **NOT** validate the four-system Spelke
architecture: of the four systems, one is absent (Agentness — no agents in CLEVR), one is
variance-orthogonal (count), one is variance-suppressed (layout), and only one (objectness) is
exploitable. This is an honest, workshop-grade claim — explicitly **not** "we built
Spelke-complete memory."

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
  corruption (≥0.3) by **≥0.05 mean cosine accuracy AND beyond run-to-run noise** (≥3 seeds,
  the gain must exceed the pooled std). Both the effect-size floor and the noise check are
  required — locked now so a 0.01 "win" can't be called a win.
- MECHANISM FAILS (honest, publishable null) iff: objectness-seed ≈ random ≈ zeroed. Reported as
  "even a dominant, decodable prior does not improve S-DAM retrieval."

**P2 — Phase 2 (interference, category = dominant shape): COMMITTED DIRECTIONAL prediction +
confound check (NOT pass/fail).** We measure same-/cross-category cosine **before** residual
encoding (raw centered features = zeroed baseline) and **after** (objectness-seed).

*Committed prediction of the EFFECT DIRECTION (decided now, before the run):* objectness-seeding
is predicted to **lower CROSS-category interference** (cleaner retrieval makes `read(A)` closer to
true A, and a different-shape B then overlaps less) while **PRESERVING same-category similarity**
(`read(A)=residual+proj_shape(A)` adds the category axis back at read). Net effect: a **wider
same−cross gap driven by the cross side dropping, NOT by same-category rising.** We explicitly do
**NOT** predict that objectness-seeding *increases* same-category similarity.
- EXPECTED (mechanism intact): cross-category cosine DOWN vs zeroed; same-category ≈ preserved.
- CONFOUND (NOT a success): if same-category similarity **collapses** vs zeroed, residual coding
  stripped the category axis and read() failed to restore it (retrieval failure). Reported as a
  confound of using the seed axis as the category — **not** evidence for or against the mechanism;
  P1 then adjudicates.
- The before/after table is reported regardless, so the seed's effect is *understood*, not graded.

## 3. Why count and layout fail — TWO DISTINCT modes (do not conflate)

count and layout fail differently; the paper characterizes them separately or a reviewer catches
the conflation.

### 3a. count — the clean dissociation: decodable but VARIANCE-ORTHOGONAL
count is highly decodable (bal_acc 0.853, R²=0.952) yet concentration = **1.03×**, i.e.
statistically indistinguishable from a random direction (random ≈ 1.0). The count information is
fully present and linearly recoverable, but it lives in a direction carrying ~average (negligible)
feature variance — a variance-based memory mechanism structurally cannot exploit it. **This is the
headline dissociation: decodability is necessary but not sufficient.**

*Stated why:* DINOv2's SSL objective (instance discrimination / view-invariant self-distillation)
rewards representing *what an object is* (shape/appearance → high variance) over *how many* there
are (count is task-irrelevant to the SSL objective, so encoded but not amplified). Decodability
needs the signal in *some* linear combination; concentration needs it to *dominate* the variance.

*Falsifiable corollary (cheap, will run):* a *k*-dim count subspace (top-k ridge/PLS) should
capture more variance than the single direction but still ≪ shape — confirming "present but
distributed, not concentrated." A count-*supervised* encoder would be expected to concentrate it
(future work, not run here).

### 3b. layout — the messier case: WEAKLY decodable AND variance-SUPPRESSED
layout fails **both** bars: decodability 0.641 is *below* the 0.70 bar (margin 0.299 is real, so
there is weak signal), and concentration **0.17× is BELOW random** — the layout direction is
*anti-aligned* with the high-variance axes, not merely orthogonal to them. So layout is a weaker,
less interpretable case than count: not a clean "decodable-but-unusable" story but "weakly encoded
and actively variance-suppressed." Characterize it as such — explicitly distinct from count — and
do not use it as the dissociation example (count is the clean one).

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
