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

**Mechanism trace (from sdam/model.py:98-105, verified — do not assume):**
`read(x) = mem.retrieve(residual(x)) + project(x)`. Retrieval is keyed on `residual(x)` (shape
removed); the shape component re-added is `project(x)` — **computed from the QUERY x, not from
memory.** Consequences locked into the design:
- Phase 1: x is *corrupted* → both the retrieval key and the re-added `project(x)` come from the
  corrupted input (degraded restoration).
- Phase 2: x is *clean* A → `project(A)` is A's true shape, **added back by hand** → a low final
  `cosine(read(A), B)` is partly an artifact of hand-adding A's clean shape, NOT proof of clean
  recall. This is why Phase 2's final cross-cosine cannot be the verdict (see P2).

**P1 — Phase 1 (retrieval under corruption) is the SOLE ADJUDICATOR of WORKS/NULL.**
- PREDICT: objectness-seed accuracy > random-seed ≳ zeroed, **widening as corruption/load rises**,
  because decorrelated residuals reduce cross-talk on the dominant axis.
- MECHANISM WORKS iff: objectness-seed strictly beats both baselines at moderate-to-high
  corruption (≥0.3) by **≥0.05 mean cosine accuracy AND beyond run-to-run noise** (≥3 seeds,
  the gain must exceed the pooled std). Both the effect-size floor and the noise check are
  required — locked now so a 0.01 "win" can't be called a win.
- MECHANISM FAILS (honest, publishable null) iff: objectness-seed ≈ random ≈ zeroed. Reported as
  "even a dominant, decodable prior does not improve S-DAM retrieval."

**P2 — Phase 2 is a MECHANISTIC DIAGNOSTIC, NOT an adjudicator.** It explains *how* the seed moved
the geometry behind whatever P1 shows; it does not by itself declare WORKS or NULL. P1 decides.

*Why not a one-tailed "cross drops" prediction:* shape is the **89× dominant separating axis**, so
stripping it from storage almost certainly makes the stored residuals of two different-shape items
**more** similar (the axis that separated them is gone). Two credible, opposite mechanisms exist,
so we pre-register **two-tailed with both tails interpreted in advance** rather than guess.

*Measure cross- and same-category interference at TWO LEVELS, separately* (this split is what makes
the two-tailed result interpretable):
- **STORAGE level (the honest one):** `cosine(r_hat, residual(B))` — does memory recall confuse the
  stored *residuals* of different-shape items? Not contaminated by re-added shape.
- **FINAL level:** `cosine(read(A), B)` — full reconstruction. NOTE: in Phase 2 the query is clean,
  so this re-adds `project(A)` by hand; a low value here is partly that artifact, not clean recall.

*Pre-committed interpretation of the CROSS direction, read at the STORAGE level:*
- **cross DOWN** → cleaner residual recall / restoration dominates → seed aids separation.
- **cross UP** (judged the more likely a priori) → stripping the load-bearing separating axis made
  residuals more confusable in memory → finding: **"seeding a dominant axis can HARM cross-category
  retrieval because that axis was load-bearing for separation"** — a real, publishable result about
  *when* core-knowledge seeding backfires.
- **cross FLAT** → effects cancel → leans NULL (C1).

*Same-category confound lock (unchanged):* if same-category similarity **collapses** vs zeroed at
the FINAL level, residual coding stripped the category axis and read() failed to restore it
(retrieval failure) — reported as a confound of seed==category-axis, not a mechanism verdict.

The two-level before/after table is reported regardless of outcome.

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
- **Phase 1 (adjudicator):** retrieval accuracy vs corruption rate (0.1–0.9), ≥3 seeds, mean±std,
  3-way (objectness/random/zeroed).
- **Phase 2 (diagnostic only):** same-/cross-shape interference at **two levels** — STORAGE
  (`cosine(r_hat, residual(B))`) and FINAL (`cosine(read(A), B)`) — reported per §2-P2, two-tailed.

### 4.4 Files
- `experiments/objectness_pipeline.py` (orchestration; reuses probe + sdam modules)
- new caching helper in `probe/encoder.py` (or `data/feature_cache.py`)
- tests for: cache key determinism, seed-injection wiring, Phase-1/2 metric functions
  (pure, known-answer where possible)
- Colab cell appended to `notebooks/probe_runner.ipynb` (or a new `objectness_runner.ipynb`)

## 5. Honest verdict logic — P1 alone adjudicates; P2 explains
- **Mechanism POSITIVE:** P1 satisfied — objectness-seed beats *both* baselines at corruption ≥0.3
  by ≥0.05 mean-cosine AND beyond pooled std (≥3 seeds). Phase 2 (two-level, two-tailed) is then
  reported to *explain how* (cross down = cleaner recall; cross up = load-bearing-axis harm masked
  by P1 still winning, etc.).
- **Mechanism NULL (publishable):** P1 shows no advantage → "a dominant, decodable prior still does
  not improve S-DAM" (strengthens C1). Phase 2 explains the geometry behind the null.
- **Phase 2 is never the verdict.** A same-category FINAL-level collapse is flagged as a
  seed==category-axis confound; a STORAGE-level cross rise is reported as the "seeding a load-bearing
  axis backfires" finding — but P1 alone says WORKS/NULL.
- No threshold-hacking; all predictions and bars above are locked before the run.

## 6. Out of scope
- Numerosity/geometry pipelines (boundary cases, characterized via the probe only).
- The four-system architecture claim. Agentness. Motion/Heider-Simmel. Encoder pivots.
- Any change to `sdam/seeds.py`, `sdam/hopfield.py`, `sdam/model.py`.

## 7. Risks
- **R1:** Phase-2 confound (seed == category axis) — mitigated by the pre-registered before/after
  reading in §2-P2.
- **R2:** objectness pipeline may be null even at 89× — mitigated: null is publishable (C1).
- **R3:** single-factor result over-claimed — mitigated by explicit scope statement in §1/C2.
