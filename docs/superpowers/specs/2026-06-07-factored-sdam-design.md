# FactoredSDAM — Design Spec (Milestone 3, factored storage variant)

**Date:** 2026-06-07
**Status:** Draft for review
**Depends on:** `docs/findings/2026-06-07-objectness-null-and-dissociation.md` (the crossover)
**Constraint (non-negotiable):** new module only. `sdam/model.py`, `sdam/seeds.py`,
`sdam/hopfield.py` are **not** edited. FactoredSDAM lives in a new file.

---

## 0. Where this comes from

The objectness pipeline returned NULL with a **crossover**: objectness-seeding helped at low
corruption (+0.021 @ 0.3) and hurt at high corruption (−0.092 @ 0.9), crossing ≈0.55. Diagnosed
cause (verified `read()` trace): residual coding strips the dominant shape axis from storage, and
`read()` restores it via `project(query)` — so at high corruption shape is reconstructed from a
garbage query, discarding redundancy the plain-Hopfield baselines kept in memory.

FactoredSDAM tests the fix the data points at: keep the residual decorrelation **and** keep the
shape information in memory, by storing the shape coefficient as a separate channel.

## 1. The mechanism actually under test (REFRAME — read this carefully)

The naive framing is "restore shape from memory instead of the query." But note the math: if we
store `[residual, c]` and reconstruct `x̂ = r_hat + c_hat·seed` with `c_hat` recalled cleanly, then
`x̂ ≈ the full original pattern` — which is *what plain Hopfield on raw `x` already stores*. So
FactoredSDAM is mathematically close to the zeroed baseline **unless** storing `[residual, c]` has
**better recall properties than storing raw `x`** — better attractor separation, higher capacity,
cleaner retrieval under corruption.

**Therefore the real question this experiment answers is NOT "memory vs query." It is:**

> Does factoring the dominant axis into a separate scalar channel — storing `[residual, c]` —
> improve the Hopfield's recall (capacity / basin separation / robustness) over storing the raw
> pattern `x` directly?

A WORKS result means *factoring improves Hopfield recall over raw storage*. If factoring buys no
recall improvement, FactoredSDAM collapses to the baseline. Pre-register it this way.

## 2. Architecture (new module `sdam/factored.py`, class `FactoredSDAM`)

Single dominant seed `s` (unit, the held-out objectness/shape direction). For each centered
pattern `x`:
- coefficient `c = x · s`  (scalar)
- residual `r = x − c·s`  (D-dim, shape removed)
- **store the augmented vector `[r, w·c]` (D+1 dims)** in a Modern Hopfield memory, where `w` is a
  **channel-scale** (see §2.1).

Read, given (possibly corrupted) query `x'`:
- form augmented key `[residual(x'), w·(x'·s)]`
- Hopfield-retrieve `[r_hat, w·c_hat]`
- reconstruct `x̂ = r_hat + c_hat · s`  — **`c_hat` comes from memory (the retrieved channel),
  NOT from the query.**

`sdam.hopfield.HopfieldLayer` is reused unmodified on (D+1)-dim vectors. `SDAM` is untouched;
FactoredSDAM is a parallel implementation that imports `HopfieldLayer`.

### 2.1 The channel-scale `w` (implementation risk to control)
shape is the 89× dominant axis, so `c = x·s` can be large; the c-channel could dominate (or, if
tiny, vanish from) the Hopfield dot-product similarity. `w` balances the c-channel against the
residual norm. Default `w` chosen so the c-channel variance ≈ mean residual-dimension variance
(documented in code). `w` is a config value, not a magic number; we report sensitivity if results
hinge on it.

## 3. Pre-registered predictions — FOUR outcomes (locked before the run)

Phase 1, 4-way: **factored vs objectness vs random vs zeroed**, retrieval accuracy vs corruption
0.1–0.9, ≥3 seeds, mean±std. We explicitly report the **crossover point** (corruption where the
seeded curve drops below the zeroed baseline) for factored and objectness.

1. **WORKS** — factored ≥ both random and zeroed at *every* corruption ≥0.3, low-corruption gain
   preserved (≥0.02 at ≤0.3) **and** high-corruption harm eliminated (no reversal; gain ≥ 0 at 0.9).
   Interpretation: factoring the dominant axis improves Hopfield recall over raw storage. Genuine
   positive.
2. **PARTIAL FIX** *(the fourth outcome, per review)* — factored's **crossover point moves right**
   vs objectness (e.g. ~0.55 → ~0.7+) but does **not** vanish. Interpretation: storing `c` in
   memory fixes "shape restored from corrupted query," but FactoredSDAM **inherits the corrupted
   retrieval KEY** — the augmented key `[residual(x'), ...]` is still built from the corrupted
   input, so at extreme corruption Hopfield lands in the wrong basin and recalls the wrong `c_hat`.
   A residual high-corruption dip is therefore *anticipated*, not a surprise. Publishable: "memory
   restoration addresses query-side reconstruction but not query-side addressing."
3. **COLLAPSE-TO-BASELINE** — factored ≈ zeroed everywhere → factoring `[residual, c]` gave no
   recall advantage over raw `x`; residual decorrelation bought nothing. Honest null; the §0
   crossover finding stands.
4. **STILL-HURTS** — factored reverses at ≈the same point as objectness → the harm is driven by the
   decorrelation itself, not by query-side restoration. Strong negative; also publishable.

No threshold moves post-hoc. The crossover-point metric is reported numerically regardless.

## 4. Files
- **Create** `sdam/factored.py` — `FactoredSDAM` (imports `HopfieldLayer`; core untouched).
- **Create** `tests/test_factored.py` — pure/torch-light unit tests: coefficient+residual round-trip
  (`r + c·s == x`), augmented-vector shape, reconstruction-from-clean-memory ≈ identity,
  channel-scale `w` wiring. (SDAM-style retrieval run on Colab.)
- **Modify** `experiments/objectness_pipeline.py` OR add `experiments/factored_pipeline.py` —
  4-way Phase 1 + explicit crossover-point computation. (Prefer a new file to keep the banked
  Option-1 pipeline intact.)
- **Add** a Colab cell / `notebooks/factored_runner.ipynb`.

## 5. Scope
- Single dominant factor (objectness/shape). Not a four-system claim.
- `sdam/` core modules untouched (new module + new experiment file only).
- All four outcomes above are publishable; the experiment is honest regardless of which fires.

## 6. Risks
- **R1 (most likely per review):** collapse-to-baseline — factoring may not improve recall over raw
  storage. Mitigated: pre-registered as outcome 3; the crossover finding is already banked.
- **R2:** channel-scale `w` confound (§2.1) — c-channel dominating/vanishing. Mitigated: principled
  default + sensitivity report.
- **R3:** corrupted retrieval key (§3 outcome 2) — the fix may only move the crossover right.
  Mitigated: pre-registered as outcome 2, crossover point reported numerically.
