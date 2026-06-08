# Findings: Decodability≠Usability Dissociation + the Objectness Crossover

**Date:** 2026-06-07
**Status:** SECURED standalone result (Milestone 1 + objectness pipeline). Complete and
publishable on its own, independent of any later FactoredSDAM experiment.
**Sources:** `results_archive/probe/probe_results.json`,
`results_archive/objectness/objectness_pipeline.json` (committed run outputs).

---

## Summary

Pre-seeding a Modern Hopfield memory with Spelke core-knowledge directions (S-DAM) was tested on
real CLEVR features (DINOv2 ViT-S/14, mean-centered). Two standalone findings:

1. **A decodability≠usability dissociation:** a factor can be *perfectly linearly decodable* yet
   carry *negligible feature variance*, making it unusable by a variance-based memory mechanism.
   Linear decodability is **necessary but not sufficient** for a factor to serve as a memory prior.
2. **An objectness crossover:** on the one factor that *is* a dominant axis (shape, 89×), residual
   coding **helps at low corruption and hurts at high corruption** — because the architecture
   restores the seeded axis from the (corrupted) query, discarding redundancy exactly when
   denoising needs it.

Neither is a "Spelke-complete memory" claim. Scope is explicitly limited (see §5).

## 1. Feature space is real (gate-the-gate passed)

DINOv2 + mean-centering removed the common-mode that sank earlier CIFAR/ImageNet attempts:
raw pairwise cosine mean **0.807** → centered **−0.004**, centered std **0.283** (5.5× the
isotropic-at-384 baseline of 0.051). Every downstream metric is computed in a non-degenerate space.

## 2. The dissociation — TWO DISTINCT failure modes (do not conflate)

| factor (Spelke system) | decodability (bal-acc) | continuous | concentration | mode |
|---|---|---|---|---|
| shape (Objectness, proxy) | 0.851 | — | **89.2×** | decodable **and** dominant — usable |
| count (Numerosity) | 0.853 | R²=0.952 | **1.03×** | **decodable but variance-ORTHOGONAL** |
| layout (Geometry) | 0.641 | R²=0.746 | **0.17×** | **weakly decodable AND variance-SUPPRESSED** |

**§2a — count, the clean dissociation.** count is *highly* recoverable (bal-acc 0.853, R²=0.952)
yet sits at concentration 1.03× — indistinguishable from a random direction (≈1.0). The
information is fully present in *some* linear combination, but lives in a direction carrying
~average (negligible) variance. A variance-based memory mechanism structurally cannot exploit it.
*Hypothesised why:* DINOv2's self-supervised instance-discrimination objective organises feature
variance around *what an object is* (appearance/shape) over *how many* there are; count is encoded
but not amplified. Decodability needs the signal in *some* combination; concentration needs it to
*dominate*.

**§2b — layout, the messier mode (distinct from count).** layout fails *both* bars: decodability
0.641 is below the 0.70 threshold (margin 0.299 — real but weak signal), and concentration 0.17×
is **below random**, i.e. the layout direction is *anti-aligned* with the high-variance axes. This
is "weakly encoded and actively variance-suppressed," not the clean decodable-but-unusable story.
Reported separately; count is the dissociation exemplar.

**The dual-bar gate is the instrument.** Combined orthonormalised-span variance was 0.235 (clears
a 0.10 bar) — but that is carried almost entirely by shape (89×). Gating on the combined number
alone would have read as a healthy 3-factor pass; the per-factor concentration floor (>2.0)
correctly exposed it as a 1-factor result. The gate did its job, live, on real data.

## 3. The objectness crossover (Phase 1 retrieval, 3-way, ≥3 seeds)

On shape — the only factor clearing both bars — S-DAM (objectness-seed) vs random-seed vs zeroed
(plain Hopfield), retrieval accuracy (mean cosine) under corruption:

| corruption | objectness | best baseline | gain |
|---|---|---|---|
| 0.1 | 0.986 | 0.960 | **+0.026** |
| 0.3 | 0.976 | 0.955 | **+0.021** |
| 0.5 | 0.954 | 0.947 | +0.007 |
| 0.6 | 0.936 | 0.941 | −0.005 |
| 0.8 | 0.856 | 0.899 | −0.043 |
| 0.9 | 0.717 | 0.809 | **−0.092** |

Objectness-seeding **helps at low corruption, hurts at high corruption, crossing over ≈0.55.** The
low-corruption gains are real (e.g. +0.021 vs pooled std 0.0007 — far beyond noise) but **below the
pre-registered 0.05 effect-size floor**, and there is a clean reversal past 0.5.

**Verdict: NULL**, by the criteria locked *before* the run (gain ≥0.05 AND beyond pooled std at
corruption ≥0.3, no reversal). No threshold was moved post-hoc.

**Mechanism (architecture-specific, tied to the verified `read()` trace).**
`read(x) = retrieve(residual(x)) + project(x)`, with `project(x)` computed from the **query**.
Objectness-seeding strips the dominant shape axis from *storage*, so shape can only be restored
from the query at read. When the query is clean, decorrelated storage yields a small genuine win;
when the query is 80–90% corrupted, `project(corrupted_x)` is garbage and the shape information the
plain-Hopfield baselines kept safely in memory has been discarded. **Residual coding on a dominant
axis trades clean-query gains for a corrupted-query liability — it throws away redundancy exactly
when denoising needs it.**

## 4. Phase 2 two-level diagnostic (mechanistic instrumentation, not a verdict)

| mode | STORAGE same | STORAGE cross | FINAL same | FINAL cross |
|---|---|---|---|---|
| objectness | 0.115 | −0.006 | 0.120 | −0.012 |
| zeroed | 0.120 | −0.012 | 0.120 | −0.012 |

At the honest STORAGE level (`cosine(r_hat, residual(B))`), objectness-seeding *mildly worsens*
separation vs zeroed: same slightly lower (0.115 vs 0.120), cross slightly higher (−0.006 vs
−0.012). This is the pre-registered "cross up / load-bearing-axis" direction — stripping the
dominant separating axis makes stored residuals marginally more confusable — though the magnitude
is small. The seed mostly changes corruption-robustness (§3), not raw separation.

## 5. Contributions and scope

- **C1 (method + dissociation):** a dual-bar feasibility gate (decodability + variance
  concentration) and the finding that decodability is necessary but not sufficient for a memory
  prior — count is the existence proof.
- **C2 (crossover):** residual coding on a dominant axis helps clean queries and harms corrupted
  ones, due to query-side axis restoration — an architecture-specific characterization of *when*
  core-knowledge seeding helps.
- **Explicitly NOT:** validation of the four-system Spelke architecture. Of four systems, one is
  absent (Agentness — no agents in CLEVR), one variance-orthogonal (count), one variance-suppressed
  (layout), one exploitable-but-with-the-§3-tradeoff (objectness). Workshop-grade, honest.

## 6. FactoredSDAM (Milestone 3) — PARTIAL FIX, robust across w

Diagnosis from §3 said the high-corruption harm comes from `read()` restoring shape via
`project(corrupted_query)`. FactoredSDAM tests the fix: store `[residual, w·c]` so the shape
coefficient `c` is recalled **from memory**, not the query. Verdict read at the variance-matched
`w* = 0.093`; `w ∈ {0.5×, 1×, 2×}` as a pre-registered robustness check.

**Result: PARTIAL_FIX at all three `w` (robust, not INCONCLUSIVE — the result is the mechanism,
not a scaling artifact).** At `w=1×`, retrieval accuracy vs plain Hopfield (zeroed):

| corruption | factored | zeroed | objectness | factored − zeroed |
|---|---|---|---|---|
| 0.1–0.5 | 0.983→0.971 | 0.960→0.947 | 0.986→0.953 | +0.023…+0.025 |
| 0.7 | 0.949 | 0.928 | 0.909 | +0.021 |
| 0.8 | 0.907 | 0.899 | 0.856 | +0.009 |
| 0.9 | 0.760 | 0.809 | 0.717 | **−0.050** |

- **Fix largely works:** factored is the best method from 0.1–0.8 and pushes the crossover from
  objectness's **0.558 → 0.815** — storing the coefficient in memory recovers most of the
  robustness residual coding discarded, confirming the §3 diagnosis end-to-end.
- **Residual failure (pre-registered outcome #2):** still reverses at 0.9 (−0.050). The retrieval
  *key* is still built from the corrupted query, so at extreme corruption it mis-addresses —
  memory restoration fixed *reconstruction*, not *addressing*.
- **Not WORKS:** the 0.9 reversal, and the gain over plain Hopfield tops at +0.026 (< the locked
  0.05 floor). A genuine improvement over plain objectness-seeding, with a characterized remaining
  failure mode.
- **Honest correction:** spec §2.1.3 guessed the crossover would move right as `w` *shrinks*; it
  moves right as `w` *grows* (0.807→0.830 across 0.5×→2×). Directional guess wrong; the
  robustness-across-`w` conclusion holds. Source: `results_archive/factored/factored_pipeline.json`.

## 7. The full arc (paper structure)

1. **Dissociation** — decodability ≠ usability; a prior must be a dominant variance axis (count is
   R²=0.95 yet 1.03×). The dual-bar gate is the instrument.
2. **Objectness crossover** — seeding a dominant axis helps clean queries, hurts corrupted ones
   (crossover 0.56), because shape is restored from the corrupted query.
3. **FactoredSDAM partial fix** — storing the axis coefficient in memory recovers most of the
   robustness (crossover 0.56→0.82, best method to 0.8), residual failure traced to corrupted
   *addressing* (not reconstruction), robust across `w`.

A coherent, mechanistic, honest characterization of *when and why* core-knowledge seeding helps a
residual-coding associative memory — single-factor (objectness); explicitly not a four-system claim.

> Secured findings. All three results are banked; any further variant (e.g. denoising the
> retrieval key) is upside on top of this complete arc, and does not supersede it.
