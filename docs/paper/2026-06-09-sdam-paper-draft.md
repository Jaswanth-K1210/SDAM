# Decodable Is Not Usable: A Variance-Concentration Criterion for Core-Knowledge Priors in Associative Memory

**Draft — 2026-06-09.** Workshop-length. All numbers from pre-registered runs in
`results_archive/` (probe, objectness, factored). Scope is single-factor and stated honestly
throughout; this is **not** a claim of a four-system "Spelke-complete" memory.

---

## Abstract

Modern Hopfield / dense associative memories (DAMs) can be pre-seeded with structured priors, and
a natural hypothesis — inspired by Spelke core-knowledge systems — is that seeding the memory with
cognitively meaningful directions (objectness, numerosity, geometry) and storing new content as
residuals relative to them improves retrieval. We test this on real CLEVR features from a frozen
self-supervised encoder (DINOv2 ViT-S/14), with all predictions and thresholds **pre-registered
before running**. We report three findings. (1) **A decodability≠usability dissociation:** a factor
can be near-perfectly linearly decodable yet carry negligible feature variance, making it unusable
by a variance-based memory — object count is recoverable at R²=0.95 but its direction concentrates
only 1.03× the variance of a random direction. Linear decodability is **necessary but not
sufficient** for a memory prior; we introduce a dual-bar feasibility gate (decodability **and**
variance concentration) that detects this. (2) **An objectness crossover:** for the one factor that
is a dominant axis (shape, 89×), residual coding *helps* retrieval at low corruption and *hurts* it
at high corruption (crossover at 0.56), because the architecture restores the seeded axis from the
corrupted query. (3) **A factored fix:** storing the seeded axis as a separate scalar channel,
recalled from memory, pushes the crossover from 0.56 to 0.82 and is the best method up to 80%
corruption, with the residual high-corruption failure traced to corrupted *addressing* rather than
reconstruction. Together these characterize *when and why* a core-knowledge prior helps a
residual-coding associative memory.

## 1. Introduction

Spelke core-knowledge theory posits a small set of innate systems — objectness, agentness,
numerosity, geometry — that scaffold learning. A tempting machine-learning analogue is to seed an
associative memory with these directions and store new experiences as residuals, hoping the prior
improves capacity and noise robustness. We ask, plainly: **does seeding a dense associative memory
with a core-knowledge prior actually help, and if so, when?**

Answering this honestly turns out to require resisting two traps. First, a prior direction only
matters to a variance-based memory if it carries real variance — a direction can encode a factor
(be linearly decodable) while contributing nothing to the geometry the memory operates on. Second,
*how* the architecture uses the seed at read time determines whether the prior is a help or a
liability under noise. We make both precise, pre-register every criterion, and report what happens.

**Contributions.**
- A **dual-bar feasibility gate** for memory priors: a factor must be both linearly decodable
  *and* a dominant variance axis. We show the second bar is essential.
- The **decodability≠usability dissociation**, demonstrated on CLEVR/DINOv2: numerosity is
  decodable (R²=0.95) yet variance-orthogonal (concentration 1.03×).
- A characterization of **when residual-coding seeding helps**: the objectness crossover and its
  mechanism, and a **factored-storage variant** that recovers most of the lost robustness.
- A methodology: **pre-registered predictions and thresholds**, with a banked negative/partial
  result reported faithfully rather than tuned into a positive.

## 2. Method

### 2.1 S-DAM: seeds and residual coding
A Spelke-Seeded DAM holds an orthonormal seed set `S` and a Modern Hopfield memory. An input `x` is
written as its residual `r = x − proj_S(x)`; read reconstructs `x̂ = retrieve(residual(x)) +
proj_S(x)`. Crucially, `proj_S(x)` is computed from the **query** `x`, not from memory — a detail
that drives Finding 3.

### 2.2 The dual-bar feasibility gate
Before testing any memory effect, we ask whether a candidate factor is even usable. For a factor
with held-out direction `u` (unit) in centered feature covariance `Σ`:
- **Decodability:** held-out balanced 3-class accuracy of a linear probe (chance ≈ 0.33), with a
  permutation control (false-positive guard) and CV-tuned regularization (false-negative guard).
- **Concentration:** `D · (uᵀΣu / tr Σ)` — variance along `u` relative to a random direction
  (≈1). A factor passes only if it clears **both** a decodability bar (0.70) and a concentration
  floor (>2× random), and the combined orthonormalized span exceeds 0.10 of centered variance.
The two are reported separately because they can dissociate (Finding 1).

### 2.3 FactoredSDAM
To keep residual decorrelation *and* retain the seeded factor in memory, FactoredSDAM stores the
augmented vector `[r, w·c]` where `c = x·s` is the scalar coefficient on the dominant seed `s`, and
reconstructs `x̂ = r_hat + c_hat·s` with `c_hat` recalled **from memory**. The channel scale `w` is
fixed by a pre-registered variance-matching rule (c-channel variance = mean residual-dimension
variance), never tuned; robustness is checked at `w ∈ {0.5×, 1×, 2×}`.

## 3. Experimental setup

- **Data/encoder:** CLEVR v1.0 scene graphs (6,000 scenes); frozen **DINOv2 ViT-S/14** CLS features
  (384-d), **mean-centered** (train mean). Self-supervised features are essential: supervised
  ImageNet features collapse into a cosine≈0.8 cone that mean-centering cannot fix.
- **Factors (3, single-domain):** objectness→dominant shape, numerosity→object count,
  geometry→spatial layout. **Agentness is out of scope** (CLEVR has no agents).
- **Baselines:** three-way throughout — seeded vs **random orthonormal** seed vs **zeroed** seed
  (plain Hopfield). The random/zeroed controls isolate the prior's contribution.
- **Pre-registration:** all bars, predictions, and the five FactoredSDAM outcomes were committed to
  the repository *before* the corresponding runs (see `docs/superpowers/specs/`).

## 4. Results

### 4.1 The feature space is non-degenerate (gate-the-gate)
Raw pairwise cosine mean **0.807** (DINOv2 on synthetic renders still has a common-mode) collapses
to **−0.004** after mean-centering, with centered std **0.283** — 5.5× the isotropic-at-384
baseline (0.051). All subsequent metrics are computed in a trustworthy, spread-out space.

### 4.2 Decodability ≠ usability

| factor | decodability | continuous | concentration | status |
|---|---|---|---|---|
| shape (objectness) | 0.851 | — | **89.2×** | decodable **and** dominant |
| count (numerosity) | 0.853 | R²=0.952 | **1.03×** | decodable, **variance-orthogonal** |
| layout (geometry) | 0.641 | R²=0.746 | **0.17×** | weakly decodable, **variance-suppressed** |

Count is the clean dissociation: fully recoverable yet, as a direction, indistinguishable from
random in variance. A variance-based memory cannot exploit it. Layout fails differently — weakly
decodable and *anti-aligned* with the high-variance axes. The combined seed-span variance (0.235)
would pass a single-number gate, but it is carried almost entirely by shape; the per-factor
concentration floor correctly exposes a one-factor result. *Hypothesised cause:* self-supervised
instance discrimination organizes variance around object identity (shape), not "how many" or
"where"; those are encoded but not amplified.

### 4.3 The objectness crossover
On shape — the only usable factor — S-DAM retrieval (mean cosine) vs corruption, 3 seeds:
objectness-seeding *helps* at low corruption (+0.021 at 0.3) and *hurts* at high corruption
(−0.092 at 0.9), crossing over at **0.56**. The low-corruption gains are real (far beyond noise)
but below a pre-registered 0.05 effect-size floor, and the reversal is clean → pre-registered
verdict **NULL**, no threshold moved. **Mechanism:** residual coding strips the dominant shape axis
from storage, and `read()` restores it from `project(corrupted_query)`; at high corruption that
projection is garbage, so the prior discards redundancy the plain-Hopfield baseline kept in memory.

### 4.4 FactoredSDAM: a partial fix, robust across `w`
Storing the shape coefficient in memory (rather than restoring it from the query) yields, at the
variance-matched `w`:

| corruption | factored | plain Hopfield | gain |
|---|---|---|---|
| 0.1–0.5 | 0.983→0.971 | 0.960→0.947 | +0.023…+0.025 |
| 0.7 | 0.949 | 0.928 | +0.021 |
| 0.8 | 0.907 | 0.899 | +0.009 |
| 0.9 | 0.760 | 0.809 | −0.050 |

Factored is the **best method from 0% to 80% corruption** and pushes the crossover from 0.56 to
**0.82** — storing the coefficient in memory recovers most of the robustness residual coding
discarded, confirming the §4.3 diagnosis end-to-end. The verdict is **PARTIAL_FIX**, identical at
`w ∈ {0.5×, 1×, 2×}` (robust, not a scaling artifact). It is not a clean WORKS for two reasons: the
residual reversal at 0.9, and a gain that tops at +0.026 (below the 0.05 floor). The residual
failure is traced to the retrieval **key**, which is still built from the corrupted query — memory
restoration fixed *reconstruction*, not *addressing*.

## 5. Discussion: when does a core-knowledge prior help?
A prior helps a residual-coding associative memory only when (i) its factor is a **dominant variance
axis** (not merely decodable), and (ii) the architecture restores the factor at read **from memory**
rather than from a possibly-corrupted query. Violate (i) and the prior is inert (numerosity);
satisfy (i) but violate (ii) and the prior is a noise liability (objectness crossover); satisfy both
and the prior helps across a wide corruption range, with residual failure only where the *retrieval
key itself* is destroyed (FactoredSDAM).

## 6. Limitations and scope
- **Single factor.** Only objectness was usable; this is a single-factor existence proof, not a
  four-system validation. Numerosity and geometry are reported as characterized boundary cases.
- **Agentness absent.** CLEVR has no agents; the agentness system is untested here.
- **Residual high-corruption failure.** FactoredSDAM still reverses at 90% corruption (corrupted
  addressing); denoising the retrieval key is future work (Milestone 4).
- **Encoder-specific.** Findings are for DINOv2 ViT-S/14; the dissociation hypothesis predicts a
  count-supervised encoder would concentrate numerosity (untested).

## 7. Related work (to verify before submission)
Spelke & Kinzler, *Core knowledge* (2007); Krotov & Hopfield, *Dense Associative Memory* (2016);
Ramsauer et al., *Hopfield Networks is All You Need* (2021); Oquab et al., *DINOv2* (2023); Johnson
et al., *CLEVR* (2017). [Citations to be checked and completed at submission time.]

## 8. Conclusion
Pre-seeding associative memory with core-knowledge priors is neither uniformly helpful nor
uniformly useless: it helps exactly when the prior is a dominant variance axis and is restored from
memory under noise. The decodability≠usability dissociation and the dual-bar gate are, we argue, the
right tools for deciding whether any candidate prior is worth seeding at all.

## Reproducibility & pre-registration
Every bar, prediction, and the five FactoredSDAM outcomes were committed before the runs
(`docs/superpowers/specs/`). Pure metric code is unit-tested with known-answer synthetic data
(`tests/`); runs are deterministic (`set_all_seeds`). Outputs in `results_archive/`.
