"""Phase 2 -- Cross-category interference (Lemma 2). PRIMARY EXPERIMENT.

Falsifiable claim: residual encoding makes same-category retrieval associative
(high overlap) while keeping cross-category retrieval near-orthogonal (low
overlap). Concretely, after storing a pair (A, B) and querying with A:

    interference = cosine(read(A), B)

We collect same-category and cross-category interference scores, run a
one-sided Mann-Whitney U test, and check three criteria:

    cross mean < 0.20   AND   same mean > 0.40   AND   p < 0.01

3-WAY BASELINE: we run three model variants and compare them:
    spelke : standard S-DAM with Spelke-seeded orthonormal vectors
    random : same architecture but random orthonormal seeds
    none   : seeds zeroed out (pure Hopfield baseline)

The core novelty proof is: Spelke seeds specifically beat random seeds on
cross-category separation (spelke.cross_mean < random.cross_mean).

RUN THIS BEFORE Phase 1 and Phase 3. If only Phase 2 passes, you still have a
paper.
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments._common import dump_json, ensure_results_dir, get_features, load_config
from sdam.model import SDAM
from sdam.utils import cosine_similarity_matrix, mann_whitney_u, set_all_seeds


def _make_model(cfg, mode: str) -> SDAM:
    """
    mode: 'spelke' | 'random' | 'none'
    - spelke: standard S-DAM with Spelke-seeded orthonormal vectors (existing behavior)
    - random: same architecture but seeds initialized to random orthonormal vectors
    - none:   seeds zeroed out (projection contributes nothing -- pure Hopfield baseline)
    """
    import torch.nn as nn
    m = SDAM(
        input_dim=cfg["model"]["seed_dim"],
        beta=cfg["model"]["beta"],
        write_threshold=cfg["model"]["write_threshold"],
        failure_threshold=cfg["model"]["failure_threshold"],
        use_high_inertia=(mode == "spelke"),
    )
    if mode == "random":
        rand = torch.empty(4, cfg["model"]["seed_dim"])
        nn.init.orthogonal_(rand)
        m.ssl.seeds = nn.Parameter(rand)
    elif mode == "none":
        m.ssl.seeds = nn.Parameter(torch.zeros(4, cfg["model"]["seed_dim"]))
    return m


def _interference(model: SDAM, a: torch.Tensor, b: torch.Tensor) -> float:
    """Store both patterns, query with A, return cosine(read(A), B)."""
    model.reset_memory()
    model.write(a)
    model.write(b)
    if model.mem.n_stored == 0:
        return 0.0
    retrieved = model.read(a).detach()
    return float(cosine_similarity_matrix(retrieved, b).reshape(-1)[0])


def _measure(model: SDAM, feats: dict, p2: dict):
    """Collect same-category and cross-category interference scores for one model."""
    categories = list(feats.keys())

    # --- Same-category pairs: consecutive patterns within each category ----- #
    n_same = p2["n_same_pairs"]
    per_cat = max(1, n_same // len(categories))
    same_scores = []
    for cat in categories:
        x = feats[cat]
        for i in range(per_cat):
            if i + 1 >= x.shape[0] or len(same_scores) >= n_same:
                break
            same_scores.append(_interference(model, x[i], x[i + 1]))
    # top up if integer division left us short
    ci = 0
    while len(same_scores) < n_same:
        cat = categories[ci % len(categories)]
        x = feats[cat]
        j = len(same_scores)
        same_scores.append(_interference(model, x[j % (x.shape[0] - 1)], x[j % (x.shape[0] - 1) + 1]))
        ci += 1

    # --- Cross-category pairs: all combinations, spread evenly -------------- #
    n_cross = p2["n_cross_pairs"]
    combos = list(itertools.combinations(range(len(categories)), 2))
    per_combo = max(1, n_cross // len(combos))
    cross_scores = []
    for (ai, bi) in combos:
        xa, xb = feats[categories[ai]], feats[categories[bi]]
        for i in range(per_combo):
            if len(cross_scores) >= n_cross:
                break
            cross_scores.append(_interference(model, xa[i], xb[i]))
    ci = 0
    while len(cross_scores) < n_cross:
        ai, bi = combos[ci % len(combos)]
        xa, xb = feats[categories[ai]], feats[categories[bi]]
        j = len(cross_scores)
        cross_scores.append(_interference(model, xa[j % xa.shape[0]], xb[j % xb.shape[0]]))
        ci += 1

    return same_scores, cross_scores


def run() -> dict:
    set_all_seeds(42)
    cfg = load_config()
    p2 = cfg["phase2"]
    feats, synthetic = get_features(cfg)

    modes = ["spelke", "random", "none"]
    results = {}
    for mode in modes:
        # Re-seed before each model so the random-seed init and pair sampling
        # are reproducible across modes.
        set_all_seeds(42)
        model = _make_model(cfg, mode)
        same_scores, cross_scores = _measure(model, feats, p2)

        same_mean = float(sum(same_scores) / len(same_scores))
        cross_mean = float(sum(cross_scores) / len(cross_scores))
        stat, p_value = mann_whitney_u(cross_scores, same_scores)
        passed = (
            cross_mean < p2["cross_threshold"]
            and same_mean > p2["same_threshold"]
            and p_value < p2["p_value_threshold"]
        )
        results[mode] = {
            "same_mean": same_mean,
            "cross_mean": cross_mean,
            "mann_whitney_stat": stat,
            "p_value": p_value,
            "passed": passed,
            "same_scores": same_scores,
            "cross_scores": cross_scores,
        }

    # Overall verdict: Spelke passes AND Spelke specifically beats random seeds
    # on cross-category separation (the core novelty claim).
    spelke_beats_random = results["spelke"]["cross_mean"] < results["random"]["cross_mean"]
    overall_passed = results["spelke"]["passed"] and spelke_beats_random

    print("=" * 64)
    print("PHASE 2 -- Cross-category interference (Lemma 2)  [3-WAY BASELINE]")
    print(f"  data : {'SYNTHETIC' if synthetic else 'REAL'}")
    for mode in modes:
        r = results[mode]
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f"[{mode:7s}] same={r['same_mean']:.4f}  cross={r['cross_mean']:.4f}  "
              f"p={r['p_value']:.2e}  {verdict}")
    print(f"  spelke beats random (cross) : {spelke_beats_random}")
    print(f"  VERDICT : {'PASSED' if overall_passed else 'FAILED'}")
    print("=" * 64)

    # Final JSON: per-mode summary plus metadata. Keep top-level passed/synthetic.
    out = {
        "experiment": "phase2_interference",
        "synthetic": synthetic,
        "spelke_beats_random": spelke_beats_random,
        "passed": overall_passed,
        "thresholds": {
            "cross_threshold": p2["cross_threshold"],
            "same_threshold": p2["same_threshold"],
            "p_value_threshold": p2["p_value_threshold"],
        },
    }
    for mode in modes:
        r = results[mode]
        out[mode] = {
            "same_mean": r["same_mean"],
            "cross_mean": r["cross_mean"],
            "p_value": r["p_value"],
            "passed": r["passed"],
        }

    out_dir = ensure_results_dir(cfg)
    dump_json(out, os.path.join(out_dir, "phase2_interference.json"))

    _boxplot_3way(results, modes, p2, os.path.join(out_dir, "phase2_3way.png"))
    return out


def _boxplot_3way(results, modes, p2, path):
    fig, axes = plt.subplots(1, len(modes), figsize=(13, 5), sharey=True)
    for ax, mode in zip(axes, modes):
        r = results[mode]
        # matplotlib >= 3.9 renamed `labels` -> `tick_labels`; support both.
        try:
            ax.boxplot([r["same_scores"], r["cross_scores"]],
                       tick_labels=["same", "cross"])
        except TypeError:
            ax.boxplot([r["same_scores"], r["cross_scores"]],
                       labels=["same", "cross"])
        ax.axhline(p2["same_threshold"], color="green", linestyle="--", lw=0.8)
        ax.axhline(p2["cross_threshold"], color="red", linestyle="--", lw=0.8)
        ax.set_title(f"{mode} seeds\nsame={r['same_mean']:.3f}  cross={r['cross_mean']:.3f}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("interference  cos(read(A), B)")
    fig.suptitle("Phase 2: residual-encoding interference (3-way baseline)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
