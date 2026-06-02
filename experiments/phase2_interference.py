"""Phase 2 -- Cross-category interference (Lemma 2). PRIMARY EXPERIMENT.

Falsifiable claim: residual encoding makes same-category retrieval associative
(high overlap) while keeping cross-category retrieval near-orthogonal (low
overlap). Concretely, after storing a pair (A, B) and querying with A:

    interference = cosine(read(A), B)

We collect 100 same-category and 100 cross-category interference scores, run a
one-sided Mann-Whitney U test, and check three criteria:

    cross mean < 0.20   AND   same mean > 0.40   AND   p < 0.01

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


def _interference(model: SDAM, a: torch.Tensor, b: torch.Tensor) -> float:
    """Store both patterns, query with A, return cosine(read(A), B)."""
    model.reset_memory()
    model.write(a)
    model.write(b)
    if model.mem.n_stored == 0:
        return 0.0
    retrieved = model.read(a).detach()
    return float(cosine_similarity_matrix(retrieved, b).reshape(-1)[0])


def run() -> dict:
    set_all_seeds(42)
    cfg = load_config()
    p2 = cfg["phase2"]
    feats, synthetic = get_features(cfg)
    categories = list(feats.keys())

    model = SDAM(
        input_dim=cfg["model"]["seed_dim"],
        beta=cfg["model"]["beta"],
        write_threshold=cfg["model"]["write_threshold"],
        failure_threshold=cfg["model"]["failure_threshold"],
        use_high_inertia=cfg["model"]["use_high_inertia"],
    )

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

    # --- Cross-category pairs: all 6 combinations, spread evenly ------------ #
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

    same_mean = float(sum(same_scores) / len(same_scores))
    cross_mean = float(sum(cross_scores) / len(cross_scores))
    stat, p_value = mann_whitney_u(cross_scores, same_scores)

    passed = (
        cross_mean < p2["cross_threshold"]
        and same_mean > p2["same_threshold"]
        and p_value < p2["p_value_threshold"]
    )

    print("=" * 64)
    print("PHASE 2 -- Cross-category interference (Lemma 2)")
    print(f"  data            : {'SYNTHETIC' if synthetic else 'CLEVR'}")
    print(f"  same  mean      : {same_mean:.4f}  (require > {p2['same_threshold']})")
    print(f"  cross mean      : {cross_mean:.4f}  (require < {p2['cross_threshold']})")
    print(f"  Mann-Whitney U  : stat={stat:.1f}  p={p_value:.3e}  (require < {p2['p_value_threshold']})")
    print(f"  VERDICT         : {'PASSED' if passed else 'FAILED'}")
    print("=" * 64)

    results = {
        "experiment": "phase2_interference",
        "synthetic": synthetic,
        "same_mean": same_mean,
        "cross_mean": cross_mean,
        "mann_whitney_stat": stat,
        "p_value": p_value,
        "thresholds": {
            "cross_threshold": p2["cross_threshold"],
            "same_threshold": p2["same_threshold"],
            "p_value_threshold": p2["p_value_threshold"],
        },
        "passed": passed,
        "same_scores": same_scores,
        "cross_scores": cross_scores,
    }

    out_dir = ensure_results_dir(cfg)
    dump_json(results, os.path.join(out_dir, "phase2_interference.json"))

    _boxplot(same_scores, cross_scores, p2, os.path.join(out_dir, "phase2_boxplot.png"))
    return results


def _boxplot(same_scores, cross_scores, p2, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([same_scores, cross_scores], labels=["same-category", "cross-category"])
    ax.axhline(p2["same_threshold"], color="green", linestyle="--", label=f"same > {p2['same_threshold']}")
    ax.axhline(p2["cross_threshold"], color="red", linestyle="--", label=f"cross < {p2['cross_threshold']}")
    ax.set_ylabel("interference  cos(read(A), B)")
    ax.set_title("Phase 2: residual-encoding interference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
