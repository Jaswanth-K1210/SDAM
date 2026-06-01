"""Phase 3 -- Developmental curriculum ordering (Theorem 1).

Claim: introducing categories in Spelke developmental order
(OBJECTNESS -> AGENTNESS -> NUMEROSITY -> GEOMETRY) yields monotone capacity
growth and beats random orderings.

PASS criterion: Spelke-order capacity is non-decreasing across all 4 stages
AND beats the random-ordering mean at >= 3 of the 4 stages.
"""

from __future__ import annotations

import json
import os
import random as _random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments._common import ensure_results_dir, get_features, load_config
from sdam.model import SDAM
from sdam.utils import capacity_search, set_all_seeds


def _capacity_curve(model, feats, order, accuracy_threshold):
    """Cumulative capacity after introducing each category in ``order``."""
    curve = []
    introduced = []
    for cat in order:
        introduced.append(feats[cat])
        patterns = torch.cat(introduced, dim=0)
        cap = capacity_search(model, patterns, accuracy_threshold=accuracy_threshold, step=10)
        curve.append(cap)
    return curve


def run() -> dict:
    set_all_seeds(42)
    cfg = load_config()
    p3 = cfg["phase3"]
    feats, synthetic = get_features(cfg)

    spelke_order = p3["spelke_order"]
    acc_thr = p3["capacity_accuracy_threshold"]

    def fresh_model():
        return SDAM(
            input_dim=cfg["model"]["seed_dim"],
            beta=cfg["model"]["beta"],
            write_threshold=cfg["model"]["write_threshold"],
            use_high_inertia=cfg["model"]["use_high_inertia"],
        )

    spelke_curve = _capacity_curve(fresh_model(), feats, spelke_order, acc_thr)

    # Random orderings.
    categories = list(feats.keys())
    rng = _random.Random(42)
    random_curves = []
    for _ in range(p3["n_random_orderings"]):
        order = categories[:]
        rng.shuffle(order)
        random_curves.append(_capacity_curve(fresh_model(), feats, order, acc_thr))

    random_arr = np.array(random_curves, dtype=float)  # (n_orderings, 4)
    random_mean = random_arr.mean(axis=0)
    random_std = random_arr.std(axis=0)

    non_decreasing = all(spelke_curve[i + 1] >= spelke_curve[i] for i in range(len(spelke_curve) - 1))
    beats = int(sum(spelke_curve[i] > random_mean[i] for i in range(len(spelke_curve))))
    passed = non_decreasing and beats >= 3

    print("=" * 64)
    print("PHASE 3 -- Curriculum ordering (Theorem 1)")
    print(f"  data            : {'SYNTHETIC' if synthetic else 'CLEVR'}")
    print(f"  spelke curve    : {spelke_curve}")
    print(f"  random mean     : {[round(float(m), 1) for m in random_mean]}")
    print(f"  non-decreasing  : {non_decreasing}")
    print(f"  beats random at : {beats}/4 stages (require >= 3)")
    print(f"  VERDICT         : {'PASSED' if passed else 'FAILED'}")
    print("=" * 64)

    results = {
        "experiment": "phase3_curriculum",
        "synthetic": synthetic,
        "spelke_order": spelke_order,
        "spelke_curve": spelke_curve,
        "random_curves": random_curves,
        "random_mean": random_mean.tolist(),
        "random_std": random_std.tolist(),
        "non_decreasing": non_decreasing,
        "beats_random_stages": beats,
        "passed": passed,
    }

    out_dir = ensure_results_dir(cfg)
    with open(os.path.join(out_dir, "phase3_curriculum.json"), "w") as f:
        json.dump(results, f, indent=2)
    _plot(spelke_curve, random_mean, random_std, os.path.join(out_dir, "phase3_capacity.png"))
    return results


def _plot(spelke_curve, random_mean, random_std, path):
    stages = list(range(1, len(spelke_curve) + 1))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(stages, spelke_curve, marker="o", label="Spelke order")
    ax.errorbar(stages, random_mean, yerr=random_std, marker="s", label="random mean ± std")
    ax.set_xlabel("curriculum stage (#categories introduced)")
    ax.set_ylabel("storage capacity")
    ax.set_title("Phase 3: capacity vs curriculum ordering")
    ax.set_xticks(stages)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
