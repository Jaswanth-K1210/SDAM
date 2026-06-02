"""Phase 1 -- Retrieval accuracy under corruption (Lemma 1).

Compares S-DAM (Spelke seeds + residual encoding) against a Baseline (random
seeds, raw inputs stored, no residual) across corruption rates 0.1 .. 0.9,
averaged over 5 random seeds.

PASS criterion: S-DAM accuracy >= Baseline + 5% at 30% corruption.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments._common import ensure_results_dir, get_features, load_config
from sdam.hopfield import HopfieldLayer
from sdam.model import SDAM
from sdam.utils import corruption_mask, cosine_similarity_matrix, set_all_seeds


def _stack_patterns(feats, n_total):
    cats = list(feats.values())
    per = max(1, n_total // len(cats))
    chunks = [c[:per] for c in cats]
    x = torch.cat(chunks, dim=0)
    return x[:n_total]


def _accuracy(recovered, originals):
    sims = cosine_similarity_matrix(recovered.detach(), originals).diagonal()
    return float(sims.mean())


def run() -> dict:
    cfg = load_config()
    p1 = cfg["phase1"]
    feats, synthetic = get_features(cfg)

    corruption_rates = p1["corruption_rates"]
    n_patterns = p1["n_seeds"]
    n_repeats = 5

    sdam_curves = {r: [] for r in corruption_rates}
    base_curves = {r: [] for r in corruption_rates}

    for rep in range(n_repeats):
        set_all_seeds(42 + rep)
        patterns = _stack_patterns(feats, n_patterns)
        dim = patterns.shape[1]

        # S-DAM: residual encoding with Spelke seeds.
        sdam = SDAM(
            input_dim=dim,
            beta=cfg["model"]["beta"],
            write_threshold=cfg["model"]["write_threshold"],
            use_high_inertia=cfg["model"]["use_high_inertia"],
        )
        sdam.reset_memory()
        for p in patterns:
            sdam.write(p)

        # Baseline: raw patterns stored directly in a plain Hopfield memory.
        baseline = HopfieldLayer(dim, beta=cfg["model"]["beta"], max_iter=cfg["model"]["max_iter"])
        baseline.store(patterns)

        for r in corruption_rates:
            corrupted = corruption_mask(patterns, r)
            if sdam.mem.n_stored > 0:
                sdam_curves[r].append(_accuracy(sdam.read(corrupted), patterns))
            else:
                sdam_curves[r].append(0.0)
            base_curves[r].append(_accuracy(baseline.retrieve(corrupted), patterns))

    sdam_mean = {r: float(np.mean(v)) for r, v in sdam_curves.items()}
    sdam_std = {r: float(np.std(v)) for r, v in sdam_curves.items()}
    base_mean = {r: float(np.mean(v)) for r, v in base_curves.items()}
    base_std = {r: float(np.std(v)) for r, v in base_curves.items()}

    # 30% corruption check (nearest available rate to 0.30).
    r30 = min(corruption_rates, key=lambda r: abs(r - 0.30))
    improvement = sdam_mean[r30] - base_mean[r30]
    passed = improvement >= p1.get("min_improvement", 0.05)

    print("=" * 64)
    print("PHASE 1 -- Retrieval accuracy under corruption (Lemma 1)")
    print(f"  data       : {'SYNTHETIC' if synthetic else 'CLEVR'}")
    for r in corruption_rates:
        print(f"  rate {r:.1f} : S-DAM {sdam_mean[r]:.3f}±{sdam_std[r]:.3f}   "
              f"Baseline {base_mean[r]:.3f}±{base_std[r]:.3f}")
    print(f"  @rate {r30:.2f}: improvement = {improvement:+.3f} (require >= {p1.get('min_improvement', 0.05)})")
    print(f"  VERDICT    : {'PASSED' if passed else 'FAILED'}")
    print("=" * 64)

    results = {
        "experiment": "phase1_retrieval",
        "synthetic": synthetic,
        "corruption_rates": corruption_rates,
        "sdam_mean": sdam_mean,
        "sdam_std": sdam_std,
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "check_rate": r30,
        "improvement": improvement,
        "passed": passed,
    }

    out_dir = ensure_results_dir(cfg)
    with open(os.path.join(out_dir, "phase1_retrieval.json"), "w") as f:
        json.dump(results, f, indent=2)
    _plot(corruption_rates, sdam_mean, sdam_std, base_mean, base_std,
          os.path.join(out_dir, "phase1_accuracy.png"))
    return results


def _plot(rates, sm, ss, bm, bs, path):
    rates = list(rates)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(rates, [sm[r] for r in rates], yerr=[ss[r] for r in rates], label="S-DAM", marker="o")
    ax.errorbar(rates, [bm[r] for r in rates], yerr=[bs[r] for r in rates], label="Baseline", marker="s")
    ax.set_xlabel("corruption rate")
    ax.set_ylabel("retrieval accuracy (mean cosine)")
    ax.set_title("Phase 1: retrieval accuracy vs corruption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
