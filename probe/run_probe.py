"""Probe orchestration + verdict (Task 7) — the human-in-the-loop gate read.

STRICT ORDER:
  1. Load scenes + extract DINOv2 features
  2. Pairwise cosine stats + histograms     <- GATE THE GATE (degenerate space?)
  3. Per-factor decodability (bal_acc, permutation, margin, best_C, boundaries,
     spearman, R2)
  4. Variance gate (per-factor concentration + combined)
  5. VERDICT: GREEN / YELLOW / RED  (with reasoning, not just a label)

Reminder printed in output: the probe gates FEASIBILITY OF TESTING the
hypothesis, NOT the truth of the hypothesis.

Top-level imports are torch-free so compute_verdict is unit-testable without a
GPU. torch is imported lazily inside run().
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probe.clevr_factors import build_factor_arrays, load_clevr_scenes
from probe.cosine_check import pairwise_cosine_stats
from probe.decodability import decodability
from probe.variance_gate import variance_gate

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "config.yaml")
FACTORS = ["shape", "count", "layout"]
IS_CATEGORICAL = {"shape": True, "count": False, "layout": False}

PASS_SEMANTICS = (
    "A GREEN verdict means the Spelke factors are DECODABLE and NON-TRIVIAL AXES "
    "of the DINOv2-CLEVR feature space -- i.e. the hypothesis is TESTABLE here. "
    "It does NOT mean S-DAM will show an effect. The probe gates FEASIBILITY OF "
    "TESTING, not truth of the hypothesis."
)
SHAPE_PROXY_NOTE = (
    "shape->Objectness is a PROXY (object identity), not full Spelke Objectness "
    "(permanence/cohesion/solidity). Agentness is intentionally absent (CLEVR has "
    "no agents): this is a 3-factor probe."
)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_split(n: int, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(n * train_frac)
    return perm[:k], perm[k:]


def compute_verdict(decod: dict, gate: dict, cfg: dict) -> dict:
    """Pure verdict logic (torch-free, unit-tested).

    GREEN  = all 3 factors decodable (bal_acc>=bar AND margin>=min_margin)
             AND gate passes (combined>combined_bar AND min_conc>floor)
    RED    = >=2 factors not decodable, OR combined < variance_combined_red
    YELLOW = anything in between
    """
    g = cfg["gate"]
    decodable = {
        f: (r.balanced_acc >= g["decodability_bar"] and r.margin >= g["decodability_min_margin"])
        for f, r in decod.items()
    }
    n_decodable = int(sum(decodable.values()))
    combined = float(gate["combined"])
    min_conc = float(min(gate["per_factor"].values()))
    gate_pass = combined > g["variance_combined_bar"] and min_conc > g["variance_concentration_floor"]
    dead = [f for f, c in gate["per_factor"].items() if c <= g["variance_concentration_floor"]]

    if n_decodable == 3 and gate_pass:
        verdict = "GREEN"
    elif (3 - n_decodable) >= 2 or combined < g.get("variance_combined_red", 0.05):
        verdict = "RED"
    else:
        verdict = "YELLOW"

    return {
        "verdict": verdict,
        "decodable": decodable,
        "n_decodable": n_decodable,
        "gate_pass": bool(gate_pass),
        "dead_factors": dead,
        "combined": combined,
        "min_concentration": min_conc,
    }


def _plot_cosine(cos: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    rc, re = cos["histogram_raw"]
    cc, ce = cos["histogram_centered"]
    centers_r = 0.5 * (np.array(re[:-1]) + np.array(re[1:]))
    centers_c = 0.5 * (np.array(ce[:-1]) + np.array(ce[1:]))
    ax.plot(centers_r, rc, label=f"raw (mean={cos['raw_mean']:.3f}, std={cos['raw_std']:.3f})", color="tomato")
    ax.plot(centers_c, cc, label=f"centered (mean={cos['centered_mean']:.3f}, std={cos['centered_std']:.3f})",
            color="steelblue")
    ax.axvline(0.0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("pairwise cosine similarity")
    ax.set_ylabel("count")
    ax.set_title("CLEVR pairwise cosine: raw vs mean-centered (gate-the-gate)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_concentration(gate: dict, cfg: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(gate["per_factor"])
    vals = [gate["per_factor"][k] for k in names]
    ax.bar(names, vals, color="seagreen")
    ax.axhline(cfg["gate"]["variance_concentration_floor"], color="red", ls="--",
               label=f"floor ({cfg['gate']['variance_concentration_floor']})")
    ax.axhline(1.0, color="gray", ls=":", label="random (1x)")
    ax.set_ylabel("centered concentration (x random)")
    ax.set_title(f"Per-factor concentration | combined span = {gate['combined']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(config_path: str = CONFIG_PATH) -> dict:
    cfg = load_config(config_path)

    # 1. scenes + features ---------------------------------------------------
    scenes = load_clevr_scenes(cfg["data"]["clevr_scenes"], cfg["data"]["max_scenes"])
    factors = build_factor_arrays(scenes)
    image_paths = [os.path.join(cfg["data"]["clevr_images"], s["image_filename"]) for s in scenes]

    import torch  # lazy

    from probe.encoder import extract_features, get_dinov2_encoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_dinov2_encoder(device)
    X = extract_features(model, image_paths, device, batch_size=cfg["encoder"]["batch_size"])
    print(f"[features] X shape = {X.shape}  device={device}")

    # 2. cosine (gate-the-gate) ---------------------------------------------
    cos = pairwise_cosine_stats(X)
    degenerate = cos["centered_mean"] > 0.9 and cos["centered_std"] < 0.05
    print("=" * 70)
    print("STEP 2 — pairwise cosine (gate-the-gate)")
    print(f"  raw      mean={cos['raw_mean']:.4f} std={cos['raw_std']:.4f} "
          f"[p05={cos['raw_p05']:.3f}, p95={cos['raw_p95']:.3f}]")
    print(f"  centered mean={cos['centered_mean']:.4f} std={cos['centered_std']:.4f} "
          f"[p05={cos['centered_p05']:.3f}, p95={cos['centered_p95']:.3f}]")
    if cos["centered_std"] < 0.05:
        print("  !! WARNING: centered cosine std < 0.05 -> features may be DEGENERATE; "
              "downstream metrics computed in a collapsed space.")

    # 3. decodability --------------------------------------------------------
    n = X.shape[0]
    tr, te = make_split(n, cfg["split"]["train_frac"], cfg["split"]["seed"])
    decod = {}
    print("=" * 70)
    print("STEP 3 — decodability (held-out balanced 3-class accuracy)")
    for f in FACTORS:
        r = decodability(X, factors[f], f, IS_CATEGORICAL[f], tr, te, seed=cfg["split"]["seed"])
        decod[f] = r
        extra = ""
        if r.tertile_boundaries is not None:
            extra = (f"  tertiles={[round(b, 3) for b in r.tertile_boundaries]}"
                     f"  spearman={r.spearman_rho:.3f}  R2={r.multivariate_r2:.3f}")
        print(f"  {f:7s} bal_acc={r.balanced_acc:.3f}  perm={r.permutation_acc:.3f}  "
              f"margin={r.margin:.3f}  best_C={r.best_C}{extra}")

    # 4. variance gate -------------------------------------------------------
    gate = variance_gate(X, factors, IS_CATEGORICAL, ridge_lambda=cfg["ridge_lambda"])
    print("=" * 70)
    print("STEP 4 — variance gate (centered)")
    for f, c in gate["per_factor"].items():
        print(f"  {f:7s} concentration = {c:.2f}x random")
    print(f"  combined orthonormalized span = {gate['combined']:.4f}")

    # 5. verdict -------------------------------------------------------------
    v = compute_verdict(decod, gate, cfg)
    print("=" * 70)
    print(f"VERDICT: {v['verdict']}")
    print(f"  decodable factors: {v['decodable']}  ({v['n_decodable']}/3)")
    print(f"  gate_pass: {v['gate_pass']}  (combined {v['combined']:.3f} > "
          f"{cfg['gate']['variance_combined_bar']} AND min_conc {v['min_concentration']:.2f} > "
          f"{cfg['gate']['variance_concentration_floor']})")
    if v["dead_factors"]:
        print(f"  DEAD FACTORS (concentration <= floor): {v['dead_factors']}")
    print("-" * 70)
    print(f"  {PASS_SEMANTICS}")
    print(f"  {SHAPE_PROXY_NOTE}")
    print("=" * 70)

    # outputs ----------------------------------------------------------------
    results = {
        "experiment": "clevr_feasibility_probe",
        "verdict": v["verdict"],
        "passed": v["verdict"] == "GREEN",
        "spread_degenerate": bool(degenerate),
        "pass_semantics": PASS_SEMANTICS,
        "shape_proxy_note": SHAPE_PROXY_NOTE,
        "cosine": {k: val for k, val in cos.items() if not k.startswith("histogram")},
        "decodability": {f: dataclasses.asdict(r) for f, r in decod.items()},
        "variance_gate": {
            "per_factor": gate["per_factor"],
            "combined": gate["combined"],
        },
        "verdict_detail": {k: val for k, val in v.items() if k != "decodable"},
        "decodable": v["decodable"],
        "config_gate": cfg["gate"],
        "n_scenes": int(n),
        "encoder": cfg["encoder"]["model"],
    }
    with open("probe_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    _plot_cosine(cos, "cosine_histogram.png")
    _plot_concentration(gate, cfg, "concentration_bars.png")
    print("[saved] probe_results.json, cosine_histogram.png, concentration_bars.png")
    return results


if __name__ == "__main__":
    run()
