"""FactoredSDAM pipeline (Milestone 3).

4-way Phase-1 retrieval under corruption: factored vs objectness vs random vs
zeroed. Reads the verdict at the variance-matched w (1.0x); runs 0.5x/2x as a
pre-registered ROBUSTNESS check (not a search). Five pre-registered outcomes
(spec docs/superpowers/specs/2026-06-07-factored-sdam-design.md):
  WORKS / PARTIAL_FIX / COLLAPSE / STILL_HURTS / INCONCLUSIVE.

Verdict logic (classify, factored_verdict) is pure and unit-tested. The SDAM /
FactoredSDAM retrieval run is torch and executes on Colab.
"""
from __future__ import annotations

import json
import os

import numpy as np
import yaml

from sdam.factored import crossover_point

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CONFIG_PATH = os.path.join(_ROOT, "probe", "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Pure verdict logic (no torch) — the 5 pre-registered outcomes
# --------------------------------------------------------------------------- #
def classify(factored, objectness, zeroed, random_, rates,
             min_low_gain=0.02, collapse_tol=0.02, lowcorr=0.3, floor=0.3) -> str:
    """Classify ONE w-setting into WORKS / COLLAPSE / PARTIAL_FIX / STILL_HURTS.

    All args except rates are {rate: mean_accuracy}.
    """
    base = {r: max(zeroed[r], random_[r]) for r in rates}
    f_gain = [factored[r] - base[r] for r in rates]
    o_gain = [objectness[r] - base[r] for r in rates]
    f_x = crossover_point(rates, f_gain)
    o_x = crossover_point(rates, o_gain)

    # COLLAPSE: factored indistinguishable from zeroed everywhere
    if all(abs(factored[r] - zeroed[r]) < collapse_tol for r in rates):
        return "COLLAPSE"

    # WORKS: >= both baselines at all rates >= floor, real low-corruption gain,
    # and no crossover (harm eliminated)
    works = (
        all(factored[r] >= base[r] for r in rates if r >= floor)
        and any(factored[r] - zeroed[r] >= min_low_gain for r in rates if r <= lowcorr)
        and f_x is None
    )
    if works:
        return "WORKS"

    # PARTIAL_FIX: factored still crosses over, but the crossover moved RIGHT vs objectness
    if f_x is not None and o_x is not None and f_x > o_x + 1e-9:
        return "PARTIAL_FIX"

    # otherwise the harm is ~unchanged from objectness
    return "STILL_HURTS"


def factored_verdict(by_w: dict) -> dict:
    """Combine per-w classifications. by_w: {w_factor: classification}.

    INCONCLUSIVE if the classification is not identical across all w (channel
    scaling, not the mechanism, drives the result). Otherwise the common verdict,
    reported as read at the variance-matched 1.0x.
    """
    classes = set(by_w.values())
    if len(classes) > 1:
        verdict = "INCONCLUSIVE"
    else:
        verdict = next(iter(classes))
    return {
        "verdict": verdict,
        "per_w": by_w,
        "primary_w": 1.0,
        "passed": verdict == "WORKS",
    }


# --------------------------------------------------------------------------- #
# Orchestration (torch / GPU — Colab)
# --------------------------------------------------------------------------- #
def run(config_path: str = CONFIG_PATH) -> dict:
    import torch

    from probe.clevr_factors import build_factor_arrays, load_clevr_scenes
    from probe.encoder import extract_features, get_dinov2_encoder
    from probe.feature_cache import cached_features
    from probe.variance_gate import factor_seed_direction
    from sdam.factored import FactoredSDAM, augment, factor_pattern, variance_matched_w
    from sdam.utils import corruption_mask, cosine_similarity_matrix, set_all_seeds

    import experiments.objectness_pipeline as objp

    cfg = load_config(config_path)
    op = cfg["objectness_pipeline"]
    fp = cfg["factored_pipeline"]
    set_all_seeds(cfg["split"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # features (cached) + factors
    scenes = load_clevr_scenes(cfg["data"]["clevr_scenes"], cfg["data"]["max_scenes"])
    factors = build_factor_arrays(scenes)
    image_paths = [os.path.join(cfg["data"]["clevr_images"], s["image_filename"]) for s in scenes]
    enc_holder = {}

    def _extract(paths):
        if "m" not in enc_holder:
            enc_holder["m"] = get_dinov2_encoder(device)
        return extract_features(enc_holder["m"], paths, device, batch_size=cfg["encoder"]["batch_size"])

    X = cached_features(op["feature_cache_dir"], cfg["encoder"]["model"], image_paths,
                        "resize224-centercrop-imagenetnorm", _extract).astype(np.float64)
    shape = factors["shape"]

    # split + mean-center (train mean)
    n = X.shape[0]
    rng = np.random.default_rng(cfg["split"]["seed"])
    perm = rng.permutation(n)
    k = int(n * cfg["split"]["train_frac"])
    tr, te = perm[:k], perm[k:]
    Xc = (X - X[tr].mean(0, keepdims=True)).astype(np.float32)
    dim = Xc.shape[1]
    seed_dir = factor_seed_direction(Xc[tr], shape[tr], is_categorical=True)

    # variance-matched w from TRAIN residuals/coeffs
    r_tr, c_tr = factor_pattern(Xc[tr], seed_dir)
    w_star = variance_matched_w(r_tr, c_tr)

    Xte = Xc[te]
    rates = op["corruption_rates"]
    n_store = min(op["n_store"], Xte.shape[0])
    store_np = Xte[:n_store]
    store_t = torch.tensor(store_np, dtype=torch.float32)

    # baselines (SDAM objectness/random/zeroed) — independent of w
    base_curves = {}
    for mode in ["objectness", "random", "zeroed"]:
        per_rate = {r: [] for r in rates}
        for rep in range(op["n_repeats"]):
            set_all_seeds(cfg["split"]["seed"] + rep)
            m = objp._build_model(mode, dim, seed_dir, cfg)
            m.reset_memory()
            for p in store_t:
                m.write(p)
            for r in rates:
                corrupted = corruption_mask(store_t, r)
                rec = m.read(corrupted).detach()
                per_rate[r].append(float(cosine_similarity_matrix(rec, store_t).diagonal().mean()))
        base_curves[mode] = {r: float(np.mean(per_rate[r])) for r in rates}

    # factored at each w factor
    by_w = {}
    factored_curves = {}
    crossovers = {}
    for wf in fp["w_factors"]:
        w = w_star * wf
        per_rate = {r: [] for r in rates}
        for rep in range(op["n_repeats"]):
            set_all_seeds(cfg["split"]["seed"] + rep)
            fm = FactoredSDAM(seed_dir, w, beta=op["beta"])
            fm.reset()
            fm.store(store_np)
            for r in rates:
                corrupted = corruption_mask(store_t, r).numpy()
                rec = fm.read(corrupted)
                rec_t = torch.tensor(rec, dtype=torch.float32)
                per_rate[r].append(float(cosine_similarity_matrix(rec_t, store_t).diagonal().mean()))
        fcurve = {r: float(np.mean(per_rate[r])) for r in rates}
        factored_curves[str(wf)] = fcurve
        by_w[wf] = classify(fcurve, base_curves["objectness"], base_curves["zeroed"],
                            base_curves["random"], rates,
                            min_low_gain=fp["min_low_gain"], collapse_tol=fp["collapse_tol"])
        base = {r: max(base_curves["zeroed"][r], base_curves["random"][r]) for r in rates}
        crossovers[str(wf)] = crossover_point(rates, [fcurve[r] - base[r] for r in rates])

    obj_base = {r: max(base_curves["zeroed"][r], base_curves["random"][r]) for r in rates}
    obj_cross = crossover_point(rates, [base_curves["objectness"][r] - obj_base[r] for r in rates])
    verdict = factored_verdict(by_w)

    # report
    print("=" * 72)
    print("FACTORED SDAM PIPELINE  (verdict at variance-matched w=1.0x)")
    print(f"  variance-matched w* = {w_star:.4f}")
    print(f"  objectness crossover = {obj_cross}")
    for wf in fp["w_factors"]:
        print(f"  w={wf}x : classify={by_w[wf]:11s}  crossover={crossovers[str(wf)]}")
    print(f"  VERDICT: {verdict['verdict']}  (per_w={by_w})")
    print("=" * 72)

    results = {
        "experiment": "factored_pipeline",
        "synthetic": False,
        "verdict": verdict["verdict"],
        "passed": verdict["passed"],
        "w_star": w_star,
        "per_w_classification": {str(k): v for k, v in by_w.items()},
        "factored_crossovers": crossovers,
        "objectness_crossover": obj_cross,
        "baseline_curves": base_curves,
        "factored_curves": factored_curves,
        "rates": rates,
        "scope_note": "single-factor (objectness) factored-storage test; w never tuned.",
    }
    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "factored_pipeline.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    _plot(base_curves, factored_curves, rates, verdict, os.path.join(out_dir, "factored_phase1.png"))
    print("[saved] results/factored_pipeline.json, results/factored_phase1.png")
    return results


def _plot(base_curves, factored_curves, rates, verdict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, mk in [("objectness", "o"), ("random", "s"), ("zeroed", "^")]:
        ax.plot(rates, [base_curves[mode][r] for r in rates], marker=mk, label=mode, alpha=0.8)
    for wf, fc in factored_curves.items():
        ax.plot(rates, [fc[r] for r in rates], marker="*", ls="--", label=f"factored w={wf}x")
    ax.set_xlabel("corruption rate")
    ax.set_ylabel("retrieval accuracy (mean cosine)")
    ax.set_title(f"FactoredSDAM Phase 1 | VERDICT: {verdict['verdict']}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
