"""Objectness-only S-DAM pipeline (Milestone 2).

Tests whether the objectness (shape) prior — the ONE factor the probe found both
decodable AND a dominant variance axis (89x) — lets S-DAM's residual coding beat
random and zeroed seeds.

P1 (retrieval under corruption) is the SOLE ADJUDICATOR of WORKS/NULL.
Phase 2 (two-level interference) is a DIAGNOSTIC ONLY — it explains how the seed
moved the geometry; it never declares the verdict. Both pre-registered in
docs/superpowers/specs/2026-06-07-objectness-pipeline-design.md.

Top-level imports are torch-free so phase1_verdict is unit-testable without a GPU.
torch / SDAM are imported lazily inside run().
"""
from __future__ import annotations

import json
import os

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CONFIG_PATH = os.path.join(_ROOT, "probe", "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# P1 verdict — pure, unit-tested (the sole adjudicator)
# --------------------------------------------------------------------------- #
def phase1_verdict(
    obj: dict, rand: dict, zero: dict, rates: list,
    min_gain: float = 0.05, corruption_floor: float = 0.3,
) -> dict:
    """Decide WORKS / NULL from Phase-1 retrieval curves.

    obj/rand/zero: {rate: (mean, std)} retrieval accuracy per seed-mode.
    WORKS iff, at corruption >= corruption_floor, the objectness-seed beats the
    BEST baseline by >= min_gain AND by more than the pooled std, at a majority
    of those rates (and never reverses below a baseline there).
    """
    per_rate = []
    reversal = False
    for r in rates:
        if r < corruption_floor:
            continue
        om, os_ = obj[r]
        rm, rs = rand[r]
        zm, zs = zero[r]
        best_m = max(rm, zm)
        best_s = rs if rm >= zm else zs
        gain = om - best_m
        pooled = float(np.sqrt(os_ ** 2 + best_s ** 2))
        passed = (gain >= min_gain) and (gain > pooled)
        if om < best_m:
            reversal = True
        per_rate.append({"rate": r, "gain": gain, "pooled_std": pooled, "passed": passed})

    n = len(per_rate)
    n_pass = sum(p["passed"] for p in per_rate)
    works = (n > 0) and (n_pass >= (n + 1) // 2) and not reversal
    return {
        "verdict": "WORKS" if works else "NULL",
        "works": works,
        "n_pass": n_pass,
        "n_rates_considered": n,
        "reversal": reversal,
        "per_rate": per_rate,
    }


# --------------------------------------------------------------------------- #
# Seed construction (torch) — built lazily inside run()
# --------------------------------------------------------------------------- #
def _build_model(mode: str, dim: int, seed_dir: np.ndarray, cfg: dict):
    """SDAM with a single objectness/random/zeroed seed (1, dim)."""
    import torch
    import torch.nn as nn

    from sdam.model import SDAM

    m = SDAM(
        input_dim=dim,
        beta=cfg["objectness_pipeline"]["beta"],
        write_threshold=cfg["objectness_pipeline"]["write_threshold"],
        use_high_inertia=(mode == "objectness"),
    )
    if mode == "objectness":
        s = torch.tensor(seed_dir, dtype=torch.float32).reshape(1, dim)
    elif mode == "random":
        g = torch.Generator().manual_seed(0)
        v = torch.randn(1, dim, generator=g)
        s = v / v.norm()
    elif mode == "zeroed":
        s = torch.zeros(1, dim)
    else:
        raise ValueError(mode)
    m.ssl.seeds = nn.Parameter(s)
    m.ssl.n_systems = 1
    return m


# --------------------------------------------------------------------------- #
# Orchestration (torch / GPU — runs on Colab)
# --------------------------------------------------------------------------- #
def run(config_path: str = CONFIG_PATH) -> dict:
    import torch

    from probe.clevr_factors import build_factor_arrays, load_clevr_scenes
    from probe.encoder import extract_features, get_dinov2_encoder
    from probe.feature_cache import cached_features
    from probe.variance import covariance
    from probe.variance_gate import factor_seed_direction
    from sdam.utils import cosine_similarity_matrix, corruption_mask, set_all_seeds

    cfg = load_config(config_path)
    op = cfg["objectness_pipeline"]
    set_all_seeds(cfg["split"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. scenes + factors + features (cached) -------------------------------
    scenes = load_clevr_scenes(cfg["data"]["clevr_scenes"], cfg["data"]["max_scenes"])
    factors = build_factor_arrays(scenes)
    image_paths = [os.path.join(cfg["data"]["clevr_images"], s["image_filename"]) for s in scenes]

    model_enc = None

    def _extract(paths):
        nonlocal model_enc
        if model_enc is None:
            model_enc = get_dinov2_encoder(device)
        return extract_features(model_enc, paths, device, batch_size=cfg["encoder"]["batch_size"])

    X = cached_features(op["feature_cache_dir"], cfg["encoder"]["model"], image_paths,
                        "resize224-centercrop-imagenetnorm", _extract).astype(np.float64)
    shape = factors["shape"]

    # 2. split + mean-center (train mean only) ------------------------------
    n = X.shape[0]
    rng = np.random.default_rng(cfg["split"]["seed"])
    perm = rng.permutation(n)
    k = int(n * cfg["split"]["train_frac"])
    tr, te = perm[:k], perm[k:]
    mu = X[tr].mean(axis=0, keepdims=True)
    Xc = (X - mu).astype(np.float32)
    dim = Xc.shape[1]

    # 3. objectness seed direction (held-out, train only) -------------------
    seed_dir = factor_seed_direction(Xc[tr], shape[tr], is_categorical=True)

    Xte = torch.tensor(Xc[te], dtype=torch.float32)
    shape_te = shape[te]

    # 4. Phase 1 — retrieval under corruption (ADJUDICATOR) -----------------
    rates = op["corruption_rates"]
    n_store = min(op["n_store"], Xte.shape[0])
    store = Xte[:n_store]
    curves = {m: {r: [] for r in rates} for m in ["objectness", "random", "zeroed"]}

    for mode in ["objectness", "random", "zeroed"]:
        for rep in range(op["n_repeats"]):
            set_all_seeds(cfg["split"]["seed"] + rep)
            model = _build_model(mode, dim, seed_dir, cfg)
            model.reset_memory()
            for p in store:
                model.write(p)
            for r in rates:
                if model.mem.n_stored == 0:
                    curves[mode][r].append(0.0)
                    continue
                corrupted = corruption_mask(store, r)
                rec = model.read(corrupted).detach()
                acc = float(cosine_similarity_matrix(rec, store).diagonal().mean())
                curves[mode][r].append(acc)

    summary = {m: {r: (float(np.mean(v)), float(np.std(v))) for r, v in curves[m].items()}
               for m in curves}
    verdict = phase1_verdict(summary["objectness"], summary["random"], summary["zeroed"],
                             rates, op["p1_min_gain"], op["p1_corruption_floor"])

    # 5. Phase 2 — two-level interference (DIAGNOSTIC ONLY) -----------------
    phase2 = _phase2_diagnostic(Xte, shape_te, dim, seed_dir, cfg, op)

    # 6. report + save ------------------------------------------------------
    print("=" * 70)
    print("OBJECTNESS PIPELINE  (P1 adjudicates; Phase 2 is diagnostic)")
    print("-" * 70)
    for r in rates:
        o, ru, z = summary["objectness"][r], summary["random"][r], summary["zeroed"][r]
        print(f"  corrupt {r:.1f}: obj {o[0]:.3f}±{o[1]:.3f}  rand {ru[0]:.3f}±{ru[1]:.3f}  "
              f"zero {z[0]:.3f}±{z[1]:.3f}")
    print(f"  P1 VERDICT: {verdict['verdict']}  (passed {verdict['n_pass']}/"
          f"{verdict['n_rates_considered']} rates>=floor, reversal={verdict['reversal']})")
    print("-" * 70)
    print("  Phase 2 diagnostic (two-tailed; storage level is the honest one):")
    for mode in ["objectness", "random", "zeroed"]:
        d = phase2[mode]
        print(f"  [{mode:10s}] STORAGE same={d['storage_same']:.3f} cross={d['storage_cross']:.3f} "
              f"| FINAL same={d['final_same']:.3f} cross={d['final_cross']:.3f}")
    print("=" * 70)

    results = {
        "experiment": "objectness_pipeline",
        "synthetic": False,
        "verdict": verdict["verdict"],
        "passed": verdict["works"],
        "phase1_verdict": verdict,
        "phase1_curves": {m: {str(r): summary[m][r] for r in rates} for m in summary},
        "phase2_diagnostic": phase2,
        "n_store": n_store,
        "encoder": cfg["encoder"]["model"],
        "scope_note": "single-factor (objectness) existence proof; NOT four-system validation.",
    }
    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "objectness_pipeline.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    _plot(summary, rates, verdict, os.path.join(out_dir, "objectness_phase1.png"))
    print(f"[saved] results/objectness_pipeline.json, results/objectness_phase1.png")
    return results


def _phase2_diagnostic(Xte, shape_te, dim, seed_dir, cfg, op) -> dict:
    """Two-level same/cross interference per seed-mode. STORAGE = cosine(r_hat,
    residual(B)); FINAL = cosine(read(A), B). Diagnostic only."""
    import torch
    import torch.nn.functional as F

    shape_np = np.asarray(shape_te)
    same_pairs, cross_pairs = [], []
    npairs = op["n_pairs"]
    for i in range(Xte.shape[0]):
        for j in range(i + 1, Xte.shape[0]):
            if shape_np[i] == shape_np[j] and len(same_pairs) < npairs:
                same_pairs.append((i, j))
            elif shape_np[i] != shape_np[j] and len(cross_pairs) < npairs:
                cross_pairs.append((i, j))
            if len(same_pairs) >= npairs and len(cross_pairs) >= npairs:
                break
        if len(same_pairs) >= npairs and len(cross_pairs) >= npairs:
            break

    def _cos(a, b):
        return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())

    out = {}
    for mode in ["objectness", "random", "zeroed"]:
        model = _build_model(mode, dim, seed_dir, cfg)
        s_same = s_cross = f_same = f_cross = 0.0
        for label, pairs in [("same", same_pairs), ("cross", cross_pairs)]:
            ss = ff = 0.0
            for (i, j) in pairs:
                A, B = Xte[i], Xte[j]
                model.reset_memory()
                model.write(A)
                model.write(B)
                if model.mem.n_stored == 0:
                    continue
                r_hat = model.mem.retrieve(model.ssl.residual(A))
                read = r_hat + model.ssl.project(A)
                ss += _cos(r_hat, model.ssl.residual(B))      # STORAGE level
                ff += _cos(read, B)                            # FINAL level
            denom = max(len(pairs), 1)
            if label == "same":
                s_same, f_same = ss / denom, ff / denom
            else:
                s_cross, f_cross = ss / denom, ff / denom
        out[mode] = {"storage_same": s_same, "storage_cross": s_cross,
                     "final_same": f_same, "final_cross": f_cross}
    return out


def _plot(summary, rates, verdict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, marker in [("objectness", "o"), ("random", "s"), ("zeroed", "^")]:
        m = [summary[mode][r][0] for r in rates]
        e = [summary[mode][r][1] for r in rates]
        ax.errorbar(rates, m, yerr=e, marker=marker, label=mode)
    ax.axvline(0.3, color="gray", ls=":", lw=0.8, label="P1 floor (0.3)")
    ax.set_xlabel("corruption rate")
    ax.set_ylabel("retrieval accuracy (mean cosine)")
    ax.set_title(f"Objectness pipeline — Phase 1 | VERDICT: {verdict['verdict']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
