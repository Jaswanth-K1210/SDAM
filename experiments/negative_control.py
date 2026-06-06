"""Negative control — domain specificity of Spelke advantage.

Runs Phase-2-style interference measurement on SST-2 sentiment features
(positive vs negative sentences). Spelke seeds are structured for
physical/spatial/numerical/agentive domains. They should confer NO special
advantage over random seeds on purely linguistic sentiment.

If spelke.gap ≈ random.gap on SST-2 BUT spelke.gap >> random.gap on CIFAR-10,
that proves the benefit is DOMAIN-SPECIFIC, not just 'any fixed scaffold helps'.

Pass criterion (negative control): |spelke_gap - random_gap| < 0.05
i.e. no statistically meaningful difference on SST-2.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments._common import load_config, ensure_results_dir, dump_json
from sdam.model import SDAM
from sdam.utils import set_all_seeds, mann_whitney_u

SENT_CACHE = "data/sst2_feats.pt"
SENT_DIM = 384  # all-MiniLM-L6-v2 output dim


def _extract_sst2_features(cache_path: str) -> dict:
    """Download SST-2 and extract sentence embeddings with MiniLM."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("glue", "sst2", split="train[:3000]")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    enc_model = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    ).to(device).eval()

    def mean_pool(out, mask):
        tok = out.last_hidden_state
        m = mask.unsqueeze(-1).expand(tok.size()).float()
        return F.normalize((tok * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)

    pos_feats, neg_feats = [], []
    batch_size = 128
    texts, labels = ds["sentence"], ds["label"]

    for i in range(0, len(texts), batch_size):
        bt = texts[i : i + batch_size]
        bl = labels[i : i + batch_size]
        enc = tokenizer(bt, padding=True, truncation=True,
                        return_tensors="pt", max_length=64)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = enc_model(**enc)
        embs = mean_pool(out, enc["attention_mask"]).cpu()
        for emb, lbl in zip(embs, bl):
            (pos_feats if lbl == 1 else neg_feats).append(emb)

    feats = {
        "positive": torch.stack(pos_feats),
        "negative": torch.stack(neg_feats),
    }
    torch.save(feats, cache_path)
    print(f"Saved SST-2 features to {cache_path}")
    return feats


def _make_model(mode: str) -> SDAM:
    m = SDAM(input_dim=SENT_DIM, beta=16.0, write_threshold=0.05,
             use_high_inertia=(mode == "spelke"))
    if mode == "random":
        rand = torch.empty(4, SENT_DIM)
        nn.init.orthogonal_(rand)
        m.ssl.seeds = nn.Parameter(rand)
    elif mode == "none":
        m.ssl.seeds = nn.Parameter(torch.zeros(4, SENT_DIM))
    return m


def _interference(model, a: torch.Tensor, b: torch.Tensor) -> float:
    model.reset_memory()
    model.write(a.unsqueeze(0))
    model.write(b.unsqueeze(0))
    if model.mem.n_stored == 0:
        return 0.0
    retrieved = model.read(a.unsqueeze(0))
    return float(F.cosine_similarity(retrieved, b.unsqueeze(0)).item())


def run() -> dict:
    set_all_seeds(42)
    cfg = load_config()
    out_dir = ensure_results_dir(cfg)

    # Load or extract SST-2 features
    if os.path.exists(SENT_CACHE):
        print(f"Loading SST-2 features from {SENT_CACHE}")
        sst2_feats = torch.load(SENT_CACHE, map_location="cpu")
    else:
        print("Extracting SST-2 sentence embeddings (requires transformers + datasets)...")
        sst2_feats = _extract_sst2_features(SENT_CACHE)

    pos = sst2_feats["positive"]
    neg = sst2_feats["negative"]
    N = min(150, pos.shape[0] - 1, neg.shape[0])

    nc_results = {}
    for mode in ["spelke", "random", "none"]:
        m = _make_model(mode)
        same_scores, cross_scores = [], []

        for i in range(N):
            # Same-sentiment pair: pos[i] vs pos[i+1]
            same_scores.append(_interference(m, pos[i], pos[i + 1]))
            # Cross-sentiment pair: pos[i] vs neg[i]
            cross_scores.append(_interference(m, pos[i], neg[i]))

        sm = float(np.mean(same_scores))
        cm = float(np.mean(cross_scores))
        gap = sm - cm
        _, p = mann_whitney_u(cross_scores, same_scores)

        nc_results[mode] = dict(
            same_mean=sm, cross_mean=cm, gap=gap, p_value=float(p)
        )
        print(f"  [{mode:8s}] same={sm:.4f}  cross={cm:.4f}  gap={gap:.4f}  p={p:.2e}")

    # PASS if Spelke gap is NOT significantly larger than random gap
    spelke_gap = nc_results["spelke"]["gap"]
    random_gap = nc_results["random"]["gap"]
    gap_diff = abs(spelke_gap - random_gap)
    passed = gap_diff < 0.05

    print("=" * 64)
    print("NEGATIVE CONTROL — SST-2 sentiment (domain specificity)")
    print(f"  spelke_gap   : {spelke_gap:.4f}")
    print(f"  random_gap   : {random_gap:.4f}")
    print(f"  |difference| : {gap_diff:.4f} (require < 0.05 to PASS)")
    print(f"  VERDICT      : {'PASSED' if passed else 'FAILED'}")
    print("  (PASS = Spelke seeds give no special advantage on sentiment)")
    print("=" * 64)

    # Comparison bar chart
    modes = ["spelke", "random", "none"]
    x = np.arange(len(modes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    same_vals  = [nc_results[m]["same_mean"]  for m in modes]
    cross_vals = [nc_results[m]["cross_mean"] for m in modes]
    ax.bar(x - width / 2, same_vals,  width, label="same-sentiment",  color="steelblue")
    ax.bar(x + width / 2, cross_vals, width, label="cross-sentiment", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Mean interference cosine")
    ax.set_title("Negative control: SST-2 sentiment\n(Spelke advantage should vanish here)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "negative_control_sst2.png"), dpi=120)
    plt.close(fig)

    results = dict(
        experiment="negative_control_sst2",
        synthetic=False,
        sst2_results=nc_results,
        spelke_gap=spelke_gap,
        random_gap=random_gap,
        gap_difference=gap_diff,
        passed=passed,
    )
    dump_json(results, os.path.join(out_dir, "negative_control_sst2.json"))
    return results

if __name__ == "__main__":
    run()
