"""Standalone helpers shared across experiments and tests."""

from __future__ import annotations

import random

import numpy as np
import torch

from scipy.stats import mannwhitneyu


def set_all_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for deterministic runs.

    Call at the top of every experiment script. Also forces cudnn into
    deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarities between rows of ``a`` (M,D) and ``b`` (N,D).

    Returns an (M, N) tensor.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_n = torch.nn.functional.normalize(a, dim=1)
    b_n = torch.nn.functional.normalize(b, dim=1)
    return a_n @ b_n.t()


def corruption_mask(x: torch.Tensor, rate: float) -> torch.Tensor:
    """Zero-mask a ``rate`` fraction of dimensions per pattern in the batch.

    The masked dimensions are chosen independently for each row. ``x`` may be
    (D,) or (B, D); the returned tensor matches the input shape.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"corruption rate must be in [0, 1]; got {rate}.")
    squeeze = x.dim() == 1
    xb = x.unsqueeze(0) if squeeze else x
    b, d = xb.shape
    n_mask = int(round(rate * d))
    out = xb.clone()
    if n_mask > 0:
        for i in range(b):
            idx = torch.randperm(d, device=x.device)[:n_mask]
            out[i, idx] = 0.0
    return out.squeeze(0) if squeeze else out


def capacity_search(model, patterns: torch.Tensor, accuracy_threshold: float = 0.90, step: int = 10) -> int:
    """Largest number of patterns storable before retrieval accuracy drops.

    Probes increasing pattern counts (in steps of ``step``) and, for each,
    measures retrieval accuracy under 30% corruption. Resets the model's memory
    before each probe. Returns the largest count whose accuracy stays at or
    above ``accuracy_threshold``.
    """
    n_total = patterns.shape[0]
    capacity = 0
    n = step
    while n <= n_total:
        model.reset_memory()
        subset = patterns[:n]
        for p in subset:
            model.write(p)
        if model.mem.n_stored == 0:
            break
        corrupted = corruption_mask(subset, 0.30)
        recovered = model.read(corrupted)
        sims = cosine_similarity_matrix(recovered, subset).diagonal().detach()
        accuracy = float(sims.mean())
        if accuracy >= accuracy_threshold:
            capacity = n
            n += step
        else:
            break
    model.reset_memory()
    return capacity


def mann_whitney_u(a, b):
    """One-sided Mann-Whitney U test (H1: distribution of ``a`` < ``b``).

    Returns (statistic, p_value).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p_value = mannwhitneyu(a, b, alternative="less")
    return float(stat), float(p_value)
