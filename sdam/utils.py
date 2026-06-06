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


def capacity_search(
    model,
    patterns: torch.Tensor,
    accuracy_threshold: float = 0.75,
    step: int = 10,
    corruption_rate: float = 0.30,
) -> int:
    """Largest number of patterns storable with retrieval accuracy >= accuracy_threshold.

    Fix over original: if nothing is written to long-term memory (all residuals
    below tau go to provisional buffer), we call consolidate() to force promotion
    before measuring retrieval. This prevents the all-zeros failure on low-norm residuals.
    """
    n_total = patterns.shape[0]
    best_capacity = 0
    n = step

    while n <= n_total:
        model.reset_memory()
        subset = patterns[:n]

        # Write all patterns
        for p in subset:
            model.write(p)

        # Force consolidation if long-term memory is still empty
        # (happens when write_threshold > residual norms)
        if model.mem.n_stored == 0 and len(model.provisional) > 0:
            # Temporarily lower tau to force promotion
            original_tau = float(model.tau.detach())
            with torch.no_grad():
                model.tau.fill_(0.0)
            model.consolidate()
            with torch.no_grad():
                model.tau.fill_(original_tau)

        # If still empty after consolidation, nothing can be stored
        if model.mem.n_stored == 0:
            n += step
            continue

        corrupted = corruption_mask(subset, corruption_rate)
        recovered = model.read(corrupted)
        sims = cosine_similarity_matrix(recovered.detach(), subset).diagonal()
        accuracy = float(sims.mean())

        if accuracy >= accuracy_threshold:
            best_capacity = n
            n += step
        else:
            break

    model.reset_memory()
    return best_capacity


def mann_whitney_u(a, b):
    """One-sided Mann-Whitney U test (H1: distribution of ``a`` < ``b``).

    Returns (statistic, p_value).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p_value = mannwhitneyu(a, b, alternative="less")
    return float(stat), float(p_value)
