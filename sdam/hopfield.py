"""Orthogonal Residual Slots (ORS) -- a Modern Hopfield (DAM) layer.

Implements the continuous Modern Hopfield update rule from Ramsauer et al.
(2021), "Hopfield Networks is All You Need". Stores *residuals* (not raw
inputs) and retrieves them by iterated softmax attention over stored patterns.

Update rule (single step):   xi <- Xᵀ softmax(beta · X xi)
Energy:  E(xi) = -beta⁻¹ logsumexp(beta · X xi) + ½ xiᵀxi + beta⁻¹ log N
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HopfieldLayer(nn.Module):
    """Modern Hopfield associative memory storing detached patterns.

    Args:
        input_dim: dimensionality D of stored patterns / queries.
        beta: inverse-temperature of the softmax retrieval (higher = sharper).
        max_iter: number of iterative update steps performed by ``retrieve``.
    """

    def __init__(self, input_dim: int, beta: float = 1.0, max_iter: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.beta = beta
        self.max_iter = max_iter
        # Stored patterns kept as a plain (N, D) tensor (detached); None == empty.
        self.stored_patterns: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    @property
    def n_stored(self) -> int:
        """Number of stored patterns (0 if none)."""
        return 0 if self.stored_patterns is None else self.stored_patterns.shape[0]

    def store(self, patterns: torch.Tensor) -> None:
        """Append ``patterns`` (shape (D,) or (N, D)) to memory, detached."""
        if patterns.dim() == 1:
            patterns = patterns.unsqueeze(0)
        if patterns.dim() != 2 or patterns.shape[1] != self.input_dim:
            raise ValueError(
                f"store expected patterns of shape (N, {self.input_dim}); "
                f"got {tuple(patterns.shape)}."
            )
        patterns = patterns.detach()
        if self.stored_patterns is None:
            self.stored_patterns = patterns.clone()
        else:
            self.stored_patterns = torch.cat([self.stored_patterns, patterns], dim=0)

    def reset(self) -> None:
        """Clear all stored patterns."""
        self.stored_patterns = None

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _require_patterns(self, method: str) -> torch.Tensor:
        if self.stored_patterns is None:
            raise RuntimeError(
                f"HopfieldLayer.{method}() called before any patterns were stored. "
                "Call store(patterns) first (and check model.write() actually wrote "
                "something -- patterns below tau go to the provisional buffer)."
            )
        return self.stored_patterns

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Iterative Hopfield retrieval for ``max_iter`` steps.

        Accepts (D,) or (B, D) and returns the matching shape.
        """
        X = self._require_patterns("retrieve")
        squeeze = query.dim() == 1
        xi = query.unsqueeze(0) if squeeze else query  # (B, D)
        for _ in range(self.max_iter):
            scores = self.beta * (xi @ X.t())          # (B, N)
            weights = F.softmax(scores, dim=1)         # (B, N)
            xi = weights @ X                           # (B, D)
        return xi.squeeze(0) if squeeze else xi

    def retrieve_multi(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Top-k weighted-combination retrieval (single shot, not iterated).

        Returns a beta-softmax weighted blend of the k nearest stored patterns.
        ``k`` is capped at the number of stored patterns.
        """
        X = self._require_patterns("retrieve_multi")
        k = max(1, min(k, X.shape[0]))
        squeeze = query.dim() == 1
        xi = query.unsqueeze(0) if squeeze else query     # (B, D)
        scores = self.beta * (xi @ X.t())                 # (B, N)
        topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)  # (B, k)
        weights = F.softmax(topk_scores, dim=1)           # (B, k)
        topk_patterns = X[topk_idx]                       # (B, k, D)
        out = torch.einsum("bk,bkd->bd", weights, topk_patterns)
        return out.squeeze(0) if squeeze else out

    # ------------------------------------------------------------------ #
    # Energy
    # ------------------------------------------------------------------ #
    def energy(self, xi: torch.Tensor) -> torch.Tensor:
        """Modern Hopfield energy of state ``xi``.

        E(xi) = -beta⁻¹ logsumexp(beta · X xi) + ½ xiᵀxi + beta⁻¹ log N

        Accepts (D,) or (B, D) and returns a scalar or (B,) tensor.
        """
        X = self._require_patterns("energy")
        squeeze = xi.dim() == 1
        x = xi.unsqueeze(0) if squeeze else xi            # (B, D)
        n = X.shape[0]
        scores = self.beta * (x @ X.t())                  # (B, N)
        lse = torch.logsumexp(scores, dim=1)              # (B,)
        quad = 0.5 * (x * x).sum(dim=1)                   # (B,)
        log_n = torch.log(torch.tensor(float(n), device=x.device, dtype=x.dtype))
        e = -lse / self.beta + quad + log_n / self.beta
        return e.squeeze(0) if squeeze else e

    def energy_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Energy gap E(x) - E(attractor(x)), where attractor = retrieve(x).

        Used in the detection stage and as an alternative write-gate signal.
        Returns +inf (matching the input batch shape) if no patterns stored.
        """
        if self.stored_patterns is None:
            squeeze = x.dim() == 1
            shape = () if squeeze else (x.shape[0],)
            return torch.full(shape, float("inf"), device=x.device, dtype=x.dtype)
        attractor = self.retrieve(x)
        return self.energy(x) - self.energy(attractor)
