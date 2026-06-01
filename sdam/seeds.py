"""Spelke Seed Layer (SSL).

Holds four frozen-ish cognitive priors drawn from Spelke core-knowledge theory
(Spelke & Kinzler 2007): Objectness, Agentness, Numerosity, Geometry.

All knowledge in S-DAM is stored as a geometric *residual* relative to the
subspace spanned by these seeds. This module provides the projection /
residual / category machinery and the orthogonality guarantees the rest of the
system relies on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Canonical mapping from Spelke system name -> seed row index.
SPELKE_SYSTEMS = {
    "OBJECTNESS": 0,
    "AGENTNESS": 1,
    "NUMEROSITY": 2,
    "GEOMETRY": 3,
}


class SpelkeSeedLayer(nn.Module):
    """Orthonormal seed subspace with projection / residual operations.

    Args:
        seed_dim: dimensionality D of each seed vector.
        n_systems: number of seed rows (default 4 Spelke systems).
        use_high_inertia: if True, seeds are an ``nn.Parameter`` (trained at a
            very low LR by the caller's optimizer -- "high inertia"). If False,
            seeds are a frozen ``register_buffer``. Default True.
    """

    def __init__(self, seed_dim: int = 128, n_systems: int = 4, use_high_inertia: bool = True):
        super().__init__()
        if n_systems > seed_dim:
            raise ValueError(
                f"n_systems ({n_systems}) cannot exceed seed_dim ({seed_dim}); "
                "orthonormal seeds require n_systems <= seed_dim."
            )
        self.seed_dim = seed_dim
        self.n_systems = n_systems
        self.use_high_inertia = use_high_inertia

        # Orthonormal initialization: rows of `seeds` are mutually orthonormal.
        seeds = torch.empty(n_systems, seed_dim)
        nn.init.orthogonal_(seeds)

        if use_high_inertia:
            self.seeds = nn.Parameter(seeds)
        else:
            self.register_buffer("seeds", seeds)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_batched(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return (batched_x, was_unbatched). Accepts (D,) or (B, D)."""
        if x.dim() == 1:
            return x.unsqueeze(0), True
        if x.dim() == 2:
            return x, False
        raise ValueError(
            f"SpelkeSeedLayer expects input of shape (D,) or (B, D); got shape {tuple(x.shape)}."
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection of ``x`` onto the seed subspace.

        Because seeds are orthonormal, proj(x) = (x @ Sᵀ) @ S.
        Handles both (D,) and (B, D) inputs and returns the matching shape.
        """
        xb, squeeze = self._ensure_batched(x)
        coeffs = xb @ self.seeds.t()          # (B, n_systems)
        proj = coeffs @ self.seeds            # (B, D)
        return proj.squeeze(0) if squeeze else proj

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return r = x - project(x). The part of x not explained by the priors."""
        return x - self.project(x)

    def category(self, x: torch.Tensor) -> torch.Tensor:
        """Return index of the dominant seed for each input, shape (B,).

        Dominance = largest dot product with a seed.
        """
        xb, _ = self._ensure_batched(x)
        dots = xb @ self.seeds.t()            # (B, n_systems)
        return dots.argmax(dim=1)

    def verify_orthogonality(self, atol: float = 1e-4) -> bool:
        """True if seed rows are (near) orthonormal: S Sᵀ ≈ I within atol."""
        gram = self.seeds @ self.seeds.t()
        identity = torch.eye(self.seeds.shape[0], device=gram.device, dtype=gram.dtype)
        return bool(torch.allclose(gram, identity, atol=atol))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - thin wrapper
        """Forward pass returns the residual of x."""
        return self.residual(x)
