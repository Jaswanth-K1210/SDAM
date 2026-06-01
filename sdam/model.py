"""S-DAM full model (v2).

Spelke-Seeded Dense Associative Memory:

    write : store the *residual* of an input if its norm clears the learned
            surprise threshold tau; otherwise file it in the provisional
            (hippocampal fast-intake) buffer -- never discard.
    read  : retrieve a residual from the Hopfield memory and reconstruct
            x_hat = retrieved_residual + proj(x).
    consolidate : review the provisional buffer, promote survivors, count
            failures, and grow a new attractor when a pattern repeatedly fails.

v2 additions over the original build: high-inertia seeds, the provisional
buffer, consolidate(), _create_new_attractor(), multi-attractor read, and
energy-distance diagnostics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hopfield import HopfieldLayer
from .seeds import SpelkeSeedLayer


class SDAM(nn.Module):
    """Spelke-Seeded Dense Associative Memory.

    Args:
        input_dim: dimensionality D of inputs / seeds / stored residuals.
        beta: Hopfield inverse-temperature.
        write_threshold: initial value of the learned surprise threshold tau.
        failure_threshold: consolidation failures before a new attractor is born.
        use_high_inertia: passed through to the Spelke seed layer.
    """

    def __init__(
        self,
        input_dim: int,
        beta: float = 1.0,
        write_threshold: float = 0.1,
        failure_threshold: int = 3,
        use_high_inertia: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.failure_threshold = failure_threshold

        self.ssl = SpelkeSeedLayer(seed_dim=input_dim, n_systems=4, use_high_inertia=use_high_inertia)
        self.mem = HopfieldLayer(input_dim, beta=beta, max_iter=3)

        # Learned surprise threshold (Surprise-Gated Write Rule).
        self.tau = nn.Parameter(torch.tensor(float(write_threshold)))

        # Provisional buffer (hippocampal fast intake) and bookkeeping.
        self.provisional: list[torch.Tensor] = []
        self._pids: list[int] = []
        self.failure_counts: dict[int, int] = {}
        self._next_pid = 0

    # ------------------------------------------------------------------ #
    # Write / read
    # ------------------------------------------------------------------ #
    def write(self, x: torch.Tensor) -> dict:
        """Compute residuals and gate them on tau.

        Above-tau residuals are stored in long-term Hopfield memory; below-tau
        residuals go to the provisional buffer. Returns a diagnostic dict.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        residuals = self.ssl.residual(x)                  # (B, D)
        categories = self.ssl.category(x)                 # (B,)
        norms = residuals.norm(dim=1)                     # (B,)
        written = norms > self.tau                        # (B,) bool

        to_store = residuals[written]
        if to_store.shape[0] > 0:
            self.mem.store(to_store)

        for r, keep in zip(residuals, written):
            if not bool(keep):
                self.provisional.append(r.detach().clone())
                self._pids.append(self._next_pid)
                self.failure_counts[self._next_pid] = 0
                self._next_pid += 1

        return {
            "residuals": residuals.detach(),
            "categories": categories.detach(),
            "norms": norms.detach(),
            "written": written.detach(),
            "n_provisional": len(self.provisional),
            "n_stored": self.mem.n_stored,
        }

    def read(self, x: torch.Tensor, multi: bool = False, k: int = 3) -> torch.Tensor:
        """Retrieve residual and reconstruct x_hat = r_hat + proj(x)."""
        query = self.ssl.residual(x)
        if multi:
            r_hat = self.mem.retrieve_multi(query, k=k)
        else:
            r_hat = self.mem.retrieve(query)
        return r_hat + self.ssl.project(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - thin wrapper
        """Forward pass returns the reconstruction read(x)."""
        return self.read(x)

    # ------------------------------------------------------------------ #
    # Consolidation
    # ------------------------------------------------------------------ #
    def consolidate(self) -> dict:
        """Review the provisional buffer.

        Promotes residuals that now clear tau, increments failure counts for
        those that don't, and triggers a new attractor when a pattern's failure
        count reaches ``failure_threshold``. Prints a one-line summary.
        """
        promoted = 0
        new_attractors = 0
        survivors: list[torch.Tensor] = []
        survivor_pids: list[int] = []

        for r, pid in zip(self.provisional, self._pids):
            if r.norm() > self.tau:
                self.mem.store(r)
                promoted += 1
                self.failure_counts.pop(pid, None)
                continue

            self.failure_counts[pid] = self.failure_counts.get(pid, 0) + 1
            if self.failure_counts[pid] >= self.failure_threshold:
                self._create_new_attractor(r)
                new_attractors += 1
                self.failure_counts.pop(pid, None)
            else:
                survivors.append(r)
                survivor_pids.append(pid)

        self.provisional = survivors
        self._pids = survivor_pids

        print(
            f"[consolidate] promoted={promoted} "
            f"new_attractors={new_attractors} remaining_provisional={len(self.provisional)}"
        )
        return {
            "promoted": promoted,
            "new_attractors": new_attractors,
            "remaining_provisional": len(self.provisional),
        }

    def _create_new_attractor(self, r: torch.Tensor) -> None:
        """Normalize ``r`` and append it as a new seed row in the SSL."""
        r = r.detach().flatten()
        norm = r.norm()
        if norm > 0:
            r = r / norm
        new_seeds = torch.cat([self.ssl.seeds.detach(), r.unsqueeze(0)], dim=0)
        if self.ssl.use_high_inertia:
            self.ssl.seeds = nn.Parameter(new_seeds)
        else:
            # Re-register the buffer with the grown tensor.
            del self.ssl._buffers["seeds"]
            self.ssl.register_buffer("seeds", new_seeds)
        self.ssl.n_systems = new_seeds.shape[0]
        print(f"[_create_new_attractor] total seeds = {self.ssl.n_systems}")

    # ------------------------------------------------------------------ #
    # Lifecycle / diagnostics
    # ------------------------------------------------------------------ #
    def reset_memory(self) -> None:
        """Clear long-term memory, the provisional buffer, and failure counts."""
        self.mem.reset()
        self.provisional = []
        self._pids = []
        self.failure_counts = {}
        self._next_pid = 0

    def diagnostics(self) -> dict:
        """Snapshot of memory state for experiments and tests."""
        return {
            "n_stored": self.mem.n_stored,
            "n_provisional": len(self.provisional),
            "n_seeds": self.ssl.n_systems,
            "tau": float(self.tau.detach()),
            "seeds_orthogonal": self.ssl.verify_orthogonality(),
        }

    def configure_optimizer(self, seed_lr: float = 1e-6, default_lr: float = 1e-3):
        """Two-group optimizer: high-inertia seeds at ``seed_lr``, rest at ``default_lr``."""
        seed_params = [self.ssl.seeds] if isinstance(self.ssl.seeds, nn.Parameter) else []
        seed_ids = {id(p) for p in seed_params}
        other_params = [p for p in self.parameters() if id(p) not in seed_ids]
        groups = [{"params": other_params, "lr": default_lr}]
        if seed_params:
            groups.append({"params": seed_params, "lr": seed_lr})
        return torch.optim.Adam(groups)
