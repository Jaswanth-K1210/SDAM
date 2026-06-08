"""S-DAM: Spelke-Seeded Dense Associative Memory."""

__all__ = ["SDAM", "HopfieldLayer", "SpelkeSeedLayer", "SPELKE_SYSTEMS"]
__version__ = "2.0.0"

# model/hopfield/seeds require torch. sdam.factored's pure factoring math
# (factor_pattern, augment, reconstruct, variance_matched_w, crossover_point) is
# torch-free and must be unit-testable without torch, so guard the torch-backed
# exports. The three core modules themselves are NOT edited.
try:  # pragma: no cover - environment-dependent
    from .hopfield import HopfieldLayer
    from .model import SDAM
    from .seeds import SPELKE_SYSTEMS, SpelkeSeedLayer
except ModuleNotFoundError as _exc:
    if _exc.name != "torch":
        raise
