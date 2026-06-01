"""S-DAM: Spelke-Seeded Dense Associative Memory."""

from .hopfield import HopfieldLayer
from .model import SDAM
from .seeds import SPELKE_SYSTEMS, SpelkeSeedLayer

__all__ = ["SDAM", "HopfieldLayer", "SpelkeSeedLayer", "SPELKE_SYSTEMS"]
__version__ = "2.0.0"
