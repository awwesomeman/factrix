"""
Layer 3: Scoring Core — Registry, metrics, and FactorScorer.
Refactored into a package; public API unchanged.
"""

from factorlib.scoring.registry import METRIC_REGISTRY, register
from factorlib.scoring.config import SCORING_CONFIG

# Import metric modules to trigger @register decorators
import factorlib.scoring.selection  # noqa: F401
import factorlib.scoring.timing     # noqa: F401

from factorlib.scoring.scorer import FactorScorer

__all__ = ["FactorScorer", "SCORING_CONFIG", "METRIC_REGISTRY"]
