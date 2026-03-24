"""
Layer 3: Scoring Core — Registry, metrics, and FactorScorer.
"""

from factorlib.scoring.registry import METRIC_REGISTRY, register
from factorlib.scoring.config import FACTOR_CONFIGS, DIMENSIONS
from factorlib.scoring._utils import MetricResult

# Import metric modules to trigger @register decorators
import factorlib.scoring.selection  # noqa: F401
import factorlib.scoring.timing     # noqa: F401

from factorlib.scoring.scorer import FactorScorer

__all__ = ["FactorScorer", "FACTOR_CONFIGS", "DIMENSIONS", "METRIC_REGISTRY", "MetricResult"]
