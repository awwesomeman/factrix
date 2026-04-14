"""factorlib — Modular factor evaluation toolkit.

Public API:
    preprocessing:  preprocess_cs_factor
    gates:          evaluate_factor, PipelineConfig, CROSS_SECTIONAL_GATES
    experiment:     FactorTracker
    validation:     validate_factor_data
"""

from factorlib.preprocessing.pipeline import preprocess_cs_factor
from factorlib.gates.pipeline import evaluate_factor
from factorlib.gates.config import PipelineConfig, MARKET_DEFAULTS
from factorlib.gates.presets import CROSS_SECTIONAL_GATES
from factorlib.experiment import FactorTracker
from factorlib.validation import validate_factor_data

__all__ = [
    "preprocess_cs_factor",
    "evaluate_factor",
    "PipelineConfig",
    "MARKET_DEFAULTS",
    "CROSS_SECTIONAL_GATES",
    "FactorTracker",
    "validate_factor_data",
]
