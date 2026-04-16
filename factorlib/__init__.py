"""factorlib — Modular factor evaluation toolkit.

Public API:
    adapt:          adapt(df, date=..., asset_id=..., price=...)
    preprocess:     preprocess_cs_factor
    gates:          evaluate_factor, PipelineConfig, CROSS_SECTIONAL_GATES
    experiment:     FactorTracker
    validation:     validate_factor_data
"""

from factorlib.adapt import adapt
from factorlib.preprocess.pipeline import preprocess_cs_factor
from factorlib.evaluation.pipeline import evaluate_factor
from factorlib.config import PipelineConfig, MARKET_DEFAULTS
from factorlib.evaluation.presets import CROSS_SECTIONAL_GATES
from factorlib.integrations.mlflow import FactorTracker
from factorlib.validation import validate_factor_data

__all__ = [
    "adapt",
    "preprocess_cs_factor",
    "evaluate_factor",
    "PipelineConfig",
    "MARKET_DEFAULTS",
    "CROSS_SECTIONAL_GATES",
    "FactorTracker",
    "validate_factor_data",
]
