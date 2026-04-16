"""factorlib — Modular factor evaluation toolkit.

Public API:
    adapt:          adapt(df, date=..., asset_id=..., price=...)
    preprocess:     preprocess_cs_factor
    evaluate:       evaluate_factor, CrossSectionalConfig, CROSS_SECTIONAL_GATES
    validation:     validate_factor_data
"""

from factorlib.adapt import adapt
from factorlib.preprocess.pipeline import preprocess_cs_factor
from factorlib.evaluation.pipeline import evaluate_factor
from factorlib.config import (
    CrossSectionalConfig,
    EventConfig,
    MacroPanelConfig,
    MacroCommonConfig,
    MARKET_DEFAULTS,
)
from factorlib.evaluation.presets import CROSS_SECTIONAL_GATES
from factorlib.validation import validate_factor_data

__all__ = [
    "adapt",
    "preprocess_cs_factor",
    "evaluate_factor",
    "CrossSectionalConfig",
    "EventConfig",
    "MacroPanelConfig",
    "MacroCommonConfig",
    "MARKET_DEFAULTS",
    "CROSS_SECTIONAL_GATES",
    "validate_factor_data",
]
