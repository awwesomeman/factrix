"""factorlib — Modular factor evaluation toolkit.

Quick start::

    import factorlib as fl
    result = fl.quick_check(df, "Mom_20D")
    print(result)

Full pipeline::

    prepared = fl.preprocess(df, config=fl.CrossSectionalConfig())
    result = fl.evaluate(prepared, "Mom_20D")
"""

from factorlib.adapt import adapt
from factorlib.preprocess.pipeline import preprocess, preprocess_cs_factor
from factorlib.evaluation.pipeline import evaluate, build_artifacts
from factorlib.config import (
    CrossSectionalConfig,
    EventConfig,
    MacroPanelConfig,
    MacroCommonConfig,
    MARKET_DEFAULTS,
)
from factorlib.evaluation.presets import CROSS_SECTIONAL_GATES
from factorlib.validation import validate_factor_data
from factorlib._api import (
    quick_check,
    batch_evaluate,
    compare,
    split_by_group,
    FACTOR_TYPES,
    describe_factor_types,
)

__all__ = [
    # Core workflow
    "adapt", "preprocess", "evaluate", "quick_check",
    # Batch & comparison
    "batch_evaluate", "compare", "split_by_group",
    # Configuration
    "CrossSectionalConfig", "EventConfig",
    "MacroPanelConfig", "MacroCommonConfig",
    "FACTOR_TYPES", "describe_factor_types",
    "MARKET_DEFAULTS",
    # Artifacts (advanced)
    "build_artifacts",
    # Presets
    "CROSS_SECTIONAL_GATES",
    # Validation
    "validate_factor_data",
    # Legacy (will be removed)
    "preprocess_cs_factor",
]
