"""factorlib — Modular factor evaluation toolkit.

Profile-era (new, Phase A):

    import factorlib as fl

    profile  = fl.evaluate(df, "Mom_20D", factor_type="cross_sectional")
    profiles = fl.evaluate_batch(candidates, factor_type="cross_sectional")

    top = (
        profiles
        .multiple_testing_correct(p_source="canonical_p", fdr=0.05)
        .filter(pl.col("bhy_significant"))
        .rank_by("ic_ir")
        .top(10)
    )

Gate-era (legacy, kept during Phase A; removed in Phase B):

    fl.quick_check / fl.batch_evaluate / fl.compare
    fl.EvaluationResult, fl.GateResult, fl.CROSS_SECTIONAL_GATES, ...
"""

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

from factorlib.adapt import adapt
from factorlib.preprocess.pipeline import preprocess, preprocess_cs_factor
from factorlib.evaluation.pipeline import build_artifacts
from factorlib.config import (
    CrossSectionalConfig,
    EventConfig,
    MacroPanelConfig,
    MacroCommonConfig,
    MARKET_DEFAULTS,
)
from factorlib._types import (
    Diagnostic,
    FactorType,
    MetricOutput,
    PValue,
    Verdict,
)
from factorlib.validation import validate_factor_data

# ---------------------------------------------------------------------------
# Profile-era API (new)
# ---------------------------------------------------------------------------

from factorlib.evaluation._protocol import Artifacts
from factorlib.evaluation.profiles import (
    FactorProfile,  # Protocol
    CrossSectionalProfile,
    EventProfile,
    MacroPanelProfile,
    MacroCommonProfile,
)
from factorlib.evaluation.profile_set import ProfileSet
from factorlib.stats.multiple_testing import bhy_adjust, bhy_adjusted_p
from factorlib._api import (
    evaluate,
    evaluate_batch,
    list_factor_types,
    redundancy_matrix,
    split_by_group,
    FACTOR_TYPES,
    describe_factor_types,
    describe_profile,
)

# ---------------------------------------------------------------------------
# Gate-era API (legacy; kept for dual-export during Phase A)
# ---------------------------------------------------------------------------

from factorlib.evaluation._protocol import (
    EvaluationResult,
    GateResult,
)
from factorlib.evaluation._protocol import FactorProfile as _LegacyFactorProfile  # noqa: F401
from factorlib.evaluation.presets import (
    CROSS_SECTIONAL_GATES,
    MACRO_PANEL_GATES,
    MACRO_COMMON_GATES,
)
from factorlib._api import (
    quick_check,
    batch_evaluate,
    compare,
)


__all__ = [
    # Profile-era (new)
    "evaluate", "evaluate_batch", "list_factor_types",
    "ProfileSet", "redundancy_matrix",
    "FactorProfile",
    "CrossSectionalProfile", "EventProfile",
    "MacroPanelProfile", "MacroCommonProfile",
    "bhy_adjust", "bhy_adjusted_p",
    "Diagnostic", "PValue", "Verdict",
    # Shared core
    "adapt", "preprocess", "build_artifacts",
    "CrossSectionalConfig", "EventConfig",
    "MacroPanelConfig", "MacroCommonConfig",
    "FACTOR_TYPES", "describe_factor_types", "describe_profile",
    "MARKET_DEFAULTS",
    "FactorType", "MetricOutput", "Artifacts",
    "split_by_group",
    "validate_factor_data",
    # Gate-era (legacy; drops in Phase B)
    "quick_check", "batch_evaluate", "compare",
    "EvaluationResult", "GateResult",
    "CROSS_SECTIONAL_GATES", "MACRO_PANEL_GATES", "MACRO_COMMON_GATES",
    "preprocess_cs_factor",
]
