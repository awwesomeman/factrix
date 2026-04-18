"""factorlib — Modular factor evaluation toolkit (profile-era API).

Single-factor usage::

    import factorlib as fl

    profile = fl.evaluate(df, "Mom_20D", factor_type="cross_sectional")
    print(profile.verdict(), profile.canonical_p)
    for d in profile.diagnose():
        print(d.severity, d.code, d.message)

Batch + BHY multiple-testing::

    import polars as pl
    import factorlib as fl

    profiles = fl.evaluate_batch(candidates, factor_type="cross_sectional")

    top = (
        profiles
        .multiple_testing_correct(p_source="canonical_p", fdr=0.05)
        .filter(pl.col("bhy_significant"))
        .rank_by("ic_ir")
        .top(10)
    )

Schema reflection::

    fl.describe_factor_types()
    fl.describe_profile("event_signal")
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
# Profile-era API
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
from factorlib.evaluation.diagnostics import (
    Rule,
    clear_custom_rules,
    register_rule,
)
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


__all__ = [
    # Top-level API
    "evaluate", "evaluate_batch", "list_factor_types",
    "ProfileSet", "redundancy_matrix",
    "FactorProfile",
    "CrossSectionalProfile", "EventProfile",
    "MacroPanelProfile", "MacroCommonProfile",
    "bhy_adjust", "bhy_adjusted_p",
    "Diagnostic", "PValue", "Verdict",
    "Rule", "register_rule", "clear_custom_rules",
    # Shared core
    "adapt", "preprocess", "preprocess_cs_factor", "build_artifacts",
    "CrossSectionalConfig", "EventConfig",
    "MacroPanelConfig", "MacroCommonConfig",
    "FACTOR_TYPES", "describe_factor_types", "describe_profile",
    "MARKET_DEFAULTS",
    "FactorType", "MetricOutput", "Artifacts",
    "split_by_group",
    "validate_factor_data",
]
