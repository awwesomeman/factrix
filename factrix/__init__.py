"""factrix — Modular factor evaluation toolkit (profile-era API).

Single-factor usage::

    import factrix as fl

    profile = fl.evaluate(df, "Mom_20D", factor_type="cross_sectional")
    print(profile.verdict(), profile.canonical_p)
    # verdict() returns "PASS" | "PASS_WITH_WARNINGS" | "FAILED".
    # PASS_WITH_WARNINGS fires when a warn-severity diagnostic names a
    # whitelisted alternative p_source the user has not adopted.
    for d in profile.diagnose():
        print(d.severity, d.code, d.message, d.recommended_p_source)

Batch + BHY multiple-testing::

    import polars as pl
    import factrix as fl

    profiles = fl.evaluate_batch(candidates, factor_type="cross_sectional")

    top = (
        profiles
        .multiple_testing_correct(p_source="canonical_p", fdr=0.05)
        .filter(pl.col("bhy_significant"))
        .rank_by("ic_ir")
        .top(10)
    )

    # Zoo-scale diagnostic triage
    profiles.diagnose_all()              # tidy DataFrame: (factor_name,
                                         # severity, code, message,
                                         # recommended_p_source)
    profiles.with_canonical("ic_nw_p")   # rebind canonical for BHY

Schema reflection::

    fl.describe_factor_types()
    fl.describe_profile("event_signal")
"""

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

from factrix.adapt import adapt
from factrix.preprocess.pipeline import preprocess, preprocess_cs_factor
from factrix.evaluation.pipeline import build_artifacts
from factrix.config import (
    CrossSectionalConfig,
    EventConfig,
    MacroPanelConfig,
    MacroCommonConfig,
    OrthoConfig,
    MARKET_DEFAULTS,
)
from factrix._types import (
    Diagnostic,
    FactorType,
    MetricOutput,
    PValue,
    Verdict,
)
from factrix.validation import validate_factor_data

# ---------------------------------------------------------------------------
# Profile-era API
# ---------------------------------------------------------------------------

from factrix.evaluation._protocol import Artifacts
from factrix.evaluation.profiles import (
    FactorProfile,  # Protocol
    CrossSectionalProfile,
    EventProfile,
    MacroPanelProfile,
    MacroCommonProfile,
)
from factrix.evaluation.profile_set import ProfileSet
from factrix.evaluation.diagnostics import (
    Rule,
    clear_custom_rules,
    register_rule,
)
from factrix.reporting import describe_profile_values
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p
# Factor subclasses are re-exported for isinstance() checks and type
# hints. The ``Factor`` base class is intentionally NOT re-exported —
# direct instantiation is an advanced path and the factory ``fl.factor()``
# is the canonical entry. If you need ``Factor`` as a type hint, import
# it from ``factrix.factor`` directly.
from factrix.factor import (
    CrossSectionalFactor,
    EventFactor,
    MacroPanelFactor,
    MacroCommonFactor,
)
from factrix._api import (
    evaluate,
    evaluate_batch,
    factor,
    list_factor_types,
    redundancy_matrix,
    split_by_group,
    FACTOR_TYPES,
    describe_factor_types,
    describe_profile,
)
from factrix import datasets

__version__ = "0.4.0"

__all__ = [
    # Top-level API
    "evaluate", "evaluate_batch", "factor", "list_factor_types",
    "CrossSectionalFactor",
    "EventFactor", "MacroPanelFactor", "MacroCommonFactor",
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
    "OrthoConfig",
    "FACTOR_TYPES", "describe_factor_types", "describe_profile",
    "describe_profile_values",
    "MARKET_DEFAULTS",
    "FactorType", "MetricOutput", "Artifacts",
    "split_by_group",
    "validate_factor_data",
    "datasets",
]
