"""factrix — Single-factor evaluation toolkit (v0.5).

Three orthogonal user-facing axes — ``FactorScope``, ``Signal``,
``Metric`` — plus an evaluate-time-derived ``Mode`` define the analysis
cell. Construct a config via the four type-safe factories on
``AnalysisConfig``, dispatch via ``evaluate()``, inspect via the
returned ``FactorProfile``, and aggregate across factors with
``multi_factor.bhy`` for FDR-corrected screening.

Single-factor::

    import factrix as fl

    cfg = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC)
    profile = fl.evaluate(panel, cfg)
    print(profile.verdict(), profile.primary_p)
    print(profile.diagnose())

Batch + BHY::

    profiles = [fl.evaluate(panel, cfg) for cfg in candidate_configs]
    survivors = fl.multi_factor.bhy(profiles, threshold=0.05)

Schema reflection::

    print(fl.describe_analysis_modes())
    print(fl.suggest_config(panel))
"""

from factrix import datasets, multi_factor
from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import InfoCode, StatCode, Verdict, WarningCode
from factrix._describe import (
    SuggestConfigResult,
    describe_analysis_modes,
    suggest_config,
)
from factrix._errors import (
    ConfigError,
    FactrixError,
    IncompatibleAxisError,
    InsufficientSampleError,
    ModeAxisError,
)
from factrix._evaluate import _evaluate as evaluate
from factrix._profile import FactorProfile
from factrix._types import MetricOutput

__version__ = "0.7.0"

__all__ = [
    # Configuration
    "AnalysisConfig",
    # Axis enums (Mode intentionally NOT exported — it is derived at
    # evaluate-time from N and read off profile.mode, never set by user
    # code; review fix UX-7. Still importable from factrix._axis.)
    "FactorScope",
    "Metric",
    "Signal",
    # Code enums
    "InfoCode",
    "StatCode",
    "Verdict",
    "WarningCode",
    # Errors
    "ConfigError",
    "FactrixError",
    "IncompatibleAxisError",
    "InsufficientSampleError",
    "ModeAxisError",
    # Profile + dispatch
    "FactorProfile",
    "MetricOutput",
    "evaluate",
    # Introspection
    "SuggestConfigResult",
    "describe_analysis_modes",
    "suggest_config",
    # Multi-factor namespace
    "multi_factor",
    # Synthetic panels
    "datasets",
]
