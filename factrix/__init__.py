"""factrix — Single-factor evaluation toolkit.

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

LLM agent reference: ``llms-full.txt`` covers concepts, public API, and
typical usage patterns in a single fetch. Two access paths::

    # Web — deployed at the docs site root
    https://awwesomeman.github.io/factrix/llms-full.txt

    # Local — shipped inside the wheel as package data
    import importlib.resources
    text = importlib.resources.files("factrix").joinpath("llms-full.txt").read_text()
"""

from typing import Any

from factrix import datasets, multi_factor, preprocess
from factrix._analysis_config import AnalysisConfig
from factrix._axis import (  # noqa: F401  Mode re-exported for namespace access; intentionally not in __all__
    FactorScope,
    Metric,
    Mode,
    Signal,
)
from factrix._codes import InfoCode, StatCode, Verdict, WarningCode
from factrix._describe import (
    SuggestConfigResult,
    describe_analysis_modes,
    list_metrics,
    suggest_config,
)
from factrix._errors import (
    ConfigError,
    FactrixError,
    IncompatibleAxisError,
    InsufficientSampleError,
    MissingConfigError,
    ModeAxisError,
)
from factrix._evaluate import _evaluate as _evaluate
from factrix._profile import FactorProfile
from factrix._types import MetricOutput


def evaluate(
    raw: Any,
    config: AnalysisConfig | None = None,
    /,
    *,
    factor_col: str = "factor",
) -> FactorProfile:
    """Dispatch ``raw`` through the cell selected by ``config``.

    Thin public wrapper around the private ``_evaluate`` dispatcher.
    Intercepts the common onboarding miss — ``evaluate(panel)`` — with
    a friendly :class:`MissingConfigError` pointing at
    :func:`suggest_config` and the Get Started guide.

    Args:
        raw: Long-format panel.
        config: Validated ``AnalysisConfig``.
        factor_col: Name of the signal column on ``raw`` (default
            ``"factor"``). Renamed to ``"factor"`` internally before
            dispatch. Looping over candidates with different
            ``factor_col=`` values is the canonical multi-factor
            pattern; see the batch screening guide.
    """
    if config is None:
        raise MissingConfigError(
            "evaluate() requires an AnalysisConfig. "
            "Call factrix.suggest_config(raw) for a recommendation, "
            "or see the Get Started guide: "
            "https://awwesomeman.github.io/factrix/getting-started/"
        )
    return _evaluate(raw, config, factor_col=factor_col)


__version__ = "0.8.0"

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
    "MissingConfigError",
    "ModeAxisError",
    # Profile + dispatch
    "FactorProfile",
    "MetricOutput",
    "evaluate",
    # Introspection
    "SuggestConfigResult",
    "describe_analysis_modes",
    "list_metrics",
    "suggest_config",
    # Multi-factor namespace
    "multi_factor",
    # Synthetic panels
    "datasets",
    # Forward-return preprocessing
    "preprocess",
]
