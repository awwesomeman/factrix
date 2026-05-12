"""factrix — Single-factor evaluation toolkit.

Three orthogonal user-facing axes — ``FactorScope``, ``Signal``,
``Metric`` — plus an evaluate-time-derived ``Mode`` define the analysis
cell. Construct a config via the four type-safe factories on
``AnalysisConfig``, dispatch via ``evaluate()``, inspect via the
returned ``FactorProfile``, and aggregate across factors with
``multi_factor.bhy`` for FDR-corrected screening.

Single-factor::

    import factrix as fx

    cfg = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC)
    profile = fx.evaluate(panel, cfg)
    print(profile.primary_p)
    print(profile.diagnose())

Batch + BHY::

    profiles = [fx.evaluate(panel, cfg) for cfg in candidate_configs]
    survivors = fx.multi_factor.bhy(profiles, q=0.05)

Schema reflection::

    print(fx.describe_analysis_modes())
    print(fx.suggest_config(panel))

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
from factrix._codes import InfoCode, StatCode, WarningCode
from factrix._compare import compare
from factrix._describe import (
    SuggestConfigResult,
    describe_analysis_modes,
    list_estimators,
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
    RunMetricsError,
    UnknownEstimatorError,
    UserInputError,
)
from factrix._evaluate import _evaluate as _evaluate
from factrix._profile import FactorProfile
from factrix._run_metrics import MetricsBundle, run_metrics
from factrix._types import MetricOutput
from factrix.slicing import (
    SliceResult,
    by_slice,
    slice_joint_test,
    slice_pairwise_test,
)


def evaluate(
    raw: Any,
    config: AnalysisConfig | None = None,
    /,
    *,
    factor_col: str = "factor",
) -> FactorProfile:
    """Evaluate one factor against its forward returns and return a FactorProfile.

    The profile carries ``primary_p`` (the headline p-value for downstream
    FDR), the cell-specific statistics, sample-size diagnostics, warnings,
    and the ``identity`` / ``context`` tuple used by multi-factor
    aggregators ([`bhy`][factrix.multi_factor.bhy] /
    [`partial_conjunction`][factrix.multi_factor.partial_conjunction] /
    [`bhy_hierarchical`][factrix.multi_factor.bhy_hierarchical]).

    Args:
        raw: Long-format panel satisfying the four-column floor
            ``(date, asset_id, factor, forward_return)``. See
            [Panel schema](panel-schema.md) for the canonical contract
            and dtype semantics.
        config: Validated ``AnalysisConfig`` selecting the dispatch cell
            (``Scope × Signal × Metric``). Construct via one of the four
            factories on the class.
        factor_col: Name of the signal column on ``raw`` (default
            ``"factor"``). Renamed to ``"factor"`` internally before
            dispatch. Looping over candidates with different
            ``factor_col=`` values is the canonical multi-factor pattern.

    Returns:
        [`FactorProfile`][factrix.FactorProfile] with ``primary_p``,
        ``stats``, ``warnings``, ``info_notes``, ``mode``, ``n_obs``,
        ``n_assets``, plus ``identity = (factor_id, forward_periods)``
        and ``context = {universe_id, regime_id, ...}``.

    Raises:
        MissingConfigError: ``evaluate(raw)`` called without an
            ``AnalysisConfig``. Recovery: call
            [`suggest_config`][factrix.suggest_config].
        IncompatibleAxisError: ``config`` axes form an illegal cell.
        ModeAxisError: Legal cell has no procedure under the derived
            ``Mode``. Carries ``.suggested_fix: AnalysisConfig | None``
            with the nearest-legal config.
        InsufficientSampleError: ``T`` below the procedure's
            ``MIN_PERIODS_HARD`` floor. Carries ``.actual_periods`` /
            ``.required_periods``.
        ValueError: ``factor_col`` not present on ``raw``, or both
            ``"factor"`` and ``factor_col`` present with differing values
            (ambiguous which is the signal — drop the unused column).

    All factrix-raised errors inherit from
    [`FactrixError`](errors.md).

    ??? note "Dispatch lore — cell schema, Mode, multi-factor cost"
        **Dispatch is explicit.** No auto-fallback when the panel shape
        does not match the cell. The one exception: `Common × Continuous`
        at `N == 1` auto-routes to the TIMESERIES single-series path
        (`profile.mode == "TIMESERIES"`) so single-asset macro factors
        still flow through.

        **Required columns per cell.** Every cell floors its
        `INPUT_SCHEMA` at the same four columns; optional columns
        activate additional standalone metrics and short-circuit
        gracefully (`NaN` + `reason`) when absent.

        | Cell                                              | Required                                | Optional column → enables                                                                                  |
        |---------------------------------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------|
        | Individual × Continuous (`ic`, `fama_macbeth`)    | `date, asset_id, factor, forward_return` | `market_cap` (or any name passed as `weight_col=`) → `quantile_spread_vw` value-weighting                  |
        | Individual × Sparse (event studies)               | `date, asset_id, factor, forward_return` | `price` → `event_around_return`, `mfe_mae_summary` (degrade gracefully if absent)                          |
        | Common × Continuous (broadcast macro factor)      | `date, asset_id, factor, forward_return` | —                                                                                                          |
        | Common × Sparse (broadcast event dummy)           | `date, asset_id, factor, forward_return` | —                                                                                                          |

        `forward_return` is part of the input contract — attach it via
        [`compute_forward_return`][factrix.preprocess.compute_forward_return]
        before the call so the horizon is explicit and aligned with
        `config.forward_periods`.

        **Mode — PANEL vs TIMESERIES.** Derived at evaluate-time from
        `N = panel["asset_id"].n_unique()` and surfaced on
        `profile.mode`:

        | `profile.mode`   | When                                    | Inference                                                            |
        |------------------|-----------------------------------------|----------------------------------------------------------------------|
        | `"PANEL"`        | `N ≥ 2` cross-sectional / event cells   | per-date statistic → time-series mean with NW HAC                    |
        | `"TIMESERIES"`   | `Common × Continuous` with `N == 1`     | single-series OLS with plain SE; HAC only on stage-2 aggregation     |

        Full conventions: [TIMESERIES-mode conventions](../reference/ts-mode-conventions.md).
        Sample-guard contract: [PANEL vs TIMESERIES](../guides/panel-timeseries.md).

        **Multi-factor cost.** Each call repeats the per-date
        cross-section work (sort / group-by / rank / HHI) on its own, so
        cost scales as `O(n_factors × per_date_cost)`. There is no
        shared-pass primitive; [`bhy`][factrix.multi_factor.bhy] controls
        FDR but does **not** reduce the per-signal evaluation cost.

    Examples:
        Single-factor inference:

        >>> config = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC)
        >>> profile = fx.evaluate(panel, config)
        >>> profile.primary_p
        0.0001

        Non-default signal column name:

        >>> profile = fx.evaluate(panel, config, factor_col="alpha")

        Multi-factor screening with FDR (see batch screening guide):

        >>> profiles = [fx.evaluate(panel, config, factor_col=name)
        ...             for name in candidate_signals]
        >>> survivors = fx.multi_factor.bhy(profiles)
    """
    if config is None:
        raise MissingConfigError(
            "evaluate() requires an AnalysisConfig. "
            "Call factrix.suggest_config(raw) for a recommendation, "
            "or see the Get Started guide: "
            "https://awwesomeman.github.io/factrix/getting-started/"
        )
    return _evaluate(raw, config, factor_col=factor_col)


__version__ = "0.12.0"

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
    "WarningCode",
    # Errors
    "ConfigError",
    "FactrixError",
    "IncompatibleAxisError",
    "InsufficientSampleError",
    "MissingConfigError",
    "ModeAxisError",
    "RunMetricsError",
    "UnknownEstimatorError",
    "UserInputError",
    # Profile + dispatch
    "FactorProfile",
    "MetricOutput",
    "MetricsBundle",
    "compare",
    "evaluate",
    "run_metrics",
    # Introspection
    "SuggestConfigResult",
    "describe_analysis_modes",
    "list_estimators",
    "list_metrics",
    "suggest_config",
    # Slicing dispatcher + cross-slice inference verbs
    "SliceResult",
    "by_slice",
    "slice_joint_test",
    "slice_pairwise_test",
    # Multi-factor namespace
    "multi_factor",
    # Synthetic panels
    "datasets",
    # Forward-return preprocessing
    "preprocess",
]
