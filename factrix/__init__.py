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
    """Single-factor evaluation entry point.

    Routes a ``(raw, AnalysisConfig)`` pair to the procedure registered
    for the dispatch cell selected by ``config`` and returns a
    :class:`FactorProfile`. Dispatch is **explicit** — there is no
    auto-fallback when the panel shape does not match the cell. ``N == 1``
    is the one exception: ``Common × Continuous`` auto-routes to the
    TIMESERIES single-series path (``profile.mode == "TIMESERIES"``) so
    single-asset macro factors still flow through.

    Args:
        raw: Long-format panel satisfying the four-column floor
            ``(date, asset_id, factor, forward_return)``. See
            [Panel schema](panel-schema.md) for the canonical contract
            and dtype semantics. Per-cell optional columns (``market_cap``
            / ``price``) activate additional standalone metrics — see
            "Required columns per cell" below.
        config: Validated ``AnalysisConfig`` selecting the dispatch cell
            (``Scope × Signal × Metric``). Construct via one of the four
            factories on the class.
        factor_col: Name of the signal column on ``raw`` (default
            ``"factor"``). Renamed to ``"factor"`` internally before
            dispatch so each procedure's ``INPUT_SCHEMA`` sees the
            canonical schema. Looping over candidates with different
            ``factor_col=`` values is the canonical multi-factor pattern;
            downstream aggregation goes through
            :func:`factrix.multi_factor.bhy` for FDR control.

    Returns:
        :class:`FactorProfile` carrying ``primary_p``, ``stats``,
        ``warnings``, ``info_notes``, ``mode``, ``n_obs``, ``n_assets``,
        and ``identity`` / ``context``.

    Raises:
        MissingConfigError: ``evaluate(raw)`` called without an
            ``AnalysisConfig``. Recovery: call
            :func:`factrix.suggest_config`.
        IncompatibleAxisError: ``config`` axes form an illegal cell.
        ModeAxisError: Legal cell has no procedure under the derived
            ``Mode``. Carries ``.suggested_fix: AnalysisConfig | None``
            with the nearest-legal config.
        InsufficientSampleError: ``T`` below the procedure's
            ``MIN_PERIODS_HARD`` floor. Carries ``.actual_periods`` /
            ``.required_periods``.
        ValueError: ``factor_col`` not present on ``raw``, or both
            ``"factor"`` and ``factor_col`` present with differing values
            (ambiguous which is the signal — drop the unused column
            before calling).

    All factrix-raised errors inherit from :class:`FactrixError`.

    Required columns per cell:
        Every dispatch cell floors its ``INPUT_SCHEMA`` at the same four
        columns; some cells expose optional columns that activate
        additional standalone metrics (absent optional columns
        short-circuit gracefully with ``NaN`` + ``reason``, not failure).

        | Cell                                              | Required                                | Optional column → enables                                                                                  |
        |---------------------------------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------|
        | Individual × Continuous (``ic``, ``fama_macbeth``) | ``date, asset_id, factor, forward_return`` | ``market_cap`` (or any name passed as ``weight_col=``) → ``quantile_spread_vw`` value-weighting             |
        | Individual × Sparse (event studies)               | ``date, asset_id, factor, forward_return`` | ``price`` → ``event_around_return``, ``mfe_mae_summary`` (degrade gracefully if absent)                    |
        | Common × Continuous (broadcast macro factor)      | ``date, asset_id, factor, forward_return`` | —                                                                                                          |
        | Common × Sparse (broadcast event dummy)           | ``date, asset_id, factor, forward_return`` | —                                                                                                          |

        ``forward_return`` is part of the input contract, not computed
        inside ``evaluate``. Attach it via
        :func:`factrix.preprocess.compute_forward_return` before the call
        so the horizon is explicit and aligned with
        ``config.forward_periods``. The two synthetic dataset generators
        (``make_cs_panel``, ``make_event_panel``) emit
        ``(date, asset_id, factor, price)`` and require the same
        preprocessing step.

    Mode — PANEL vs TIMESERIES:
        ``Mode`` is not user-facing; it is derived at evaluate-time from
        ``N = panel["asset_id"].n_unique()`` and surfaces on
        ``profile.mode``.

        | ``profile.mode`` | When                                    | Inference                                                            |
        |------------------|-----------------------------------------|----------------------------------------------------------------------|
        | ``"PANEL"``      | ``N ≥ 2`` cross-sectional / event cells | per-date statistic → time-series mean with NW HAC                    |
        | ``"TIMESERIES"`` | ``Common × Continuous`` with ``N == 1`` | single-series OLS with plain SE; HAC only on stage-2 aggregation     |

        Full conventions: [TIMESERIES-mode conventions](../reference/ts-mode-conventions.md).
        Sample-guard contract and dispatch matrix: [PANEL vs TIMESERIES](../guides/panel-timeseries.md).

    Multi-factor cost:
        Each ``evaluate`` call repeats the per-date cross-section work
        (sort / group-by / rank / HHI) on its own, so cost scales as
        ``O(n_factors × per_date_cost)`` — there is no shared-pass
        primitive. :func:`factrix.multi_factor.bhy` operates on the
        resulting profile list for FDR control; it does **not** reduce
        the per-signal evaluation cost.

    Examples:
        Single-factor inference:

        >>> import factrix as fx
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

    See Also:
        - :class:`factrix.FactorProfile` — return type.
        - :class:`factrix.AnalysisConfig` — cell selection.
        - :func:`factrix.run_metrics` — descriptive twin (no FDR claim).
        - :func:`factrix.suggest_config` — recover from ``MissingConfigError``.
        - :func:`factrix.multi_factor.bhy` — multi-factor FDR step.
        - [Panel schema](panel-schema.md) — input contract.
        - [Batch screening guide](../guides/batch-screening.md) — end-to-end
          multi-factor workflow.
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
