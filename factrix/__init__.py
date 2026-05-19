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
    profile = fx.evaluate(panel, cfg)["factor"]
    print(profile.primary_p)
    print(profile.diagnose())

Batch + Benjamini-Hochberg-Yekutieli (BHY)::

    profiles = fx.evaluate(wide_panel, cfg, factor_cols=candidate_cols)
    survivors = fx.multi_factor.bhy(profiles.values(), q=0.05)

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

from collections.abc import Sequence

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
from factrix._dag import CycleError, DagExecutor
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
from factrix._evaluate import evaluate_chunked as evaluate_chunked
from factrix._evaluate import evaluate_iter as evaluate_iter
from factrix._inspect import (
    MetricApplicability,
    PanelInspection,
    PanelProperties,
    PanelReasoning,
    inspect_panel,
)
from factrix._metric_index import SampleFloor
from factrix._panel_input import PanelInput, _coerce_panel
from factrix._profile import FactorProfile
from factrix._results import EvaluationResult, MetricResultGroup, Warning
from factrix._run_metrics import (
    MetricsBundle,
    run_metrics,
    run_metrics_chunked,
    run_metrics_iter,
)
from factrix._types import MetricOutput
from factrix.slicing import (
    SliceResult,
    by_slice,
    slice_joint_test,
    slice_pairwise_test,
)


def evaluate(
    panel: PanelInput,
    config: AnalysisConfig | None = None,
    /,
    *,
    factor_cols: Sequence[str] = ("factor",),
) -> dict[str, FactorProfile]:
    """Evaluate one or more factors against forward returns.

    Returns ``dict[factor_id, FactorProfile]`` keyed by the names in
    ``factor_cols``. Mirrors the ``run_metrics`` contract from #402 so
    the two entry points share an input arity model — pass a list,
    get back a dict — even for the single-factor case
    (``fx.evaluate(panel, cfg)["factor"]``).

    The profile carries ``primary_p`` (the headline p-value for downstream
    false discovery rate (FDR)), the cell-specific statistics, sample-size diagnostics, warnings,
    and the ``identity`` / ``context`` tuple used by multi-factor
    aggregators ([`bhy`][factrix.multi_factor.bhy] /
    [`partial_conjunction`][factrix.multi_factor.partial_conjunction] /
    [`bhy_hierarchical`][factrix.multi_factor.bhy_hierarchical]).

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
        | `"PANEL"`        | `N ≥ 2` cross-sectional / event cells   | per-date statistic → time-series mean with Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) |
        | `"TIMESERIES"`   | `Common × Continuous` with `N == 1`     | single-series ordinary least squares (OLS) with plain SE; HAC only on stage-2 aggregation |

        Full conventions: [Timeseries-mode conventions](../reference/ts-mode-conventions.md).
        Sample-guard contract: [Panel vs timeseries](../guides/panel-timeseries.md).

        **Multi-factor cost.** Each call repeats the per-date
        cross-section work (sort / group-by / rank / Herfindahl-Hirschman index (HHI)) on its own, so
        cost scales as `O(n_factors × per_date_cost)`. There is no
        shared-pass primitive; [`bhy`][factrix.multi_factor.bhy] controls
        FDR but does **not** reduce the per-signal evaluation cost.

    Args:
        panel: Long-format panel satisfying the four-column floor
            ``(date, asset_id, factor, forward_return)``. Accepts
            ``pl.DataFrame`` or ``pl.LazyFrame`` (collected at the
            boundary). pandas users go through ``factrix.adapt``
            (which also renames columns) or ``pl.from_pandas`` first.
            See
            [Panel schema](panel-schema.md) for the column contract
            and [Efficient data loading](../guides/efficient-loading.md)
            for large-panel recipes.
        config: Validated ``AnalysisConfig`` selecting the dispatch cell
            (``Scope × Signal × Metric``). Construct via one of the four
            factories on the class.
        factor_cols: Names of the signal columns on ``panel`` to
            evaluate. Each column is projected and renamed to
            ``"factor"`` before dispatch; the returned dict is keyed
            by the original ``factor_cols`` name and each profile's
            ``factor_id`` is stamped to match. Default ``("factor",)``
            keeps the canonical single-factor case ergonomic — index
            via ``["factor"]`` to get the profile.

    Returns:
        ``dict[factor_name, FactorProfile]`` — one entry per name in
        ``factor_cols``. Each [`FactorProfile`][factrix.FactorProfile]
        carries ``primary_p``, ``stats``, ``warnings``, ``info_notes``,
        ``mode``, ``n_obs``, ``n_assets``, plus
        ``identity = (factor_id, forward_periods)`` and
        ``context = {universe_id, regime_id, ...}``. Feed
        ``.values()`` to [`bhy`][factrix.multi_factor.bhy] for FDR
        screening.

    Raises:
        MissingConfigError: ``evaluate(panel)`` called without an
            ``AnalysisConfig``. Recovery: call
            [`suggest_config`][factrix.suggest_config].
        IncompatibleAxisError: ``config`` axes form an illegal cell.
        ModeAxisError: Legal cell has no procedure under the derived
            ``Mode``. Carries ``.suggested_fix: AnalysisConfig | None``
            with the nearest-legal config.
        InsufficientSampleError: ``T`` below the procedure's
            ``MIN_PERIODS_HARD`` floor. Carries ``.actual_periods`` /
            ``.required_periods``.
        UserInputError: ``factor_cols`` empty, contains duplicates, or
            references a column not present on ``panel``.

    Examples:
        Single-factor inference on a cross-sectional panel:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=250)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> cfg = fx.AnalysisConfig.individual_continuous(forward_periods=5)
        >>> profile = fx.evaluate(panel, cfg)["factor"]

        Non-default signal column name:

        >>> panel_renamed = panel.rename({"factor": "alpha"})
        >>> profile = fx.evaluate(panel_renamed, cfg, factor_cols=["alpha"])["alpha"]

        Multi-factor batch:

        >>> profiles = fx.evaluate(wide_panel, cfg, factor_cols=["alpha", "beta"])  # doctest: +SKIP
        >>> survivors = fx.multi_factor.bhy(profiles.values())                       # doctest: +SKIP
    """
    if config is None:
        raise MissingConfigError(
            "evaluate() requires an AnalysisConfig. "
            "Call factrix.suggest_config(panel) for a recommendation, "
            "or see the Get Started guide: "
            "https://awwesomeman.github.io/factrix/getting-started/"
        )
    panel = _coerce_panel(panel)
    return _evaluate(panel, config, factor_cols=factor_cols)


__version__ = "0.13.0"

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
    "CycleError",
    "DagExecutor",
    "EvaluationResult",
    "FactorProfile",
    "MetricOutput",
    "MetricResultGroup",
    "MetricsBundle",
    "PanelInput",
    "Warning",
    "compare",
    "evaluate",
    "evaluate_chunked",
    "evaluate_iter",
    "run_metrics",
    "run_metrics_chunked",
    "run_metrics_iter",
    # Introspection
    "MetricApplicability",
    "PanelInspection",
    "PanelProperties",
    "PanelReasoning",
    "SampleFloor",
    "SuggestConfigResult",
    "describe_analysis_modes",
    "inspect_panel",
    "list_estimators",
    "list_metrics",
    "suggest_config",
    # Slicing dispatcher + cross-slice inference functions
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
