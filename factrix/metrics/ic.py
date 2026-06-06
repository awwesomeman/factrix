"""IC (Information Coefficient) computation for cross-sectional panels.

Notes:
    **Pipeline.** Per-date Spearman rank IC (cross-section step) → IC
    time series, then non-overlapping cross-asset t or Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) t on its
    mean; the regime variant slices the same pipeline.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.

    **Output.** Time-indexed IC series (``date, ic``) that can be fed
    into any ``series/`` tool (oos, trend, significance, hit_rate).
"""

from __future__ import annotations

import math
import warnings as _warnings

import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
    SEMethod,
    TestMethod,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import (
    _calc_t_stat,
    _newey_west_t_test,
    _p_value_from_t,
)
from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_WARN
from factrix._types import (
    EPSILON,
    MIN_ASSETS_PER_DATE_IC,
)
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    TIE_RATIO_WARN_THRESHOLD,
    _sample_non_overlapping,
    _scaled_min_periods,
    _short_circuit_output,
)
from factrix.metrics._metric_capabilities import per_date_series_rename
from factrix.metrics._primitives import compute_ic

__all__ = [  # noqa: RUF022 (teaching order, see SSOT note)
    "ic",
    "ic_newey_west",
    "ic_ir",
]

_IC_CELL = cell(
    FactorScope.INDIVIDUAL,
    FactorDensity.DENSE,
    structure=DataStructure.PANEL,
)

# Slice-test contract (#153 §5): IC is per-date Spearman rank
# correlation, not a bucketed metric — slice tests never need to
# downscale `n_groups`. The min-cross-section-per-date constraint
# (Spearman ρ asymptotic distribution requires ≥ 30 obs per date,
# Hollander-Wolfe-Chicken §8.6) lives in the procedure as
# `MIN_ASSETS_PER_DATE_IC` short-circuit, parallel to (not exposed
# via) this attribute.
min_assets_per_group: int | None = None
per_date_series = per_date_series_rename("ic")


def _median_tie_ratio(ic_df: pl.DataFrame) -> float:
    """Median of the per-date ``tie_ratio`` column, or ``nan`` if absent/empty."""
    if "tie_ratio" not in ic_df.columns:
        return float("nan")
    med = ic_df["tie_ratio"].median()
    return float("nan") if med is None else float(med)  # type: ignore[arg-type]


def _warn_if_high_ic_tie_ratio(ic_df: pl.DataFrame, metric_name: str) -> float:
    """Emit ``UserWarning`` when median tie_ratio exceeds the global threshold.

    Returns the median for caller to stash in metadata. The Spearman ρ on
    average ranks is biased relative to the tie-corrected formula
    (Kendall-Stuart §31) at high tie densities — a bucketed / categorical
    factor will look like it has IC ≈ 0 even if the bucketing is
    informative. Threshold reuses the global ``TIE_RATIO_WARN_THRESHOLD``
    (0.3) shared with the quantile-bucketing diagnostics.
    """
    med = _median_tie_ratio(ic_df)
    if not math.isnan(med) and med > TIE_RATIO_WARN_THRESHOLD:
        _warnings.warn(
            f"{metric_name}: median tie_ratio={med:.3f} exceeds "
            f"{TIE_RATIO_WARN_THRESHOLD:.2f}. Spearman ρ on average ranks is "
            f"biased on bucketed / categorical factors; treat the IC "
            f"magnitude as a lower bound and consider a tie-corrected "
            f"correlation or a continuous transform of the factor.",
            UserWarning,
            stacklevel=2,
        )
    return med


@metric(
    cell=_IC_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
    input_shape=InputShape.SERIES,
    requires={"ic_df": compute_ic},
    sample_threshold=SampleThreshold(),
)
def ic(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> MetricResult:
    r"""Information coefficient (IC) mean significance: is mean IC significantly different from zero?

    No static panel-shape thresholds are declared (sample_threshold=SampleThreshold()) because the minimum required periods depend dynamically on the forward_periods parameter.

    Args:
        ic_df: Output of ``compute_ic()``.
        forward_periods: Sampling interval for non-overlapping dates.

    Returns:
        MetricResult with value=mean IC, t_stat from non-overlapping sampling.

    Notes:
        Given the per-date IC series $\mathrm{IC}_t$, significance is
        $t = \mathrm{mean}(\mathrm{IC}) / (\mathrm{std}(\mathrm{IC}) / \sqrt{n})$
        computed on a non-overlapping subsample (every
        ``forward_periods``-th date). $H_0: \mathbb{E}[\mathrm{IC}] = 0$.

        factrix uses non-overlapping resampling rather than Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC)
        for the default ``ic`` test to avoid the lag floor implied by
        overlapping forward returns; the HAC route is offered separately
        as ``ic_newey_west`` for callers who prefer to keep every sample.

    References:
        [Grinold 1989][grinold-1989]: IC as the canonical density-quality
        measure under the Fundamental Law of Active Management.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: K-period overlapping
        returns carry MA(K-1) autocorrelation — the motivation for the
        non-overlap stride used here.

    Examples:
        Chain from :func:`compute_ic` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic, ic
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic(ic_df, forward_periods=5)
        >>> result.name == ""
        True
    """
    median_tie = _warn_if_high_ic_tie_ratio(ic_df, "ic")
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    raw_min = _scaled_min_periods(MIN_ASSETS_PER_DATE_IC, forward_periods)
    if n < raw_min:
        return _short_circuit_output(
            "ic",
            "insufficient_ic_periods",
            n_obs=n,
            min_required=raw_min,
            forward_periods=forward_periods,
        )

    mean_ic = float(ic_vals.mean())  # type: ignore[arg-type]
    sampled = _sample_non_overlapping(ic_df, forward_periods)["ic"].drop_nulls()
    n_sampled = len(sampled)
    if n_sampled < MIN_ASSETS_PER_DATE_IC:
        return _short_circuit_output(
            "ic",
            "insufficient_sampled_ic_periods",
            n_obs=n_sampled,
            min_required=MIN_ASSETS_PER_DATE_IC,
            forward_periods=forward_periods,
        )
    t = _calc_t_stat(float(sampled.mean()), float(sampled.std()), n_sampled)  # type: ignore[arg-type]
    p = _p_value_from_t(t, n_sampled)

    return MetricResult(
        p_value=p,
        value=mean_ic,
        stat=t,
        metadata={
            "n_periods": n,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "non-overlapping t-test",
            "tie_ratio": median_tie,
        },
    )


@metric(
    cell=_IC_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
    input_shape=InputShape.SERIES,
    requires={"ic_df": compute_ic},
    sample_threshold=SampleThreshold(
        min_periods=MIN_PERIODS_HARD,
        warn_periods=MIN_PERIODS_WARN,
    ),
)
def ic_newey_west(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> MetricResult:
    r"""Information coefficient (IC) mean significance via Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) $t$-test on the overlapping series.

    Sibling of ``ic()``: same null hypothesis ($H_0$: mean IC = 0), but
    keeps every observation and absorbs the autocorrelation induced by
    overlapping ``forward_periods``-day returns through HAC standard
    errors rather than dropping samples.

    Notes:
        $t = \mathrm{mean}(\mathrm{IC}) / \mathrm{SE}_{\mathrm{NW}}(\mathrm{IC})$
        on the full overlapping IC series. Lag selection:
        $L = \max(\lfloor T^{1/3} \rfloor, h - 1)$ (with $h$ = ``forward_periods``)
        — the [Andrews (1991)][andrews-1991] Bartlett growth rate, floored against the
        Hansen-Hodrick MA($h-1$) overlap horizon so the kernel covers
        the induced dependence.

        factrix uses the Andrews fixed-rate rule rather than the
        [Newey-West (1994)][newey-west-1994] data-adaptive bandwidth — simpler, deterministic
        across reruns, and adequate at the typical $T$ of factor research.

    References:
        [Newey-West 1987][newey-west-1987]: HAC variance estimator.
        [Andrews 1991][andrews-1991]: optimal Bartlett growth rate
        $T^{1/3}$ underlying the default lag rule.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: ``forward_periods - 1``
        floor for overlapping returns.
        [Newey-West 1994][newey-west-1994]: data-adaptive lag-selection
        alternative; cited as background.

    Examples:
        Chain from :func:`compute_ic` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic, ic_newey_west
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic_newey_west(ic_df, forward_periods=5)
        >>> result.name == ""
        True
    """
    median_tie = _warn_if_high_ic_tie_ratio(ic_df, "ic_newey_west")
    ic_vals = ic_df["ic"].drop_nulls().to_numpy()
    n = len(ic_vals)
    min_periods = ic_newey_west.sample_threshold.min_periods  # type: ignore[attr-defined]
    if min_periods is not None and n < min_periods:
        return _short_circuit_output(
            "ic_newey_west",
            "insufficient_ic_periods",
            n_obs=n,
            min_required=min_periods,
        )

    from factrix._stats import _resolve_nw_lags

    lags = _resolve_nw_lags(n, lags=None, forward_periods=forward_periods)
    t, p, _ = _newey_west_t_test(ic_vals, forward_periods=forward_periods)
    return MetricResult(
        p_value=p,
        value=float(ic_vals.mean()),
        stat=t,
        metadata={
            "n_periods": n,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "Newey-West HAC t-test on overlapping IC series",
            "newey_west_lags": lags,
            "forward_periods": forward_periods,
            "tie_ratio": median_tie,
        },
    )


@metric(
    cell=_IC_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
    input_shape=InputShape.SERIES,
    requires={"ic_df": compute_ic},
    sample_threshold=SampleThreshold(
        min_periods=MIN_PERIODS_HARD,
        warn_periods=MIN_PERIODS_WARN,
    ),
)
def ic_ir(
    ic_df: pl.DataFrame,
) -> MetricResult:
    r"""$\mathrm{ICIR} = \mathrm{mean}(\mathrm{IC}) / \mathrm{std}(\mathrm{IC})$.

    Signed ratio — positive when information coefficient (IC) is consistently positive, negative
    when consistently negative.  Analogous to a Sharpe ratio for the
    factor density.

    This is a **descriptive statistic**, not a hypothesis test (t_stat=None).
    For significance testing, use ``ic()``.

    Args:
        ic_df: Output of ``compute_ic()``.

    Returns:
        MetricResult with value=IC_IR (signed), t_stat=None.

    Notes:
        $\mathrm{ICIR} = \mathrm{mean}(\mathrm{IC}) / \mathrm{std}(\mathrm{IC})$
        over the per-date IC series — a Sharpe-style ratio describing
        time-series stability of the density. Reported as a descriptive
        statistic; no inference is attached because the heteroskedasticity-and-autocorrelation-consistent (HAC)-corrected
        significance test on $\mathrm{mean}(\mathrm{IC})$ lives in ``ic``
        / ``ic_newey_west``.

    References:
        [Grinold 1989][grinold-1989]: ICIR is the time-stability
        normalisation that completes the IR decomposition.

    Examples:
        Chain from :func:`compute_ic` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic, ic_ir
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic_ir(ic_df)
        >>> result.name == ""
        True
    """
    median_tie = _warn_if_high_ic_tie_ratio(ic_df, "ic_ir")
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    min_periods = ic_ir.sample_threshold.min_periods  # type: ignore[attr-defined]
    if min_periods is not None and n < min_periods:
        return _short_circuit_output(
            "ic_ir",
            "insufficient_ic_periods",
            n_obs=n,
            min_required=min_periods,
        )

    mean_ic = float(ic_vals.mean())  # type: ignore[arg-type]
    std_ic = float(ic_vals.std())  # type: ignore[arg-type]

    if std_ic < EPSILON:
        return _short_circuit_output(
            "ic_ir",
            "degenerate_ic_variance",
            std_ic=std_ic,
        )

    ratio = mean_ic / std_ic

    return MetricResult(
        value=ratio,
        metadata={
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "n_periods": n,
            "tie_ratio": median_tie,
        },
    )
