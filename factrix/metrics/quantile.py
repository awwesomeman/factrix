"""Quantile analysis for cross-sectional panels.

All spread series are time-indexed (``date, value``) and can be fed
into any ``series/`` tool.

Notes:
    **Pipeline.** Per-date long-short spread on quantile groups
    (cross-section step), then non-overlapping t on the spread series.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.

    **Output.** Spread series, long/short alpha decomposition.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat, _p_value_from_t, _significance_marker
from factrix._types import (
    DDOF,
    MIN_PORTFOLIO_PERIODS_HARD,
)
from factrix.inference import NEWEY_WEST, NON_OVERLAPPING, NeweyWest, NonOverlapping
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _all_dates_degenerate,
    _assign_quantile_groups,
    _check_applicable_inference,
    _compute_tie_ratio,
    _enforce_scaled_floor,
    _is_thin_quantile_groups,
    _lag_within_asset,
    _no_signal_zero_variance,
    _sample_non_overlapping,
    _scaled_periods_threshold,
    _short_circuit_output,
    _spread_significance_with_inference,
    _surface_null_drop,
    _warn_high_tie_ratio,
)
from factrix.metrics._primitives import (
    compute_group_returns,
    compute_spread_series,
)

__all__ = [  # noqa: RUF022 (teaching order, see SSOT note)
    "compute_spread_series",
    "compute_group_returns",
    "quantile_spread",
    "quantile_spread_vw",
]

_Q_CELL = cell(
    FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
)


# Periods floor scales with the non-overlap stride: the headline t-test runs on
# ``raw_n / forward_periods`` sampled dates, so pre-flight needs ``raw_n >=
# MIN_PORTFOLIO_PERIODS_HARD * forward_periods`` to land that many effective
# periods. The resolver and the in-body :func:`_enforce_scaled_floor` gate share
# ``MIN_PORTFOLIO_PERIODS_HARD`` + ``_scaled_min_periods``, so the floors agree.
_PORTFOLIO_PERIODS_FLOOR = _scaled_periods_threshold(MIN_PORTFOLIO_PERIODS_HARD)

# Inference allowlist: the spread dispatch hard-branches on ``NeweyWest`` for the
# HAC path and runs the non-overlap t-test otherwise, so the union is the exact
# set it handles. Anything else (``HansenHodrick``, a non-``Inference`` object)
# is rejected rather than silently reported as non-overlap.
applicable_inference: frozenset[NonOverlapping | NeweyWest] = frozenset(
    {NON_OVERLAPPING, NEWEY_WEST}
)


def _quantile_groups_threshold(self) -> SampleThreshold:
    """Periods floor plus a ``min_assets >= n_groups`` cross-sectional floor.

    Shared by ``quantile_spread`` and ``quantile_spread_vw``: both bucket each
    date into ``n_groups`` quantiles, so a date needs at least ``n_groups``
    valid names to fill the top and bottom legs.
    """
    periods = _PORTFOLIO_PERIODS_FLOOR(self)
    return SampleThreshold(
        min_periods=periods.min_periods,
        warn_periods=periods.warn_periods,
        min_assets=self.n_groups,
    )


@metric(
    cell=_Q_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    batchable=True,
    sample_threshold=_quantile_groups_threshold,
)
def quantile_spread(
    data: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_cols: Sequence[str] = ("factor",),
    tie_policy: str = "ordinal",
    inference: NonOverlapping | NeweyWest = NON_OVERLAPPING,
    *,
    _precomputed_series: dict[str, pl.DataFrame] | None = None,
) -> dict[str, MetricResult]:
    """long-short spread (per-period mean).

    Args:
        inference: Headline significance method on the per-date spread.
            ``fx.inference.NON_OVERLAPPING`` (default) runs the OLS t-test
            on the non-overlap stride subsample; ``fx.inference.NEWEY_WEST``
            keeps every date and absorbs the MA(h-1) overlap in a HAC SE.
            On a small cross-section (``n_assets < 30``) the heavy-tail
            block bootstrap takes precedence over either (see Notes).
        _precomputed_series: If provided, skip recomputing ``compute_spread_series``.
        tie_policy: Bucketing tie-break policy, see ``_assign_quantile_groups``.
            When ``_precomputed_series`` is passed, this only affects the
            ``tie_ratio`` diagnostic — the series itself was already built.

    Returns:
        MetricResult with per-period mean spread, t-stat from the chosen
        ``inference``.

    Notes:
        ``t = mean(spread) / (std(spread) / sqrt(n))`` on the non-overlap
        spread series. H0: ``E[spread] = 0``. The Newey-West (NW)
        heteroskedasticity-and-autocorrelation-consistent (HAC) route is
        the sibling that keeps the full overlapping series instead of
        striding — select it via ``inference=fx.inference.NEWEY_WEST``.
        Because HAC corrects autocorrelation rather than heavy tails, the
        small-cross-section block bootstrap still wins when it fires and the
        requested HAC is flagged ``inference_overridden`` in metadata.

        Long/short alpha decomposition stays a descriptive OLS t-test on
        ``top_return - universe_return`` and ``universe_return -
        bottom_return`` regardless of ``inference`` — it attributes the
        spread to long-side vs short-side excess, it is not the headline H0.

    References:
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: overlapping-return
        autocorrelation, motivating the non-overlap stride.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import quantile_spread
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = quantile_spread(panel, forward_periods=5, n_groups=5)
        >>> result["factor"].name == ""
        True
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")
    if _precomputed_series is not None and set(_precomputed_series) != set(cols):
        raise ValueError(
            "_precomputed_series keys must match factor_cols "
            f"(got {sorted(_precomputed_series)} vs {sorted(cols)})"
        )
    _check_applicable_inference(
        inference, applicable_inference, func_name="quantile_spread"
    )

    # Sample once across all factors; bucketing tie_ratio is computed
    # on the sampled subset (what bucketing actually sees) rather than
    # the full panel — ~N/forward_periods smaller scan.
    sampled = _sample_non_overlapping(data, forward_periods)
    series_by_factor = (
        _precomputed_series
        if _precomputed_series is not None
        else compute_spread_series(
            data,
            n_groups=n_groups,
            factor_cols=cols,
            tie_policy=tie_policy,
            forward_periods=forward_periods,
        )
    )
    # The HAC path needs the full overlapping spread series (every date);
    # ``forward_periods=1`` is the no-stride build of the same primitive.
    full_series_by_factor: dict[str, pl.DataFrame] | None = (
        compute_spread_series(
            data,
            n_groups=n_groups,
            factor_cols=cols,
            tie_policy=tie_policy,
            forward_periods=1,
        )
        if isinstance(inference, NeweyWest)
        else None
    )
    # Raw (pre-sampling) date count: the axis the stride-scaled periods floor is
    # calibrated against, shared across factors.
    n_raw_periods = data["date"].n_unique()
    return {
        f: _quantile_spread_from_series(
            series=series_by_factor[f],
            sampled=sampled,
            n_raw_periods=n_raw_periods,
            factor_col=f,
            tie_policy=tie_policy,
            inference=inference,
            forward_periods=forward_periods,
            n_groups=n_groups,
            full_series=(
                full_series_by_factor[f] if full_series_by_factor is not None else None
            ),
        )
        for f in cols
    }


def _quantile_spread_from_series(
    *,
    series: pl.DataFrame,
    sampled: pl.DataFrame,
    n_raw_periods: int,
    factor_col: str,
    tie_policy: str,
    inference: NonOverlapping | NeweyWest,
    forward_periods: int,
    n_groups: int,
    full_series: pl.DataFrame | None,
) -> MetricResult:
    """Per-factor t-test pipeline shared by single and batch paths.

    ``sampled`` is the (already non-overlap-sampled) panel; ``series``
    is this factor's spread DataFrame; ``n_raw_periods`` is the full date count
    before sampling. Splitting this out lets the batch path share the sample
    step across every factor while the single-factor path stays a one-liner.
    """
    tie_ratio = _compute_tie_ratio(sampled, factor_col)
    _warn_high_tie_ratio(tie_ratio, "quantile_spread", tie_policy)
    spread_vals = series["spread"].drop_nulls()
    n = len(spread_vals)
    sc = _enforce_scaled_floor(
        "quantile_spread",
        n_raw_periods,
        MIN_PORTFOLIO_PERIODS_HARD,
        forward_periods,
        "insufficient_portfolio_periods",
        tie_ratio=tie_ratio,
        tie_policy=tie_policy,
    )
    if sc is not None:
        return sc
    per_date_assets = series["_n_assets"]
    if bool((per_date_assets < n_groups).all()):
        max_assets_value = per_date_assets.max()
        max_assets = 0 if max_assets_value is None else cast(int, max_assets_value)
        return _short_circuit_output(
            "quantile_spread",
            "insufficient_assets_for_quantile_groups",
            n_obs=max_assets,
            n_obs_axis="assets",
            n_groups=n_groups,
            min_required=n_groups,
            max_assets_per_date=max_assets,
        )
    if bool(series["_zero_variance_factor"].all()):
        return _no_signal_zero_variance(
            series.height,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
            n_groups=n_groups,
        )

    arr = spread_vals.to_numpy()
    # Headline test: ``inference`` selects non-overlap t vs Newey-West HAC;
    # small cross-sections (n_assets < MIN_ASSETS_WARN) switch either to a
    # block-bootstrap CI (shared policy with k_spread). ``mean_spread`` is the
    # full-sample mean under HAC, the non-overlap mean otherwise.
    n_assets = sampled["asset_id"].n_unique()
    mean_spread, t, p, sig_method, sig_extra, sig_codes = (
        _spread_significance_with_inference(
            inference,
            strided_spread=arr,
            full_spread=full_series,
            forward_periods=forward_periods,
            n_assets=n_assets,
        )
    )

    # Long/short decomposition (spread = long_alpha + short_alpha)
    long_excess = (series["top_return"] - series["universe_return"]).drop_nulls()
    short_excess = (series["universe_return"] - series["bottom_return"]).drop_nulls()

    long_arr = long_excess.to_numpy()
    short_arr = short_excess.to_numpy()

    mean_long = float(np.mean(long_arr))
    std_long = float(np.std(long_arr, ddof=DDOF))
    t_long = _calc_t_stat(mean_long, std_long, len(long_arr))
    p_long = _p_value_from_t(t_long, len(long_arr))

    mean_short = float(np.mean(short_arr))
    std_short = float(np.std(short_arr, ddof=DDOF))
    t_short = _calc_t_stat(mean_short, std_short, len(short_arr))
    p_short = _p_value_from_t(t_short, len(short_arr))

    metadata: dict[str, object] = {
        "n_periods": n,
        "stat_type": "t",
        "h0": "mu=0",
        "method": sig_method,
        "long_alpha": mean_long,
        "short_alpha": mean_short,
        "long_stat": t_long,
        "long_p_value": p_long,
        "short_stat": t_short,
        "short_p_value": p_short,
        "short_significance": _significance_marker(p_short),
        "tie_ratio": tie_ratio,
        "tie_policy": tie_policy,
        **sig_extra,
    }
    warning_codes = list(sig_codes)
    # Structured twin of the spread primitive's thin-group advisory: surface the
    # same condition on warning_codes so result-only inspection sees it.
    if _is_thin_quantile_groups(sampled, n_groups):
        warning_codes.append(WarningCode.THIN_QUANTILE_GROUPS.value)
    _surface_null_drop(
        n_periods_in=series.height,
        n_periods_out=n,
        drop_reason="null spread observations in the series",
        metric_name="quantile_spread",
        metadata=metadata,
        warning_codes=warning_codes,
    )
    return MetricResult(
        p_value=p,
        value=mean_spread,
        n_obs=n,
        n_obs_axis="periods",
        stat=t,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


@metric(
    cell=_Q_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    sample_threshold=_quantile_groups_threshold,
)
def quantile_spread_vw(
    data: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    weight_col: str = "market_cap",
    tie_policy: str = "ordinal",
    lag_weights: bool = True,
) -> MetricResult:
    r"""Value-weighted long-short spread — alpha concentration diagnostic.

    Formula (per non-overlapping date $t$):

    $$
    \begin{aligned}
    \text{vw}_b[t] &= \frac{\sum_{i \in b} w_{i,t-1} \cdot \text{return}_{i, t \to t+h}}{\sum_{i \in b} w_{i,t-1}}, \quad b \in \{\text{bottom}, \text{top}\} \\
    \text{spread}[t] &= \text{vw}_{\text{top}}[t] - \text{vw}_{\text{bottom}}[t] \\
    \text{value} &= \mathrm{mean}_t\, \text{spread}[t], \quad t = \sqrt{n} \cdot \text{value} / \mathrm{std}(\text{spread}), \quad \text{DDOF}=1
    \end{aligned}
    $$

    Weights are **lagged by one sampled period per asset** by default
    (``lag_weights=True``): a portfolio rebalanced at date t uses the
    market-cap observed at the previous rebalance, not at t. Pairing
    contemporaneous ``market_cap[t]`` with ``forward_return[t→t+h]`` is
    a classic look-ahead trap — market cap measured on date t embeds
    news that the t→t+h return has not yet realized.

    Pass ``lag_weights=False`` only when the caller has **already**
    supplied a lagged weight column (e.g., prior-month-end cap) and
    wants the function to treat it as observed at t.

    Compare with equal-weighted ``quantile_spread``: if VW spread much
    smaller (e.g., < 1/3 of EW), the alpha is driven by small-cap assets
    and may not survive capacity / liquidity constraints.

    Args:
        data: Panel with ``date, asset_id, factor, forward_return,
            market_cap`` (or whatever ``weight_col`` names).
        weight_col: Column for value weighting (default ``market_cap``).
        lag_weights: When True (default), shift ``weight_col`` by 1
            period per asset (on the non-overlap-sampled frame) before
            weighting. When False, use weights as supplied.

    Returns:
        MetricResult with per-period mean VW spread, t-stat, and p-value.
        Short-circuits if ``weight_col`` is missing or post-sampling n <
        ``MIN_PORTFOLIO_PERIODS_HARD``.

    Notes:
        Per non-overlapping date ``t``, per bucket ``b in {bot, top}``::

            vw_b[t] = sum_{i in b} w[i, t-1] * return[i, t -> t+h]
                      / sum_{i in b} w[i, t-1]
            spread[t] = vw_top[t] - vw_bot[t]
            value = mean_t spread[t];  t = sqrt(n) * value / std(spread)

        factrix lags weights by one **sampled** period per asset by default
        (not one raw bar) so the lag aligns with the rebalance stride;
        contemporaneous ``weight × forward_return`` would embed look-ahead
        bias from market-cap moves that the forward return has not yet
        realized.

    References:
        [Hou-Xue-Zhang (2020)][hou-xue-zhang-2020]: ~65% of anomalies
        fail $|t| \geq 1.96$ once microcaps are mitigated via NYSE
        breakpoints and value weighting jointly.

    Examples:
        >>> import polars as pl
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import quantile_spread_vw
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... ).with_columns(pl.lit(1e6).alias("market_cap"))
        >>> result = quantile_spread_vw(panel, forward_periods=5, n_groups=5)
        >>> result.name == ""
        True
    """
    if weight_col not in data.columns:
        return _short_circuit_output(
            "quantile_spread_vw",
            "no_weight_column",
            missing_column=weight_col,
        )

    sampled = _sample_non_overlapping(data, forward_periods)
    if lag_weights:
        sampled = _lag_within_asset(sampled, weight_col)
    tie_ratio = _compute_tie_ratio(sampled, factor_col)
    _warn_high_tie_ratio(tie_ratio, "quantile_spread_vw", tie_policy)

    grouped = _assign_quantile_groups(
        sampled,
        factor_col,
        n_groups,
        tie_policy=tie_policy,
    )

    top_group = n_groups - 1
    bottom_group = 0

    # WHY: per-date weighted mean for top and bottom buckets
    vw_series = (
        grouped.with_columns(
            (pl.col(return_col) * pl.col(weight_col)).alias("_wr"),
        )
        .group_by("date")
        .agg(
            (
                pl.col("_wr").filter(pl.col("_group") == top_group).sum()
                / pl.col(weight_col).filter(pl.col("_group") == top_group).sum()
            ).alias("top_return_vw"),
            (
                pl.col("_wr").filter(pl.col("_group") == bottom_group).sum()
                / pl.col(weight_col).filter(pl.col("_group") == bottom_group).sum()
            ).alias("bottom_return_vw"),
        )
        .with_columns(
            (pl.col("top_return_vw") - pl.col("bottom_return_vw")).alias("spread_vw"),
        )
        .sort("date")
    )

    spread_vals = vw_series["spread_vw"].drop_nulls()
    n = len(spread_vals)
    sc = _enforce_scaled_floor(
        "quantile_spread_vw",
        data["date"].n_unique(),
        MIN_PORTFOLIO_PERIODS_HARD,
        forward_periods,
        "insufficient_portfolio_periods",
        tie_ratio=tie_ratio,
        tie_policy=tie_policy,
    )
    if sc is not None:
        return sc
    # Mirror the EW path (see ``_quantile_spread_from_series``): gate the
    # n_groups buckets on per-date valid factor counts, then treat a constant
    # factor as no-signal — otherwise value weighting manufactures an
    # ordering-artifact spread (ordinal ties) or empty-bucket NaN (average).
    per_date_assets = sampled.group_by("date").agg(
        pl.col(factor_col).count().alias("_n")
    )["_n"]
    if bool((per_date_assets < n_groups).all()):
        max_assets_value = per_date_assets.max()
        max_assets = 0 if max_assets_value is None else cast(int, max_assets_value)
        return _short_circuit_output(
            "quantile_spread_vw",
            "insufficient_assets_for_quantile_groups",
            n_obs=max_assets,
            n_obs_axis="assets",
            n_groups=n_groups,
            min_required=n_groups,
            max_assets_per_date=max_assets,
        )
    if _all_dates_degenerate(sampled, factor_col):
        return _no_signal_zero_variance(
            n,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
            n_groups=n_groups,
            weights_lagged=lag_weights,
        )

    arr = spread_vals.to_numpy()
    mean_spread = float(np.mean(arr))
    std_spread = float(np.std(arr, ddof=DDOF))
    t = _calc_t_stat(mean_spread, std_spread, n)

    p = _p_value_from_t(t, n)
    metadata: dict[str, object] = {
        "n_periods": n,
        "method": "non-overlapping t-test",
        "stat_type": "t",
        "h0": "mu=0",
        "tie_ratio": tie_ratio,
        "tie_policy": tie_policy,
        "weights_lagged": lag_weights,
    }
    warning_codes: list[str] = []
    _surface_null_drop(
        n_periods_in=vw_series.height,
        n_periods_out=n,
        drop_reason="null spread observations in the series",
        metric_name="quantile_spread_vw",
        metadata=metadata,
        warning_codes=warning_codes,
    )
    return MetricResult(
        p_value=p,
        value=mean_spread,
        n_obs=n,
        n_obs_axis="periods",
        stat=t,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
