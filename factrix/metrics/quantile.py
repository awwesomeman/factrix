"""Quantile analysis for cross-sectional panels.

All spread series are time-indexed (``date, value``) and can be fed
into any ``series/`` tool.

Notes:
    **Pipeline.** Per-period long-short spread on quantile groups
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
    DEFAULT_FORWARD_PERIODS,
    DEFAULT_N_GROUPS,
    MIN_PORTFOLIO_PERIODS_HARD,
)
from factrix.inference import NEWEY_WEST, NON_OVERLAPPING, NeweyWest, NonOverlapping
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _all_dates_degenerate,
    _assign_quantile_groups,
    _check_applicable_inference,
    _compute_tie_ratio,
    _degenerate_test_fields,
    _enforce_scaled_floor,
    _finite_expr,
    _finite_values,
    _is_thin_quantile_groups,
    _lag_within_asset,
    _no_signal_zero_variance,
    _sample_non_overlapping,
    _scaled_periods_threshold,
    _short_circuit_output,
    _spread_significance_with_inference,
    _surface_null_drop,
    _warn_high_tie_ratio,
    _warn_thin_quantile_groups,
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


def _median_finite_cross_section(panel: pl.DataFrame, factor_col: str) -> int:
    """Median per-period count of finite ``factor_col`` values.

    The size of the cross-section that is actually ranked on a typical date
    — not ``asset_id.n_unique()`` over the whole panel, which counts every
    name that ever appeared. The ``FEW_ASSETS`` advisory keys on this: how
    many names back a single date's bucket mean is a per-period quantity, so a
    12-name-per-period universe that rotated through 200 tickers over the
    sample is thin, not wide.
    """
    if panel.is_empty():
        return 0
    per_period = panel.group_by("date").agg(_finite_expr(factor_col).sum().alias("_n"))[
        "_n"
    ]
    med = per_period.median()
    return 0 if med is None else int(med)  # type: ignore[arg-type]


def _excess_leg_test(arr: np.ndarray) -> tuple[float, float, float]:
    """``(mean, t, p)`` for a long/short excess leg, NaN-safe on an empty leg.

    Descriptive metadata only — the headline test is the spread, not a leg —
    so a degenerate leg is reported as ``t = p = NaN`` (``_calc_t_stat``'s
    not-computable value) rather than short-circuiting the whole metric.
    """
    n_leg = int(arr.size)
    if n_leg == 0:
        nan = float("nan")
        return nan, nan, nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=DDOF))
    t_leg = _calc_t_stat(mean, std, n_leg)
    return mean, t_leg, _p_value_from_t(t_leg, n_leg)


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
    forward_periods: int = DEFAULT_FORWARD_PERIODS,
    n_groups: int = DEFAULT_N_GROUPS,
    factor_cols: Sequence[str] = ("factor",),
    tie_policy: str = "ordinal",
    inference: NonOverlapping | NeweyWest = NON_OVERLAPPING,
    *,
    expected_warnings: tuple[str, ...] = (),
    _precomputed_series: dict[str, pl.DataFrame] | None = None,
) -> dict[str, MetricResult]:
    """long-short spread (per-period mean).

    Args:
        inference: Headline significance method on the per-period spread.
            ``fx.inference.NON_OVERLAPPING`` (default) runs the OLS t-test
            on the non-overlap stride subsample; ``fx.inference.NEWEY_WEST``
            keeps every date and absorbs the MA(h-1) overlap in a HAC SE.
            On a small cross-section (median per-period finite-factor count
            ``< 30``) the heavy-tail block bootstrap takes precedence over
            either (see Notes).
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
        A thin cross-section (``median_cross_section < MIN_ASSETS_WARN``)
        attaches ``FEW_ASSETS`` on either route and changes nothing else.

        Long/short alpha decomposition stays a descriptive OLS t-test on
        ``top_return - universe_return`` and ``universe_return -
        bottom_return`` regardless of ``inference`` — it attributes the
        spread to long-side vs short-side excess, it is not the headline H0.
        Each leg is tested on its own finite sample (``n_periods_long_leg``
        / ``n_periods_short_leg``), which can be shorter than the spread
        sample when only one bucket was empty on a date.

        **Which sample each count describes.** ``value``, ``stat``,
        ``p_value`` and ``n_obs`` always come from the *same* sample — the
        one the selected inference ran on:

        - ``n_obs`` == ``metadata["n_periods"]``: the periods the headline
          test used. Under ``NON_OVERLAPPING`` (and under any bootstrap
          override) that is the strided series; under ``NEWEY_WEST`` it is
          the full overlapping series, which is ~``forward_periods`` times
          longer — the HAC test never ran on the strided sample, so
          reporting the strided count beside a HAC p-value would misstate
          the test's degrees of freedom.
        - ``metadata["n_periods_strided"]``: the non-overlap sample,
          always present so the two are comparable.
        - ``metadata["n_periods_full"]``: the overlapping sample (HAC path
          only).
        - ``metadata["n_dropped"]`` / ``n_periods_in`` / ``n_periods_out``:
          the null- and NaN-drop bookkeeping on the **strided** series.

        **Non-finite observations.** Every series column consumed here is
        filtered with ``drop_nulls().drop_nans()``: polars' ``drop_nulls``
        keeps float NaN, and one NaN in the spread would make the t-path
        report ``degenerate_variance`` (mislabelling missing data as a
        dispersion-free sample) or raise in the bootstrap path.

        **Thin cross-sections** are judged by the median *per-period* count
        of finite factor values (``metadata["median_cross_section"]``),
        not by the number of distinct ``asset_id`` values in the panel: the
        bootstrap fires because a single date's bucket mean rests on few
        names, which a rotating universe with a large lifetime asset count
        does not fix.

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
            expected_warnings=expected_warnings,
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
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    """Per-factor t-test pipeline shared by single and batch paths.

    ``sampled`` is the (already non-overlap-sampled) panel; ``series``
    is this factor's spread DataFrame; ``n_raw_periods`` is the full date count
    before sampling. Splitting this out lets the batch path share the sample
    step across every factor while the single-factor path stays a one-liner.
    """
    tie_ratio = _compute_tie_ratio(sampled, factor_col)
    _warn_high_tie_ratio(tie_ratio, "quantile_spread", tie_policy)
    spread_vals = _finite_values(series["spread"])
    n_strided = len(spread_vals)
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
    if n_strided == 0:
        # Every sampled date lost its spread to a null / NaN bucket mean; there
        # is nothing to average, so refuse rather than emit ``mean([]) = nan``.
        return _short_circuit_output(
            "quantile_spread",
            "insufficient_portfolio_periods",
            n_obs=0,
            n_obs_axis="periods",
            n_periods_in=series.height,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
            n_groups=n_groups,
        )

    arr = spread_vals.to_numpy()
    # Headline test: ``inference`` selects non-overlap t vs Newey-West HAC.
    # ``mean_spread`` is the full-sample mean under HAC, the non-overlap mean
    # otherwise. The FEW_ASSETS advisory keys on the median *per-period*
    # finite-factor count, not the universe-wide unique asset count: how many
    # names back one bucket mean is a per-period quantity, so a rotating
    # universe must not read as wide.
    n_assets = _median_finite_cross_section(sampled, factor_col)
    # Non-finite spreads must not reach the HAC regression either — same reason
    # the strided array is cleaned above.
    clean_full_series = (
        full_series.filter(_finite_expr("spread")) if full_series is not None else None
    )
    mean_spread, t, p, sig_method, sig_extra, sig_codes = (
        _spread_significance_with_inference(
            inference,
            strided_spread=arr,
            full_spread=clean_full_series,
            forward_periods=forward_periods,
            n_assets=n_assets,
        )
    )
    # Sample the headline stat/p actually ran on: the full overlapping series on
    # the HAC path, the strided one otherwise. ``value``/``stat``/``p_value``/
    # ``n_obs`` must all describe it.
    n = int(
        cast(
            int,
            sig_extra.get(
                "n_periods_tested", sig_extra.get("n_periods_full", n_strided)
            ),
        )
    )

    # Long/short decomposition (spread = long_alpha + short_alpha)
    long_arr = _finite_values(
        series["top_return"] - series["universe_return"]
    ).to_numpy()
    short_arr = _finite_values(
        series["universe_return"] - series["bottom_return"]
    ).to_numpy()

    mean_long, t_long, p_long = _excess_leg_test(long_arr)
    mean_short, t_short, p_short = _excess_leg_test(short_arr)

    metadata: dict[str, object] = {
        "n_periods": n,
        "n_periods_strided": n_strided,
        "n_periods_long_leg": int(long_arr.size),
        "n_periods_short_leg": int(short_arr.size),
        "median_cross_section": n_assets,
        "n_groups": n_groups,
        "forward_periods": forward_periods,
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
    # Drop stats describe the strided series this consumer collapsed, whatever
    # sample the headline test ended up running on.
    _surface_null_drop(
        n_periods_in=series.height,
        n_periods_out=n_strided,
        drop_reason="null / NaN value observations in the series",
        metric_name="quantile_spread",
        metadata=metadata,
        warning_codes=warning_codes,
        expected_warnings=expected_warnings,
    )
    # A NaN headline stat means the tested spread series carries no dispersion
    # (or the HAC SE collapsed): ``mean_spread`` still stands, the t does not.
    stat, p_out, alternative = _degenerate_test_fields(
        t, p, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=mean_spread,
        n_obs=n,
        n_obs_axis="periods",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


def _vw_spread_series(
    panel: pl.DataFrame,
    *,
    factor_col: str,
    return_col: str,
    weight_col: str,
    n_groups: int,
    tie_policy: str,
) -> pl.DataFrame:
    """Per-period value-weighted top-minus-bottom spread for ``panel``.

    Returns ``date, _n_assets, _n_unique, top_return_vw, bottom_return_vw,
    spread_vw``. Split out of the metric body so the non-overlap path (strided
    panel) and the HAC path (full overlapping panel) build the series the same
    way, exactly as ``compute_spread_series`` serves both paths of the
    equal-weighted sibling.
    """
    grouped = _assign_quantile_groups(
        panel,
        factor_col,
        n_groups,
        tie_policy=tie_policy,
    )

    top_group = n_groups - 1
    bottom_group = 0

    # A weighted mean is ``sum(w·r) / sum(w)``, and the two sums must run over
    # the *same* names. A name with a missing return contributes nothing to the
    # numerator (polars ``sum`` skips nulls) but its weight would still sit in
    # the denominator, shrinking the bucket return toward 0 in proportion to how
    # much of the bucket is missing. Masking the weight to null wherever the
    # return (or the weight itself) is non-finite keeps both sums on the same
    # sample.
    finite_return = _finite_expr(return_col)
    finite_weight = _finite_expr(weight_col)
    finite_factor = _finite_expr(factor_col)

    def _bucket_vw(group: int) -> pl.Expr:
        """Weighted bucket return, null when no name in it carries a weight."""
        in_bucket = pl.col("_group") == group
        w_sum = pl.col("_w").filter(in_bucket).sum()
        wr_sum = pl.col("_wr").filter(in_bucket).sum()
        # ``sum()`` of an all-null column is 0, not null, so an empty or
        # fully-missing bucket would otherwise divide 0/0 -> NaN, or (worse)
        # surface as a manufactured 0.0 return. Emit null instead: no
        # observation, not a zero return.
        return pl.when(w_sum.is_not_null() & (w_sum != 0)).then(wr_sum / w_sum)

    # WHY: per-period weighted mean for top and bottom buckets
    return (
        grouped.with_columns(
            pl.when(finite_return & finite_weight).then(pl.col(weight_col)).alias("_w"),
        )
        .with_columns((pl.col(return_col) * pl.col("_w")).alias("_wr"))
        .group_by("date")
        .agg(
            pl.col(factor_col).filter(finite_factor).len().alias("_n_assets"),
            pl.col(factor_col).filter(finite_factor).n_unique().alias("_n_unique"),
            _bucket_vw(top_group).alias("top_return_vw"),
            _bucket_vw(bottom_group).alias("bottom_return_vw"),
        )
        .with_columns(
            pl.when((pl.col("_n_assets") > 0) & (pl.col("_n_unique") <= 1))
            .then(pl.lit(0.0))
            .otherwise(pl.col("top_return_vw") - pl.col("bottom_return_vw"))
            .alias("spread_vw"),
        )
        .sort("date")
    )


@metric(
    cell=_Q_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    sample_threshold=_quantile_groups_threshold,
)
def quantile_spread_vw(
    data: pl.DataFrame,
    forward_periods: int = DEFAULT_FORWARD_PERIODS,
    n_groups: int = DEFAULT_N_GROUPS,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    weight_col: str = "market_cap",
    tie_policy: str = "ordinal",
    inference: NonOverlapping | NeweyWest = NON_OVERLAPPING,
    lag_weights: bool = True,
    expected_warnings: tuple[str, ...] = (),
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
        inference: Headline significance method on the per-period spread —
            the same knob, allowlist and default as ``quantile_spread``, so
            the equal-weighted / value-weighted pair is tested the same way on
            the same date set. ``fx.inference.NON_OVERLAPPING`` (default) runs
            the OLS t-test on the non-overlap stride subsample;
            ``fx.inference.NEWEY_WEST`` keeps every date and absorbs the
            MA(h-1) overlap in a HAC SE.
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
        (not one raw bar) so the lag aligns with the rebalance stride. Under
        ``inference=NEWEY_WEST`` there is no stride — every date is its own
        rebalance — so the lag is one bar there, and the first date drops out
        for want of a lagged weight;
        contemporaneous ``weight × forward_return`` would embed look-ahead
        bias from market-cap moves that the forward return has not yet
        realized.

        **Missing names leave the weighted mean entirely.** The sums in
        ``vw_b[t]`` run over the names of bucket ``b`` that have *both* a
        finite weight and a finite return. A name missing either is dropped
        from the numerator **and** the denominator — keeping its weight in
        the denominator only (the naive ``sum(w·r)/sum(w)``) would pull the
        bucket return toward zero in proportion to the missing share, so a
        half-reported bucket would report half its true return.

        The alternative convention re-normalises nothing and treats a
        missing return as ``0`` — defensible for a portfolio that really
        did hold cash, wrong for a panel where the return is merely
        unobserved. factrix assumes the latter: the bucket return is the
        value-weighted mean of what was observed.

        When a leg has **no** weighted name at all, its bucket return is
        null and the date's spread is null and excluded from ``n_obs`` —
        not ``0.0``. ``sum()`` of an all-null polars column is ``0``, so
        the unguarded ratio would manufacture a ``0.0`` spread on a fully
        missing date and count it as a real observation, biasing both the
        mean and the t-stat toward zero.

        **Thin-cross-section diagnostics.** This path used to build its
        buckets inline and call the t-test directly, so it emitted none of
        the diagnostics its equal-weighted sibling does: on an 8-name panel
        cut into 5 buckets — 1.6 names per leg — ``quantile_spread``
        reported ``('few_assets', 'thin_quantile_groups')`` and a
        ``UserWarning``, while ``quantile_spread_vw`` reported clean. That
        is exactly backwards for the metric whose purpose is a capacity /
        robustness cross-check. It now routes through the same headline
        chokepoint (``FEW_ASSETS`` on the median per-period finite-factor
        count) and raises the same thin-bucket advisory and
        ``THIN_QUANTILE_GROUPS`` code off the same threshold.

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

    _check_applicable_inference(
        inference, applicable_inference, func_name="quantile_spread_vw"
    )

    sampled = _sample_non_overlapping(data, forward_periods)
    if lag_weights:
        sampled = _lag_within_asset(sampled, weight_col)
    tie_ratio = _compute_tie_ratio(sampled, factor_col)
    _warn_high_tie_ratio(tie_ratio, "quantile_spread_vw", tie_policy)
    # Same dual-channel thin-bucket advisory the equal-weighted sibling gets,
    # off the same threshold: this metric exists as a capacity / robustness
    # cross-check, so it must not be the one that reports clean on a panel
    # whose legs hold a single name each.
    _warn_thin_quantile_groups(sampled, n_groups)

    vw_series = _vw_spread_series(
        sampled,
        factor_col=factor_col,
        return_col=return_col,
        weight_col=weight_col,
        n_groups=n_groups,
        tie_policy=tie_policy,
    )

    spread_vals = _finite_values(vw_series["spread_vw"])
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
    # n_groups buckets on per-period valid factor counts, then treat a constant
    # factor as no-signal — otherwise value weighting manufactures an
    # ordering-artifact spread (ordinal ties) or empty-bucket NaN (average).
    per_date_assets = vw_series["_n_assets"]
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
    if n == 0:
        # Every sampled date lost its VW spread (no weighted name in a leg);
        # nothing to average.
        return _short_circuit_output(
            "quantile_spread_vw",
            "insufficient_portfolio_periods",
            n_obs=0,
            n_obs_axis="periods",
            n_periods_in=vw_series.height,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
            n_groups=n_groups,
            weights_lagged=lag_weights,
        )

    arr = spread_vals.to_numpy()
    # Same headline chokepoint as the equal-weighted sibling, so the pair is
    # tested the same way on the same date set: ``NON_OVERLAPPING`` reproduces
    # the previous strided t bit-for-bit, ``NEWEY_WEST`` keeps every date and
    # HAC-corrects the MA(h-1) overlap. The FEW_ASSETS advisory rides along on
    # the median per-period finite-factor count.
    n_assets = _median_finite_cross_section(sampled, factor_col)
    full_series = None
    if isinstance(inference, NeweyWest):
        full_panel = _lag_within_asset(data, weight_col) if lag_weights else data
        full_series = _vw_spread_series(
            full_panel,
            factor_col=factor_col,
            return_col=return_col,
            weight_col=weight_col,
            n_groups=n_groups,
            tie_policy=tie_policy,
        ).select("date", pl.col("spread_vw").alias("spread"))
        full_series = full_series.filter(_finite_expr("spread"))

    mean_spread, t, p, sig_method, sig_extra, sig_codes = (
        _spread_significance_with_inference(
            inference,
            strided_spread=arr,
            full_spread=full_series,
            forward_periods=forward_periods,
            n_assets=n_assets,
        )
    )
    n_tested = int(
        cast(
            int,
            sig_extra.get("n_periods_tested", sig_extra.get("n_periods_full", n)),
        )
    )
    metadata: dict[str, object] = {
        "n_periods": n_tested,
        "n_periods_strided": n,
        "median_cross_section": n_assets,
        "n_groups": n_groups,
        "forward_periods": forward_periods,
        "method": sig_method,
        "stat_type": "t",
        "h0": "mu=0",
        "tie_ratio": tie_ratio,
        "tie_policy": tie_policy,
        "weights_lagged": lag_weights,
        **sig_extra,
    }
    warning_codes = list(sig_codes)
    # Structured twin of the thin-bucket advisory raised above.
    if _is_thin_quantile_groups(sampled, n_groups):
        warning_codes.append(WarningCode.THIN_QUANTILE_GROUPS.value)
    # Drop stats describe the strided series this consumer collapsed, whatever
    # sample the headline test ended up running on.
    _surface_null_drop(
        n_periods_in=vw_series.height,
        n_periods_out=n,
        drop_reason="null / NaN value observations in the series",
        metric_name="quantile_spread_vw",
        metadata=metadata,
        warning_codes=warning_codes,
        expected_warnings=expected_warnings,
    )
    # A NaN headline stat means the tested spread series carries no dispersion
    # (or the HAC SE collapsed): ``mean_spread`` still stands, the t does not.
    stat, p_out, alternative = _degenerate_test_fields(
        t, p, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=mean_spread,
        n_obs=n_tested,
        n_obs_axis="periods",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
