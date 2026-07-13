"""Monotonicity test for cross-sectional panels.

Measures whether factor quantile groups exhibit monotonic return ordering.
Per-date: split into n_groups by factor rank, compute mean return per group,
Spearman corr between group index and return.

Notes:
    **Pipeline.** Per-date Spearman corr between quantile index and
    group mean return (cross-section step), then non-overlapping
    cross-asset t on the per-date series.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import scipy.stats as scipy_stats

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._metric_index import cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat, _p_value_from_t
from factrix._types import (
    DDOF,
    MIN_MONOTONICITY_PERIODS_HARD,
)
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _assign_quantile_groups_batch,
    _enforce_scaled_floor,
    _sample_non_overlapping,
    _scaled_periods_threshold,
    _short_circuit_output,
    _warn_high_tie_ratio,
)

__all__ = [
    "monotonicity",
]


# Slice-test contract: monotonicity buckets the
# cross-section into `n_groups` (default 10) and computes Spearman ρ
# across per-bucket means. Patton & Timmermann (2010) "Monotonicity
# in Asset Returns" recommend ≥ 50 assets per bucket so the per-date
# bucket means converge to their cross-sectional expectation; below
# this floor individual-asset noise dominates the rank statistic.
# `_downscale_n_groups(base, n_assets, min_assets_per_group=50)` caps
# `n_groups` accordingly inside the slice-test function.
min_assets_per_group: int | None = 50


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    batchable=True,
    # Periods floor scales with the non-overlap stride (see ``quantile``): the
    # per-date Spearman series is sub-sampled at ``forward_periods``, so
    # pre-flight and the in-body gate share ``MIN_MONOTONICITY_PERIODS_HARD`` +
    # ``_scaled_min_periods``.
    sample_threshold=_scaled_periods_threshold(MIN_MONOTONICITY_PERIODS_HARD),
)
def monotonicity(
    data: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 10,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
    tie_policy: str = "ordinal",
) -> dict[str, MetricResult]:
    """Quantile return monotonicity (Spearman correlation).

    ``value`` = mean |Spearman| — magnitude of monotonicity (always ≥ 0).
    ``t_stat`` = t-test on signed Spearman — whether direction is consistent.

    A high ``value`` with insignificant ``t_stat`` means the factor has
    strong monotonicity but the direction flips across dates.

    Args:
        data: Panel with ``date, asset_id, factor, forward_return``.
        n_groups: Number of quantile groups (default 10 for Taiwan ~2000 stocks).
            Use 5 for ``n_assets < 1000``, 3 for ``n_assets < 200``.
        tie_policy: Bucketing tie-break policy, see ``_assign_quantile_groups``.

    Returns:
        MetricResult with value = mean |Spearman(group_idx, group_return)|.

    Notes:
        Per non-overlap date ``t``, bucket assets into ``n_groups`` by
        factor rank and compute ``mono_t = Spearman(group_idx,
        group_mean_return)``. ``value = mean_t |mono_t|`` (magnitude of
        monotonicity, ≥ 0); ``t-stat = mean(mono) / (std(mono) /
        sqrt(n))`` on the signed series tests directional consistency.

        factrix splits magnitude (``value``) and direction (``stat``)
        deliberately: a high ``value`` paired with insignificant ``t``
        means the factor monotonically discriminates returns but its sign
        flips across dates — useful information that a single signed
        average would hide.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.monotonicity import monotonicity
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=200, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = monotonicity(panel, forward_periods=5, n_groups=5)
        >>> result["factor"].name == ""
        True
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")

    # Raw (pre-sampling) date count: the axis the stride-scaled periods floor is
    # calibrated against, shared across all factors.
    n_raw_periods = data["date"].n_unique()

    # Sample non-overlapping once — shared across all factors on the
    # same panel (depends only on `date` + `forward_periods`).
    filtered = _sample_non_overlapping(data, forward_periods)
    tie_ratios = _compute_tie_ratios_batch(filtered, cols)
    for f in cols:
        _warn_high_tie_ratio(tie_ratios[f], "monotonicity", tie_policy)

    grouped = _assign_quantile_groups_batch(filtered, cols, n_groups, tie_policy)

    # Stage 1: per-(date, factor, group) mean return, expressed as one
    # ``group_by("date").agg(...)`` carrying N × n_groups filter+mean
    # expressions. Wide output: (n_dates, 1 + N * n_groups). For large
    # F × n_groups this is a tall ask of the planner, but a single
    # ``collect()`` beats N separate per-factor queries because the
    # rank / group columns get computed once and the panel is scanned
    # only once.
    agg_exprs: list[pl.Expr] = [
        pl.col(return_col)
        .filter(pl.col(f"_group__{f}") == g)
        .mean()
        .alias(f"_gr__{f}__{g}")
        for f in cols
        for g in range(n_groups)
    ]
    group_returns_wide = (
        grouped.lazy().group_by("date").agg(agg_exprs).sort("date").collect()
    )

    # Stage 2: per-(factor, date) Spearman ρ between group index and
    # the rank of the per-group mean return. Done in numpy because the
    # inner shape (n_dates rows × n_groups cells per factor) is small
    # and the operation is uniform — vectorising in numpy beats the
    # polars-side rank+corr pipeline at this size. Materialise the
    # full (n_dates, N * n_groups) block once (zero-copy via Arrow)
    # and slice per factor — saves N * n_groups individual ``.to_numpy``
    # calls.
    gr_col_names = [f"_gr__{f}__{g}" for f in cols for g in range(n_groups)]
    all_means = group_returns_wide.select(gr_col_names).to_numpy()
    group_idx = np.arange(n_groups, dtype=np.float64)
    group_idx_centered = group_idx - group_idx.mean()
    group_idx_norm = float(np.sqrt(np.sum(group_idx_centered**2)))
    results: dict[str, MetricResult] = {}
    for i, f in enumerate(cols):
        mat = all_means[:, i * n_groups : (i + 1) * n_groups]
        # Drop dates with any null/nan bucket mean (matches the
        # original filter `n == n_groups` and `mono.is_not_null`).
        mat = mat[np.all(np.isfinite(mat), axis=1)]
        if mat.shape[0] == 0:
            mono_arr = np.empty(0)
        else:
            # Spearman = Pearson(group_idx, rank(group_mean)).
            ranks = scipy_stats.rankdata(mat, axis=1, method="average")
            ranks_centered = ranks - ranks.mean(axis=1, keepdims=True)
            ranks_norm = np.sqrt(np.sum(ranks_centered**2, axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                mono_arr = (ranks_centered @ group_idx_centered) / (
                    ranks_norm * group_idx_norm
                )
            mono_arr = mono_arr[np.isfinite(mono_arr)]

        sc = _enforce_scaled_floor(
            "monotonicity",
            n_raw_periods,
            MIN_MONOTONICITY_PERIODS_HARD,
            forward_periods,
            "insufficient_monotonicity_periods",
            n_groups=n_groups,
            tie_ratio=tie_ratios[f],
            tie_policy=tie_policy,
        )
        if sc is not None:
            results[f] = sc
            continue
        if len(mono_arr) == 0:
            # n_raw_periods cleared the scaled floor above, but every
            # sampled date had a null bucket mean for this factor (e.g. a
            # sparse column), leaving nothing to correlate.
            results[f] = _short_circuit_output(
                "monotonicity",
                "insufficient_monotonicity_periods",
                n_obs=0,
                n_obs_axis="periods",
                n_groups=n_groups,
                tie_ratio=tie_ratios[f],
                tie_policy=tie_policy,
            )
            continue
        avg_mono = float(np.mean(np.abs(mono_arr)))
        mean_mono = float(np.mean(mono_arr))
        std_mono = float(np.std(mono_arr, ddof=DDOF))
        t = _calc_t_stat(mean_mono, std_mono, len(mono_arr))
        p = _p_value_from_t(t, len(mono_arr))
        results[f] = MetricResult(
            p_value=p,
            alternative="two-sided",
            value=avg_mono,
            n_obs=len(mono_arr),
            n_obs_axis="periods",
            stat=t,
            metadata={
                "method": "t-test on per-period signed monotonicity",
                "stat_type": "t",
                "h0": "mu=0",
                "mean_signed": mean_mono,
                "n_valid_periods": len(mono_arr),
                "n_groups": n_groups,
                "tie_ratio": tie_ratios[f],
                "tie_policy": tie_policy,
            },
        )

    return results


def _compute_tie_ratios_batch(
    data: pl.DataFrame, factor_cols: list[str]
) -> dict[str, float]:
    """Median-across-dates tie ratio (``1 - n_unique / n``) for many factors.

    The single-factor :func:`_compute_tie_ratio` runs a separate polars
    aggregation per factor; this batches them into one ``group_by("date")`` so
    the sampled panel is scanned once for any number of factors. The tie ratio
    is **per date** then median-reduced — the same statistic the single-factor
    helper returns. Computing it globally (``n_unique`` / ``len`` over the whole
    frame) would conflate cross-sectional ties with values merely repeating
    across dates, inflating the ratio toward 1 and tripping spurious
    high-tie-ratio warnings on a continuous factor.
    """
    if not factor_cols:
        return {}
    per_date = data.group_by("date").agg(
        pl.len().alias("_n"),
        *[pl.col(f).n_unique().alias(f"_u__{f}") for f in factor_cols],
    )
    # ``median`` over the (possibly empty) per-date ratio yields ``None`` on an
    # empty frame, which maps to ``nan`` below — the same empty-panel contract as
    # the single-factor :func:`_compute_tie_ratio`, no separate guard needed.
    medians = per_date.select(
        (1.0 - pl.col(f"_u__{f}") / pl.col("_n")).median().alias(f"_tr__{f}")
        for f in factor_cols
    ).row(0, named=True)
    return {
        f: float("nan") if medians[f"_tr__{f}"] is None else float(medians[f"_tr__{f}"])
        for f in factor_cols
    }
