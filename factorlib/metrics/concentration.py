"""Q1 concentration analysis for cross-sectional panels.

Measures whether Q1 (top quantile) alpha is concentrated in a few stocks
or broadly distributed, using HHI (Herfindahl-Hirschman Index) inverse.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import DDOF, EPSILON, MIN_PORTFOLIO_PERIODS, MetricOutput
from factorlib.metrics._helpers import _sample_non_overlapping
from factorlib._stats import _calc_t_stat, _p_value_from_t, _significance_marker


def top_concentration(
    df: pl.DataFrame,
    forward_periods: int = 5,
    q_top: float = 0.2,
    factor_col: str = "factor",
) -> MetricOutput:
    """Q1 concentration via HHI inverse.

    Per date, selects top ``q_top`` stocks by factor rank, computes
    HHI of their (absolute) factor values, and returns 1/HHI as
    the effective number of independent bets.

    Args:
        df: Panel with ``date, asset_id, factor``.
        q_top: Fraction of stocks in Q1 (default 0.2 = top 20%).

    Returns:
        MetricOutput with value = mean(1/HHI) across dates.
        Higher = more diversified Q1.
    """
    filtered = _sample_non_overlapping(df, forward_periods)

    q1 = (
        filtered.with_columns(
            (
                pl.col(factor_col).rank(method="average").over("date")
                / pl.len().over("date")
            ).alias("_pct_rank")
        )
        .filter(pl.col("_pct_rank") >= (1 - q_top))
    )

    hhi_per_date = (
        q1.with_columns(
            pl.col(factor_col).abs().alias("_abs_factor")
        )
        .with_columns(
            (pl.col("_abs_factor") / pl.col("_abs_factor").sum().over("date"))
            .alias("_weight")
        )
        .group_by("date")
        .agg(
            (pl.col("_weight") ** 2).sum().alias("hhi"),
            pl.len().alias("n_top"),
        )
        .filter(pl.col("hhi") > EPSILON)
        .with_columns(
            (1.0 / pl.col("hhi")).alias("eff_n")
        )
        .sort("date")
    )

    if len(hhi_per_date) < MIN_PORTFOLIO_PERIODS:
        return MetricOutput(
            name="top_concentration", value=0.0, stat=0.0, significance="",
            metadata={
                "reason": "insufficient_portfolio_periods",
                "n_observed": len(hhi_per_date),
                "min_required": MIN_PORTFOLIO_PERIODS,
            },
        )

    eff_n_arr = hhi_per_date["eff_n"].to_numpy()
    n_top_arr = hhi_per_date["n_top"].to_numpy()
    mean_eff_n = float(np.mean(eff_n_arr))
    mean_n_top = float(np.mean(n_top_arr))
    ratio = mean_eff_n / max(mean_n_top, 1)

    # WHY: t-stat tests H₀: ratio ≥ 0.5 (well-diversified).
    # Per-date ratio = eff_n / n_top; if mean ratio < 0.5 with significant t,
    # alpha is concentrated in a few stocks.
    ratio_arr = eff_n_arr / np.maximum(n_top_arr, 1)
    n = len(ratio_arr)
    mean_ratio = float(np.mean(ratio_arr))
    std_ratio = float(np.std(ratio_arr, ddof=DDOF))
    # Test H₀: ratio ≥ 0.5 → shift by 0.5 then use standard t-test
    t = _calc_t_stat(mean_ratio - 0.5, std_ratio, n)

    # WHY: one-sided test → p = P(T < t), not two-sided
    p = _p_value_from_t(t, n, alternative="less")
    return MetricOutput(
        name="top_concentration",
        value=mean_eff_n,
        stat=t,
        significance=_significance_marker(p),
        metadata={
            "p_value": p,
            "stat_type": "t",
            "h0": "ratio>=0.5",
            "method": "one-sided t-test on ratio",
            "mean_n_top": mean_n_top,
            "ratio_eff_to_total": ratio,
        },
    )
