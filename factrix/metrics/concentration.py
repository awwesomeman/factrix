"""Top-bucket concentration analysis for cross-sectional panels.

Measures whether top-bucket (long-leg) alpha is concentrated in a few
stocks or broadly distributed, using HHI (Herfindahl-Hirschman Index)
inverse.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import (
    DDOF,
    EPSILON,
    MIN_PORTFOLIO_PERIODS,
    ConcentrationWeight,
    MetricOutput,
)
from factrix.metrics._helpers import (
    _compute_tie_ratio,
    _sample_non_overlapping,
    _short_circuit_output,
)
from factrix._stats import _calc_t_stat, _p_value_from_t, _significance_marker


def top_concentration(
    df: pl.DataFrame,
    forward_periods: int = 5,
    q_top: float = 0.2,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    weight_by: ConcentrationWeight = "abs_factor",
) -> MetricOutput:
    """Top-bucket concentration via HHI inverse.

    Per date, selects top ``q_top`` stocks by factor rank, computes
    HHI of their weights, and returns 1/HHI as the effective number of
    independent bets.

    Args:
        df: Panel with ``date, asset_id, factor`` (and ``forward_return``
            if ``weight_by="alpha_contribution"``).
        q_top: Fraction of top-ranked stocks to include (default 0.2 =
            top 20%).
        weight_by: HHI weight convention.
            - ``"abs_factor"`` (default): weight by ``|factor|``. Answers
              "how concentrated is the signal itself in the top bucket".
              Conservative, signal-level.
            - ``"alpha_contribution"``: weight by the magnitude of each
              name's realised contribution ``|sign(factor) · forward_return|``.
              Captures **risk-concentration**: the top bucket's realised
              return is dominated by a few outliers. Note the absolute
              value — a single big *winner* and a single big *loser*
              both register as concentration, which is the right
              framing for risk but NOT for signed-alpha attribution.
              If you need the latter, apply HHI downstream on the
              signed ``sign(factor) · forward_return`` series yourself.

    Returns:
        MetricOutput with value = mean(1/HHI) across dates.
        Higher = more diversified top bucket.

    Notes:
        Uses ``rank(method="average")`` internally for the top-bucket
        cutoff — tie_policy from Config does not apply here because HHI
        measures concentration *among* the selected stocks, not their
        bucketing. ``tie_ratio`` is still recorded in metadata as a
        data-quality diagnostic (high tie_ratio → unstable top-bucket
        membership across re-rankings).
    """
    if weight_by == "alpha_contribution" and return_col not in df.columns:
        return _short_circuit_output(
            "top_concentration", "no_return_column",
            missing_column=return_col, weight_by=weight_by,
        )

    filtered = _sample_non_overlapping(df, forward_periods)
    tie_ratio = _compute_tie_ratio(filtered, factor_col)

    q1 = (
        filtered.with_columns(
            (
                pl.col(factor_col).rank(method="average").over("date")
                / pl.len().over("date")
            ).alias("_pct_rank")
        )
        .filter(pl.col("_pct_rank") >= (1 - q_top))
    )

    if weight_by == "alpha_contribution":
        weighted = q1.with_columns(
            (
                pl.col(factor_col).sign() * pl.col(return_col)
            ).abs().alias("_raw_weight")
        )
    else:
        weighted = q1.with_columns(
            pl.col(factor_col).abs().alias("_raw_weight")
        )

    hhi_per_date = (
        weighted.with_columns(
            (pl.col("_raw_weight") / pl.col("_raw_weight").sum().over("date"))
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
        return _short_circuit_output(
            "top_concentration", "insufficient_portfolio_periods",
            n_observed=len(hhi_per_date), min_required=MIN_PORTFOLIO_PERIODS,
            tie_ratio=tie_ratio,
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
            "tie_ratio": tie_ratio,
            "weight_by": weight_by,
        },
    )
