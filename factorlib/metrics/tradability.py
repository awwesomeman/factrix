"""Tradability metrics: Turnover, Breakeven Cost, Net Spread.

These are implementation-feasibility indicators, not factor quality
measures — they belong in Profile, not in Gates.

Input for Turnover: DataFrame with ``date, asset_id, factor``.
Input for Breakeven/Net Spread: pre-computed spread and turnover values.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import DDOF, EPSILON, MetricOutput
from factorlib._stats import _significance_marker


def turnover(
    df: pl.DataFrame,
    factor_col: str = "factor",
) -> MetricOutput:
    """Factor turnover via rank autocorrelation.

    ``turnover = 1 - mean(rank_autocorrelation)``

    High rank autocorrelation = low turnover = lower rebalance cost.

    Args:
        df: Panel with ``date, asset_id, factor``.

    Returns:
        MetricOutput with value = turnover estimate (0-1).
    """
    dates = df["date"].unique().sort()
    if len(dates) < 2:
        return MetricOutput(
            name="turnover", value=float("nan"),
            metadata={
                "reason": "insufficient_dates",
                "n_observed": len(dates),
                "min_required": 2,
            },
        )

    date_map = pl.DataFrame({
        "date": dates[1:],
        "prev_date": dates[:-1],
    })

    ranked = df.select(
        "date", "asset_id",
        pl.col(factor_col).rank(method="average").over("date").alias("factor_rank"),
    )

    paired = (
        ranked.rename({"factor_rank": "rank_curr"})
        .join(date_map, on="date")
        .join(
            ranked.rename({"date": "prev_date", "factor_rank": "rank_prev"}),
            on=["prev_date", "asset_id"],
        )
    )

    rc_per_date = (
        paired.group_by("date")
        .agg(pl.corr("rank_curr", "rank_prev").alias("rc"))
        .filter(pl.col("rc").is_not_null() & pl.col("rc").is_not_nan())
        .sort("date")
    )

    if rc_per_date.is_empty():
        return MetricOutput(
            name="turnover", value=float("nan"),
            metadata={
                "reason": "no_valid_rank_autocorrelation",
                "n_observed": 0,
            },
        )

    rc_arr = rc_per_date["rc"].to_numpy()
    mean_rc = float(np.mean(rc_arr))
    turnover = 1.0 - mean_rc

    return MetricOutput(
        name="turnover",
        value=turnover,
        metadata={"mean_rank_autocorrelation": mean_rc, "n_dates": len(rc_arr)},
    )


def breakeven_cost(
    gross_spread: float,
    turnover: float,
) -> MetricOutput:
    """Breakeven single-leg trading cost in bps.

    ``Breakeven = Gross_Spread / (2 × Turnover)``

    If the actual trading cost is below this, the factor's alpha survives.

    Args:
        gross_spread: Per-period mean long-short spread.
        turnover: Factor turnover estimate (0-1).

    Returns:
        MetricOutput with value = breakeven cost in bps.

    References:
        Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs."
    """
    if turnover < EPSILON:
        return MetricOutput(
            name="breakeven_cost",
            value=float("inf"),
            metadata={"gross_spread": gross_spread, "turnover": turnover},
        )

    # WHY: ×2 因為 long-short 雙邊交易；×10000 轉 bps
    be_bps = (gross_spread / (2 * turnover)) * 10000

    return MetricOutput(
        name="breakeven_cost",
        value=be_bps,
        metadata={"gross_spread": gross_spread, "turnover": turnover},
    )


def net_spread(
    gross_spread: float,
    turnover: float,
    estimated_cost_bps: float = 30.0,
) -> MetricOutput:
    """Net spread after estimated trading costs (per-period).

    ``Net = Gross_Spread - 2 × cost_bps × Turnover``

    Args:
        gross_spread: Per-period mean long-short spread.
        turnover: Factor turnover estimate.
        estimated_cost_bps: Estimated single-leg trading cost in bps.

    Returns:
        MetricOutput with value = net spread (per-period).

    References:
        DeMiguel, Martin-Utrera & Nogales (2020).
    """
    cost_drag = 2 * (estimated_cost_bps / 10000) * turnover
    net = gross_spread - cost_drag

    return MetricOutput(
        name="net_spread",
        value=net,
        metadata={
            "gross_spread": gross_spread,
            "cost_drag": cost_drag,
            "estimated_cost_bps": estimated_cost_bps,
            "turnover": turnover,
        },
    )
