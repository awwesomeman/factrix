"""Tradability metrics: Turnover, Breakeven Cost, Net Spread.

These are implementation-feasibility indicators, not factor quality
measures — they belong in Profile, not in Gates.

Input for Turnover: DataFrame with ``date, asset_id, factor``.
Input for Breakeven/Net Spread: pre-computed spread and turnover values.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import DDOF, EPSILON, MetricOutput
from factrix.metrics._helpers import _sample_non_overlapping, _short_circuit_output
from factrix._stats import _significance_marker


def turnover(
    df: pl.DataFrame,
    factor_col: str = "factor",
    forward_periods: int = 1,
    quantile: float | None = None,
) -> MetricOutput:
    """Factor turnover via non-overlapping rank autocorrelation.

    ``turnover = 1 - mean(rank_autocorrelation)``

    Rank autocorrelation is measured between dates ``t`` and ``t +
    forward_periods``, sub-sampled at stride ``forward_periods`` (phase-0)
    so each pair is a non-overlapping snapshot. This aligns the stability
    window with the forward-return horizon used elsewhere in the profile.

    Args:
        df: Panel with ``date, asset_id, factor``.
        factor_col: Name of the factor column. Defaults to ``"factor"``.
        forward_periods: Sampling stride in periods — should match the
            forward-return horizon the factor is being evaluated against.
            ``1`` reproduces the lag-1 behaviour.
        quantile: Optional tail filter in ``(0, 0.5)``. When set, restrict
            the Spearman ρ at each pair to assets whose rank at *either*
            endpoint lies in the top-q or bottom-q of that date's cross-
            section — i.e. the statistical region where the long-short
            spread is actually measured. Union (not intersection) so names
            entering or leaving the tail both register as turnover.

            Caveat: ρ on the tail union is NOT comparable to the
            unfiltered ρ — tail names are more persistent by construction,
            so the resulting turnover will typically be lower. Compare
            only against other tail-filtered estimates at the same q.

    Returns:
        MetricOutput with ``value = turnover estimate (0–1)`` and metadata
        carrying ``mean_rank_autocorrelation``, ``std_rank_autocorrelation``,
        ``n_pairs``, ``forward_periods``, ``quantile``, and
        ``n_cross_section_mean`` (mean assets-per-pair post-filter).

        ``std_rank_autocorrelation`` is the cross-pair sample std. Using
        ``std/√n_pairs`` as an SE is a *lower bound*: consecutive pairs
        share one rank-vector endpoint (pair k and pair k+1 both involve
        ``rank @ t_{k·h}``), so the per-pair ρ's have weak positive
        dependence and the true SE is marginally larger. For publication
        grade inference, use a HAC variance estimator.
    """
    if quantile is not None and not 0.0 < quantile < 0.5:
        raise ValueError(
            f"quantile must be in (0, 0.5), got {quantile!r}"
        )
    if forward_periods < 1:
        raise ValueError(
            f"forward_periods must be ≥ 1, got {forward_periods!r}"
        )

    all_dates = df["date"].unique().sort()
    # Need ≥ 2 non-overlapping pairs so std(ρ) is defined; that requires
    # ≥ 3 sampled dates (Hansen & Hodrick 1980), i.e. ≥ 2·h + 1 raw dates.
    min_required = 2 * forward_periods + 1
    if len(all_dates) < min_required:
        return _short_circuit_output(
            "turnover", "insufficient_dates",
            n_observed=len(all_dates), min_required=min_required,
            forward_periods=forward_periods,
        )

    sampled_df = _sample_non_overlapping(df, forward_periods)

    ranked = sampled_df.select(
        "date", "asset_id",
        pl.col(factor_col).rank(method="average").over("date").alias("rank_curr"),
        pl.len().over("date").alias("n_curr"),
    ).sort("asset_id", "date")

    # Lag-within-asset avoids a self-join on (prev_date, asset_id):
    # rank_prev at date_k is this asset's rank at the previous *sampled*
    # date, which is the prior row in each asset's sorted group.
    paired = ranked.with_columns(
        pl.col("rank_curr").shift(1).over("asset_id").alias("rank_prev"),
        pl.col("n_curr").shift(1).over("asset_id").alias("n_prev"),
    ).drop_nulls(["rank_prev"])

    if quantile is not None:
        in_tail = (
            (pl.col("rank_curr") <= pl.col("n_curr") * quantile)
            | (pl.col("rank_curr") > pl.col("n_curr") * (1.0 - quantile))
            | (pl.col("rank_prev") <= pl.col("n_prev") * quantile)
            | (pl.col("rank_prev") > pl.col("n_prev") * (1.0 - quantile))
        )
        paired = paired.filter(in_tail)

    rc_per_date = (
        paired.group_by("date")
        .agg(
            pl.corr("rank_curr", "rank_prev").alias("rc"),
            pl.len().alias("n_pair"),
        )
        .filter(pl.col("rc").is_not_null() & pl.col("rc").is_not_nan())
        .sort("date")
    )

    if rc_per_date.height < 2:
        return _short_circuit_output(
            "turnover", "insufficient_pairs",
            n_observed=rc_per_date.height, min_required=2,
            forward_periods=forward_periods, quantile=quantile,
        )

    rc_arr = rc_per_date["rc"].to_numpy()
    mean_rc = float(np.mean(rc_arr))
    std_rc = float(np.std(rc_arr, ddof=DDOF))
    n_cs_mean = float(rc_per_date["n_pair"].mean())

    return MetricOutput(
        name="turnover",
        value=1.0 - mean_rc,
        metadata={
            "mean_rank_autocorrelation": mean_rc,
            "std_rank_autocorrelation": std_rc,
            "n_pairs": rc_per_date.height,
            "forward_periods": forward_periods,
            "quantile": quantile,
            "n_cross_section_mean": n_cs_mean,
        },
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

    The ``2 ×`` accounts for both legs of the long-short portfolio
    needing to be traded (long side + short side) at each rebalance.
    Default ``estimated_cost_bps=30`` is a conservative single-leg
    mid-cap US equity estimate (half-spread + impact) sized to give a
    useful headline number; override with a venue-specific estimate
    when available.

    Args:
        gross_spread: Per-period mean long-short spread.
        turnover: Factor turnover estimate.
        estimated_cost_bps: Estimated single-leg trading cost in bps.

    Returns:
        MetricOutput with value = net spread (per-period).

    References:
        DeMiguel, Martin-Utrera, Nogales & Uppal (2020), "A
        Transaction-Cost Perspective on the Multitude of Firm
        Characteristics." *Review of Financial Studies* 33(5).
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
