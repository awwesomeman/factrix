"""Tradability metrics: Turnover, Breakeven Cost, Net Spread.

Two flavours of turnover co-exist here, measuring different things:

- ``turnover()`` — ``1 − mean(rank autocorrelation)``. Rank-stability
  diagnostic; responds to mid-rank reshuffling. **Not** a notional
  trading-fraction and should **not** be fed into ``breakeven_cost`` /
  ``net_spread``.
- ``notional_turnover()`` — fraction of top-and-bottom quantile members
  replaced per rebalance. Matches Novy-Marx & Velikov (2016) τ; this is
  the quantity that drives bps trading cost for an equal-weight Q1/Qn
  long-short portfolio.

These are implementation-feasibility indicators, not factor quality
measures — they belong in Profile, not in Gates.

Input for Turnover: DataFrame with ``date, asset_id, factor``.
Input for Breakeven/Net Spread: pre-computed spread and turnover values.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import DDOF, EPSILON, MetricOutput
from factrix.metrics._helpers import (
    _assign_quantile_groups,
    _sample_non_overlapping,
    _short_circuit_output,
)


def turnover(
    df: pl.DataFrame,
    factor_col: str = "factor",
    forward_periods: int = 1,
    quantile: float | None = None,
) -> MetricOutput:
    """Factor rank-stability via non-overlapping rank autocorrelation.

    ``turnover = 1 - mean(rank_autocorrelation)``

    **What this measures.** Sensitivity of the *full* cross-section rank
    vector to reshuffling between ``t`` and ``t + forward_periods``. Mid-rank
    churn (names moving between e.g. Q4 ↔ Q5 in a 10-group split) counts
    even though those names carry zero weight in a Q1/Qn long-short
    portfolio. So this is a **rank-stability diagnostic**, *not* a notional
    trading-fraction.

    **When to use this vs ``notional_turnover``.** Feed a strategy-cost
    formula (breakeven_cost / net_spread) with ``notional_turnover``, not
    this function — the bps coefficients there assume ``turnover`` is the
    fraction of Q1/Qn positions replaced per rebalance, which ``1 − ρ``
    does not provide. Keep this metric for ranking-stability comparisons
    across factors.

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


def notional_turnover(
    df: pl.DataFrame,
    factor_col: str = "factor",
    *,
    n_groups: int = 10,
    forward_periods: int = 1,
) -> MetricOutput:
    """Portfolio notional turnover via top/bottom quantile membership churn.

    For an equal-weight Q1/Q_n long-short portfolio the only trades that
    incur cost are changes in top-quantile and bottom-quantile membership
    — reshuffling within the middle deciles triggers no rebalancing and
    should not be counted. This is the metric whose units are directly
    compatible with ``breakeven_cost`` / ``net_spread``: Novy-Marx &
    Velikov (2016) τ = fraction of portfolio value replaced per rebalance.

    Per-rebalance turnover is the mean of two one-sided overlap losses::

        top_churn = 1 - |Q_top_t ∩ Q_top_{t-1}| / |Q_top_t|
        bot_churn = 1 - |Q_bot_t ∩ Q_bot_{t-1}| / |Q_bot_t|
        turnover  = (top_churn + bot_churn) / 2

    ``(k − m) / k`` for ``k`` names in today's tail and ``m`` carry-overs
    equals the fraction of that leg that must be traded under equal
    weighting. Averaging the two legs keeps the ``× 2`` factor in
    ``breakeven_cost = spread / (2 × turnover) × 1e4`` consistent.

    Args:
        df: Panel with ``date, asset_id, factor``.
        factor_col: Name of the factor column.
        n_groups: Number of quantile groups (default ``10`` = deciles).
            Must be ≥ 3 so top and bottom are distinct buckets.
        forward_periods: Rebalance stride. When ``> 1``, sub-samples at
            stride ``h`` before pairing consecutive dates — matches a
            holding-period-aligned rebalance schedule.

    Returns:
        MetricOutput with ``value`` = mean per-rebalance turnover ∈ [0, 1].
        ``0`` = identical tail sets every rebalance; ``1`` = full rotation.
        Metadata: ``n_rebalances``, ``n_groups``, ``forward_periods``,
        ``mean_tail_size`` (per-date average of ``(|Q_top| + |Q_bot|)/2``;
        ≠ ``n_assets / n_groups`` signals unbalanced buckets from ties or
        a short universe), ``method``.

    Notes:
        Names dropped from ``Q_top_{t-1}`` / ``Q_bot_{t-1}`` by delisting
        before ``t`` (not present in ``df`` at ``t``) are silently missed
        on the sell side — real portfolios would book that liquidation
        cost. The bias is typically small but grows with universe churn.

    References:
        Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their
        Trading Costs."
    """
    if forward_periods < 1:
        raise ValueError(
            f"forward_periods must be ≥ 1, got {forward_periods!r}"
        )
    if n_groups < 3:
        raise ValueError(
            f"n_groups must be ≥ 3 (need distinct top/bottom buckets), "
            f"got {n_groups!r}"
        )

    if forward_periods > 1:
        df = _sample_non_overlapping(df, forward_periods)

    dates = df["date"].unique().sort()
    if len(dates) < 2:
        return _short_circuit_output(
            "notional_turnover", "insufficient_dates",
            n_observed=len(dates), min_required=2,
            forward_periods=forward_periods,
        )

    top_g = n_groups - 1
    bot_g = 0
    grouped = (
        _assign_quantile_groups(df, factor_col, n_groups)
        .select(
            "date", "asset_id",
            (pl.col("_group") == top_g).alias("is_top"),
            (pl.col("_group") == bot_g).alias("is_bot"),
        )
    )

    date_map = pl.DataFrame({"date": dates[1:], "prev_date": dates[:-1]})
    prev = grouped.select(
        pl.col("date").alias("prev_date"),
        "asset_id",
        pl.col("is_top").alias("was_top"),
        pl.col("is_bot").alias("was_bot"),
    )

    # WHY: fill_null(False) treats names absent at t-1 as "new top/bot" —
    # this matches a live portfolio that has to buy into them at t.
    paired = (
        grouped.join(date_map, on="date")
        .join(prev, on=["prev_date", "asset_id"], how="left")
        .with_columns(
            pl.col("was_top").fill_null(False),
            pl.col("was_bot").fill_null(False),
        )
    )

    per_date = (
        paired.group_by("date")
        .agg(
            pl.col("is_top").sum().alias("n_top"),
            (pl.col("is_top") & pl.col("was_top")).sum().alias("n_top_kept"),
            pl.col("is_bot").sum().alias("n_bot"),
            (pl.col("is_bot") & pl.col("was_bot")).sum().alias("n_bot_kept"),
        )
        .filter((pl.col("n_top") > 0) & (pl.col("n_bot") > 0))
        .with_columns(
            (
                (1 - pl.col("n_top_kept") / pl.col("n_top"))
                + (1 - pl.col("n_bot_kept") / pl.col("n_bot"))
            ).truediv(2).alias("turnover")
        )
        .sort("date")
    )

    if per_date.is_empty():
        return _short_circuit_output(
            "notional_turnover", "no_valid_pairs",
            forward_periods=forward_periods, n_groups=n_groups,
        )

    turnover_arr = per_date["turnover"].to_numpy()
    mean_turnover = float(np.mean(turnover_arr))
    tail_pct = 1.0 / n_groups

    mean_tail_size = float(
        per_date.select(
            ((pl.col("n_top") + pl.col("n_bot")) / 2).mean()
        ).item()
    )
    return MetricOutput(
        name="notional_turnover",
        value=mean_turnover,
        metadata={
            "n_rebalances": int(per_date.height),
            "n_groups": n_groups,
            "forward_periods": forward_periods,
            "mean_tail_size": mean_tail_size,
            "method": (
                f"one-sided overlap on top/bottom {tail_pct:.0%} "
                f"quantile, top/bot averaged"
            ),
        },
    )


def breakeven_cost(
    gross_spread: float,
    turnover: float,
    *,
    forward_periods: int,
) -> MetricOutput:
    """Breakeven single-leg trading cost in bps.

    ``Breakeven = Gross_Spread × forward_periods / (2 × Turnover)``

    If the actual trading cost is below this, the factor's alpha survives.

    Expects ``turnover`` to be a **notional** fraction ∈ [0, 1] — the
    share of the equal-weight Q1/Q_n portfolio replaced per rebalance.
    Use ``notional_turnover()``; do **not** feed in ``turnover()``
    (which is rank-stability, not position-change).

    Time-scale alignment: ``gross_spread`` from ``quantile_spread`` is
    per-period (forward_return is divided by N upstream), but ``turnover``
    is per-rebalance (one rotation every N periods). Multiplying spread
    by ``forward_periods`` puts both sides on the per-rebalance scale
    before solving net=0 — without it, breakeven is understated by N×.

    Args:
        gross_spread: Per-period mean long-short spread.
        turnover: Notional turnover ∈ [0, 1] from ``notional_turnover()``.
        forward_periods: Holding period N matching the upstream
            ``compute_forward_return`` and ``notional_turnover`` stride.

    Returns:
        MetricOutput with value = breakeven cost in bps.

    References:
        Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs."
    """
    if forward_periods < 1:
        raise ValueError(
            f"forward_periods must be ≥ 1, got {forward_periods!r}"
        )
    if turnover < EPSILON:
        return MetricOutput(
            name="breakeven_cost",
            value=float("inf"),
            metadata={
                "gross_spread": gross_spread,
                "turnover": turnover,
                "forward_periods": forward_periods,
            },
        )

    # WHY: ×2 因為 long-short 雙邊交易；×forward_periods 把 per-period spread
    # 升到 per-rebalance 與 turnover 對齊；×10000 轉 bps。
    be_bps = (gross_spread * forward_periods / (2 * turnover)) * 10000

    return MetricOutput(
        name="breakeven_cost",
        value=be_bps,
        metadata={
            "gross_spread": gross_spread,
            "turnover": turnover,
            "forward_periods": forward_periods,
        },
    )


def net_spread(
    gross_spread: float,
    turnover: float,
    estimated_cost_bps: float = 30.0,
    *,
    forward_periods: int,
) -> MetricOutput:
    """Net spread after estimated trading costs (per-period).

    ``Net = Gross_Spread - 2 × cost_bps × Turnover / forward_periods``

    The ``2 ×`` accounts for both legs of the long-short portfolio
    needing to be traded (long side + short side) at each rebalance.
    Default ``estimated_cost_bps=30`` is a conservative single-leg
    mid-cap US equity estimate (half-spread + impact) sized to give a
    useful headline number; override with a venue-specific estimate
    when available.

    Time-scale alignment: ``gross_spread`` is per-period (forward_return
    is divided by N upstream) but ``2 × cost × turnover`` is the cost paid
    once per N-period rebalance. Dividing by ``forward_periods`` amortises
    that cost back to per-period. Without it, net is over-charged by N×
    and any factor with h ≥ 2 is artificially killed.

    Expects ``turnover`` to be a **notional** fraction ∈ [0, 1] — the
    share of the equal-weight Q1/Q_n portfolio replaced per rebalance.
    Use ``notional_turnover()``; do **not** feed in ``turnover()``
    (which is rank-stability, not position-change).

    Args:
        gross_spread: Per-period mean long-short spread.
        turnover: Notional turnover ∈ [0, 1] from ``notional_turnover()``.
        estimated_cost_bps: Estimated single-leg trading cost in bps.
        forward_periods: Holding period N matching the upstream
            ``compute_forward_return`` and ``notional_turnover`` stride.

    Returns:
        MetricOutput with value = net spread (per-period).

    References:
        DeMiguel, Martin-Utrera, Nogales & Uppal (2020), "A
        Transaction-Cost Perspective on the Multitude of Firm
        Characteristics." *Review of Financial Studies* 33(5).
    """
    if forward_periods < 1:
        raise ValueError(
            f"forward_periods must be ≥ 1, got {forward_periods!r}"
        )
    cost_drag = 2 * (estimated_cost_bps / 10000) * turnover / forward_periods
    net = gross_spread - cost_drag

    return MetricOutput(
        name="net_spread",
        value=net,
        metadata={
            "gross_spread": gross_spread,
            "cost_drag": cost_drag,
            "estimated_cost_bps": estimated_cost_bps,
            "turnover": turnover,
            "forward_periods": forward_periods,
        },
    )
