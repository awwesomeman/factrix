"""IC (Information Coefficient) computation for cross-sectional panels.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
Output: time-indexed IC series (``date, ic``) that can be fed into
any ``series/`` tool (oos, trend, significance, hit_rate).
"""

from __future__ import annotations

import math

import polars as pl

from factrix._types import (
    EPSILON,
    MIN_IC_PERIODS,
    MetricOutput,
)
from factrix.metrics._helpers import (
    _sample_non_overlapping,
    _scaled_min_periods,
    _short_circuit_output,
)
from factrix._stats import (
    _calc_t_stat,
    _newey_west_t_test,
    _p_value_from_t,
    _significance_marker,
)
from factrix.stats import bhy_adjusted_p


def compute_ic(
    df: pl.DataFrame,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Per-date Spearman Rank IC.

    Args:
        df: Panel with ``date``, ``asset_id``, ``factor_col``, ``return_col``.

    Returns:
        DataFrame with columns ``date, ic`` sorted by date.
        Dates with fewer than ``MIN_IC_PERIODS`` assets are dropped.
    """
    ranked = df.with_columns(
        pl.col(factor_col).rank(method="average").over("date").alias("_rank_factor"),
        pl.col(return_col).rank(method="average").over("date").alias("_rank_return"),
    )
    return (
        ranked.group_by("date")
        .agg(
            pl.corr("_rank_factor", "_rank_return").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= MIN_IC_PERIODS)
        .sort("date")
        .select("date", "ic")
    )



def ic(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> MetricOutput:
    """IC mean significance: is mean IC significantly different from zero?

    Uses non-overlapping sampling (every ``forward_periods``-th date) to
    eliminate autocorrelation from overlapping forward returns.

    Statistical method: t = mean / (std / √n) on non-overlapping samples.
    H₀: mean IC = 0.

    Args:
        ic_df: Output of ``compute_ic()``.
        forward_periods: Sampling interval for non-overlapping dates.

    Returns:
        MetricOutput with value=mean IC, t_stat from non-overlapping sampling.
    """
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    raw_min = _scaled_min_periods(MIN_IC_PERIODS, forward_periods)
    if n < raw_min:
        return _short_circuit_output(
            "ic", "insufficient_ic_periods",
            n_observed=n, min_required=raw_min,
            forward_periods=forward_periods,
        )

    mean_ic = float(ic_vals.mean())
    sampled = _sample_non_overlapping(ic_df, forward_periods)["ic"].drop_nulls()
    n_sampled = len(sampled)
    if n_sampled < MIN_IC_PERIODS:
        return _short_circuit_output(
            "ic", "insufficient_sampled_ic_periods",
            n_observed=n_sampled, min_required=MIN_IC_PERIODS,
            forward_periods=forward_periods,
        )
    t = _calc_t_stat(float(sampled.mean()), float(sampled.std()), n_sampled)
    p = _p_value_from_t(t, n_sampled)

    return MetricOutput(
        name="ic",
        value=mean_ic,
        stat=t,
        significance=_significance_marker(p),
        metadata={
            "n_periods": n,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "non-overlapping t-test",
        },
    )


def ic_newey_west(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> MetricOutput:
    """IC mean significance via Newey-West HAC t-test on the overlapping series.

    Sibling of ``ic()``: same null hypothesis (H₀: mean IC = 0), but
    keeps every observation and absorbs the autocorrelation induced by
    overlapping ``forward_periods``-day returns through HAC standard
    errors rather than dropping samples.

    Lag selection: ``max(floor(T^(1/3)), forward_periods - 1)`` — the
    usual rule-of-thumb floor, raised to cover the theoretical
    dependence horizon of overlapping returns.
    """
    ic_vals = ic_df["ic"].drop_nulls().to_numpy()
    n = len(ic_vals)
    if n < MIN_IC_PERIODS:
        return _short_circuit_output(
            "ic_newey_west", "insufficient_ic_periods",
            n_observed=n, min_required=MIN_IC_PERIODS,
        )

    from factrix._stats import _resolve_nw_lags
    lags = _resolve_nw_lags(n, lags=None, forward_periods=forward_periods)
    t, p, sig = _newey_west_t_test(ic_vals, forward_periods=forward_periods)
    return MetricOutput(
        name="ic_newey_west",
        value=float(ic_vals.mean()),
        stat=t,
        significance=sig,
        metadata={
            "n_periods": n,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "Newey-West HAC t-test on overlapping IC series",
            "newey_west_lags": lags,
            "forward_periods": forward_periods,
        },
    )


def ic_ir(
    ic_df: pl.DataFrame,
) -> MetricOutput:
    """IC_IR = mean(IC) / std(IC).

    Signed ratio — positive when IC is consistently positive, negative
    when consistently negative.  Analogous to a Sharpe ratio for the
    factor signal.

    This is a **descriptive statistic**, not a hypothesis test (t_stat=None).
    For significance testing, use ``ic()``.

    Args:
        ic_df: Output of ``compute_ic()``.

    Returns:
        MetricOutput with value=IC_IR (signed), t_stat=None.
    """
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    if n < MIN_IC_PERIODS:
        return _short_circuit_output(
            "ic_ir", "insufficient_ic_periods",
            n_observed=n, min_required=MIN_IC_PERIODS,
        )

    mean_ic = float(ic_vals.mean())
    std_ic = float(ic_vals.std())

    if std_ic < EPSILON:
        return _short_circuit_output(
            "ic_ir", "degenerate_ic_variance", std_ic=std_ic,
        )

    ratio = mean_ic / std_ic

    return MetricOutput(
        name="ic_ir",
        value=ratio,
        metadata={"mean_ic": mean_ic, "std_ic": std_ic, "n_periods": n},
    )


def regime_ic(
    ic_df: pl.DataFrame,
    regime_labels: pl.DataFrame | None = None,
) -> MetricOutput:
    """IC conditioned on regime labels.

    Answers "is the factor stable across market environments?"
    unlike OOS decay which tests overfitting.

    Each regime's t-stat is an independent test (H₀: mean IC = 0 within
    that regime); sweeping k regimes manufactures k implicit tests.
    Each per-regime entry in ``per_regime`` metadata reports ``p_value``
    (raw) alongside ``p_adjusted_bhy`` (BHY-corrected across the k
    regimes); top-level metadata also surfaces ``p_value_bhy_adjusted``
    = the worst adjusted p across regimes, for aggregate decisions.

    Args:
        ic_df: Output of ``compute_ic()`` (``date, ic``).
        regime_labels: DataFrame with ``date, regime`` (string labels).
            If None, falls back to time bisection (first half / second half).

    Returns:
        MetricOutput with value = mean IC across regimes,
        stat = min |t| across regimes (conservative: if weakest passes, all pass).
        Per-regime details in metadata, each with raw and BHY-adjusted p.

    References:
        Chen & Zimmermann (2022): report sub-period t-stats separately.
        Benjamini-Yekutieli (2001): FDR control under arbitrary dependence.
    """
    if len(ic_df) < MIN_IC_PERIODS:
        return _short_circuit_output(
            "regime_ic", "insufficient_ic_periods",
            n_observed=len(ic_df), min_required=MIN_IC_PERIODS,
        )

    if regime_labels is not None:
        merged = ic_df.join(regime_labels.select("date", "regime"), on="date", how="inner")
    else:
        # Fallback: time bisection
        sorted_ic = ic_df.sort("date")
        mid = len(sorted_ic) // 2
        merged = sorted_ic.with_row_index("_idx").with_columns(
            pl.when(pl.col("_idx") < mid)
            .then(pl.lit("first_half"))
            .otherwise(pl.lit("second_half"))
            .alias("regime")
        ).drop("_idx")

    # Single-pass group_by instead of per-regime Python loop
    regime_stats = (
        merged.drop_nulls("ic")
        .group_by("regime")
        .agg(
            pl.col("ic").mean().alias("mean_ic"),
            pl.col("ic").std().alias("std_ic"),
            pl.col("ic").count().alias("n"),
        )
        .filter(pl.col("n") >= 2)
        .sort("regime")
    )

    if regime_stats.is_empty():
        return _short_circuit_output(
            "regime_ic", "no_regime_has_enough_observations",
            min_required_per_regime=2,
        )

    per_regime: dict[str, dict[str, object]] = {}
    raw_p_list: list[float] = []
    regime_order: list[str] = []
    for row in regime_stats.iter_rows(named=True):
        t = _calc_t_stat(row["mean_ic"], row["std_ic"], row["n"])
        p = _p_value_from_t(t, row["n"])
        per_regime[row["regime"]] = {
            "mean_ic": row["mean_ic"],
            "std_ic": row["std_ic"],
            "stat": t,
            "p_value": p,
            "significance": _significance_marker(p),
            "n_periods": row["n"],
        }
        raw_p_list.append(p)
        regime_order.append(row["regime"])

    # BHY across regimes: sweeping k regimes is k implicit tests on the
    # same null family. Report adjusted per-regime p for callers that
    # want to act on a specific regime's significance without manually
    # correcting; also surface min adjusted p for aggregate decisions.
    adj_p = bhy_adjusted_p(raw_p_list) if raw_p_list else []
    for name, ap in zip(regime_order, adj_p):
        per_regime[name]["p_adjusted_bhy"] = float(ap)

    mean_all = float(sum(d["mean_ic"] for d in per_regime.values()) / len(per_regime))

    # Conservative summary: if the weakest regime is significant, all are.
    min_abs_t_regime = min(per_regime.values(), key=lambda d: abs(d["stat"]))
    min_t = min_abs_t_regime["stat"]
    min_p = min_abs_t_regime["p_value"]
    min_p_adjusted = float(max(adj_p)) if len(adj_p) else 1.0

    # WHY: filter near-zero means before checking direction consistency —
    # a regime with IC ≈ 0 has no signal, not a "consistent direction"
    nonzero = [d["mean_ic"] for d in per_regime.values() if abs(d["mean_ic"]) > EPSILON]
    if nonzero:
        consistent = all(v > 0 for v in nonzero) or all(v < 0 for v in nonzero)
    else:
        consistent = False

    return MetricOutput(
        name="regime_ic",
        value=mean_all,
        stat=min_t,
        significance=_significance_marker(min_p),
        metadata={
            "p_value": min_p,
            "p_value_bhy_adjusted": min_p_adjusted,
            "stat_type": "t",
            "h0": "mu=0 (per regime)",
            "aggregation": "mean_value_min_stat",
            "per_regime": per_regime,
            "direction_consistent": consistent,
            "n_regimes": len(per_regime),
        },
    )


def multi_horizon_ic(
    df: pl.DataFrame,
    price_col: str = "price",
    factor_col: str = "factor",
    periods: list[int] | None = None,
) -> MetricOutput:
    """Compute mean IC at multiple forward horizons.

    Args:
        df: Preprocessed panel with ``date``, ``asset_id``, ``close``,
            and ``factor`` columns. Output of ``preprocess_cs_factor``.
        price_col: Price column for computing forward returns.
        periods: List of forward periods (default [1, 5, 10, 20]).

    Returns:
        MetricOutput with value = mean IC across horizons,
        stat = min |t| across horizons.
        Per-horizon details in metadata.
    """
    if periods is None:
        periods = [1, 5, 10, 20]

    per_horizon: dict[int, dict[str, object]] = {}

    sorted_df = df.sort(["asset_id", "date"])
    all_returns = sorted_df.with_columns([
        (
            pl.col(price_col).shift(-p).over("asset_id")
            / pl.col(price_col)
            - 1
        ).alias(f"_fwd_ret_{p}")
        for p in periods
    ])

    for p in periods:
        ret_col = f"_fwd_ret_{p}"
        valid = all_returns.filter(pl.col(ret_col).is_not_null())

        ic_series = compute_ic(valid, factor_col=factor_col, return_col=ret_col)
        ic_vals = ic_series["ic"].drop_nulls()
        n_ic = len(ic_vals)

        # Scale the raw threshold to land with ≥ MIN_IC_PERIODS after
        # the p-period non-overlap sub-sampling below.
        if n_ic >= _scaled_min_periods(MIN_IC_PERIODS, p):
            mean_ic = float(ic_vals.mean())
            # WHY: use non-overlapping sampling for t-stat to avoid
            # autocorrelation inflation from overlapping forward returns
            sampled = _sample_non_overlapping(ic_series, p)["ic"].drop_nulls()
            n_sampled = len(sampled)
            if n_sampled >= 2:
                t = _calc_t_stat(float(sampled.mean()), float(sampled.std()), n_sampled)
            else:
                t = 0.0
            p_val = _p_value_from_t(t, n_sampled)
            per_horizon[p] = {
                "mean_ic": mean_ic,
                "stat": t,
                "p_value": p_val,
                "n_periods": n_ic,
            }
        else:
            per_horizon[p] = {
                "mean_ic": float("nan"),
                "stat": 0.0,
                "p_value": 1.0,
                "n_periods": n_ic,
            }

    valid_horizons = [h for h in per_horizon.values() if not math.isnan(h["mean_ic"])]
    if not valid_horizons:
        return _short_circuit_output(
            "multi_horizon_ic", "no_horizon_has_enough_observations",
            min_required=MIN_IC_PERIODS,
        )

    # BHY across horizons: sweeping k horizons is k implicit tests.
    horizon_keys = [h for h in periods if not math.isnan(per_horizon[h]["mean_ic"])]
    raw_h_p = [per_horizon[h]["p_value"] for h in horizon_keys]
    adj_h_p = bhy_adjusted_p(raw_h_p) if raw_h_p else []
    for h, ap in zip(horizon_keys, adj_h_p):
        per_horizon[h]["p_adjusted_bhy"] = float(ap)

    mean_all = float(sum(h["mean_ic"] for h in valid_horizons) / len(valid_horizons))

    weakest = min(valid_horizons, key=lambda h: abs(h["stat"]))
    min_t = weakest["stat"]
    min_p = weakest["p_value"]
    min_p_adjusted = float(max(adj_h_p)) if len(adj_h_p) else 1.0

    return MetricOutput(
        name="multi_horizon_ic",
        value=mean_all,
        stat=min_t,
        significance=_significance_marker(min_p),
        metadata={
            "p_value": min_p,
            "p_value_bhy_adjusted": min_p_adjusted,
            "stat_type": "t",
            "h0": "mu=0 (per horizon)",
            "aggregation": "mean_value_min_stat",
            "per_horizon": per_horizon,
            "n_horizons": len(valid_horizons),
        },
    )
