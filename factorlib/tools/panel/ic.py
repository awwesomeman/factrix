"""IC (Information Coefficient) computation for cross-sectional panels.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
Output: time-indexed IC series (``date, ic``) that can be fed into
any ``series/`` tool (oos, trend, significance, hit_rate).
"""

from __future__ import annotations

import polars as pl

from factorlib.tools._typing import (
    EPSILON,
    MIN_IC_PERIODS,
    MetricOutput,
)
from factorlib.tools._helpers import sample_non_overlapping
from factorlib.tools.series.significance import calc_t_stat, significance_marker


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


def _non_overlapping_ic_tstat(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> float:
    """T-stat from non-overlapping IC samples (internal helper).

    Samples every ``forward_periods``-th date to eliminate autocorrelation
    from overlapping forward returns.
    """
    sampled_ic = sample_non_overlapping(ic_df, forward_periods)["ic"].drop_nulls()

    n = len(sampled_ic)
    if n < 2:
        return 0.0
    return calc_t_stat(
        float(sampled_ic.mean()),
        float(sampled_ic.std()),
        n,
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
    if n < MIN_IC_PERIODS:
        return MetricOutput(name="IC", value=0.0, t_stat=0.0, significance="")

    mean_ic = float(ic_vals.mean())
    t = _non_overlapping_ic_tstat(ic_df, forward_periods)

    return MetricOutput(
        name="IC",
        value=mean_ic,
        t_stat=t,
        significance=significance_marker(t),
        metadata={"n_periods": n},
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
        return MetricOutput(name="IC_IR", value=0.0)

    mean_ic = float(ic_vals.mean())
    std_ic = float(ic_vals.std())

    if std_ic < EPSILON:
        return MetricOutput(name="IC_IR", value=0.0)

    ratio = mean_ic / std_ic

    return MetricOutput(
        name="IC_IR",
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
    that regime).  This is **not** a joint test across all regimes.
    Strictly, testing "all regimes significant" would require a multiple
    comparison correction (e.g. Bonferroni: threshold = t_{α/2k} for k
    regimes).  In practice, with only 2-3 regimes the correction is
    small (2.0 → ~2.2 for k=2) and rarely applied in factor research.

    Args:
        ic_df: Output of ``compute_ic()`` (``date, ic``).
        regime_labels: DataFrame with ``date, regime`` (string labels).
            If None, falls back to time bisection (first half / second half).

    Returns:
        MetricOutput with value = min regime mean IC (conservative),
        metadata containing per-regime IC stats.

    References:
        Chen & Zimmermann (2022): report sub-period t-stats separately.
    """
    if len(ic_df) < MIN_IC_PERIODS:
        return MetricOutput(name="Regime_IC", value=0.0, significance="")

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
        return MetricOutput(name="Regime_IC", value=0.0, significance="")

    per_regime: dict[str, dict[str, object]] = {}
    for row in regime_stats.iter_rows(named=True):
        t = calc_t_stat(row["mean_ic"], row["std_ic"], row["n"])
        per_regime[row["regime"]] = {
            "mean_ic": row["mean_ic"],
            "std_ic": row["std_ic"],
            "t_stat": t,
            "significance": significance_marker(t),
            "n_periods": row["n"],
        }

    min_mean = min(d["mean_ic"] for d in per_regime.values())

    # WHY: filter near-zero means before checking direction consistency —
    # a regime with IC ≈ 0 has no signal, not a "consistent direction"
    nonzero = [d["mean_ic"] for d in per_regime.values() if abs(d["mean_ic"]) > EPSILON]
    if nonzero:
        consistent = all(v > 0 for v in nonzero) or all(v < 0 for v in nonzero)
    else:
        consistent = False

    return MetricOutput(
        name="Regime_IC",
        value=min_mean,
        metadata={
            "per_regime": per_regime,
            "direction_consistent": consistent,
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
        MetricOutput with value=mean IC at the default horizon,
        metadata containing per-horizon IC values.
    """
    if periods is None:
        periods = [1, 5, 10, 20]

    horizon_ics: dict[int, float] = {}

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

        if len(ic_vals) >= MIN_IC_PERIODS:
            horizon_ics[p] = float(ic_vals.mean())
        else:
            horizon_ics[p] = float("nan")

    primary = horizon_ics.get(periods[0], float("nan"))
    return MetricOutput(
        name="Multi_Horizon_IC",
        value=primary,
        metadata={"horizon_ics": horizon_ics},
    )
