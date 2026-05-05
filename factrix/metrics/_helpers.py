"""Shared helpers used across multiple tool modules.

These are internal utilities — not part of the public API.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl

from factrix._types import MetricOutput

# Median-across-dates tie_ratio above this triggers a UserWarning when
# tie_policy="ordinal". 0.3 is the empirical cutoff for "crowded" factors
# (bucketed signals, industry/size dummies routinely sit at ~0.5 — below
# 0.3 the sorting-artifact noise from ordinal tie-breaking is negligible).
TIE_RATIO_WARN_THRESHOLD = 0.3


def _aggregate_to_per_date(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    factor_alias: str = "_f",
    return_alias: str = "_r",
) -> pl.DataFrame:
    """Collapse a panel to one row per ``date`` (mean factor + mean return).

    For COMMON-scope factors (broadcast within date) the mean is the
    identity. For single-asset TIMESERIES it is also the identity.
    For INDIVIDUAL panels the cross-section is silently averaged —
    callers using this on time-series-only metrics document that
    aggregation in their own docstrings.
    """
    return (
        df.lazy()
        .group_by("date")
        .agg(
            pl.col(factor_col).mean().alias(factor_alias),
            pl.col(return_col).mean().alias(return_alias),
        )
        .filter(pl.col(factor_alias).is_not_null() & pl.col(return_alias).is_not_null())
        .sort("date")
        .collect()
    )


def _short_circuit_output(
    name: str,
    reason: str,
    **extra_metadata: object,
) -> MetricOutput:
    """Canonical short-circuit ``MetricOutput`` for "cannot compute".

    Reason vocabulary (matches ``_insufficient_metrics`` prefixes):
        - ``insufficient_<thing>`` — data shortage (dropped from BHY)
        - ``no_<thing>`` — missing input / missing config / missing data

    ``value=NaN`` (not 0.0) because 0.0 is a legal factor-metric outcome
    (IC exactly 0, β exactly 0, spread exactly 0) indistinguishable from
    a silent short-circuit. NaN propagates through downstream aggregations
    and plots, making data shortages impossible to misread as valid zeros.

    ``p_value=1.0`` is the conservative default so BHY treats short-circuited
    metrics as rejected rather than crashing; ``_pv`` reads the same key.

    Use this instead of hand-rolling ``MetricOutput(value=float("nan"),
    stat=None, significance="", metadata={"reason": ..., "p_value": 1.0, ...})``.
    """
    return MetricOutput(
        name=name,
        value=float("nan"),
        stat=None,
        significance="",
        metadata={"reason": reason, "p_value": 1.0, **extra_metadata},
    )


def _pick_event_return_col(df: pl.DataFrame) -> str:
    """Return the preferred return column for event analysis.

    ``abnormal_return`` (cross-sectionally de-meaned return) is preferred
    when present; ``forward_return`` is the fallback for single-asset
    panels where de-meaning is undefined. Centralized here so EventFactor
    sessions, EventProfile.from_artifacts, and the build_artifacts
    pipeline agree on the same choice — diverging would silently route
    the same Factor call through different series.
    """
    return "abnormal_return" if "abnormal_return" in df.columns else "forward_return"


def _sample_non_overlapping(
    df: pl.DataFrame,
    forward_periods: int,
) -> pl.DataFrame:
    """Keep every N-th date to produce a non-overlapping series.

    Algorithm:
        1. ``unique_dates = sort(df[date].unique())``
        2. ``sampled = unique_dates[::forward_periods]``  (every N-th)
        3. Return ``df.filter(date ∈ sampled)``

    Why: with h-period forward returns, consecutive dates' forward
    returns share h−1 bars of future data — the series has an MA(h−1)
    structure (Hansen & Hodrick 1980). Sub-sampling at interval h
    breaks this dependence at the cost of throwing away h−1 of every
    h observations. This is the most conservative of the Richardson-
    Stock (1989) remedies; ``_newey_west_t_test`` is the less-lossy
    alternative (keeps all obs but corrects SE).

    Logs a WARNING at ``factrix.metrics`` when the sampled series
    has < 1.5 × MIN_ASSETS_PER_DATE_IC rows — downstream t-tests may be frail
    even if they don't short-circuit.

    Args:
        df: DataFrame with a ``date`` column.
        forward_periods: Sampling interval (typically equals the
            ``forward_periods`` of the forward-return column).

    Returns:
        Filtered DataFrame containing only the sampled dates; all
        other columns untouched.
    """
    from factrix._logging import get_metrics_logger
    from factrix._types import MIN_ASSETS_PER_DATE_IC

    sampled_dates = df["date"].unique().sort().gather_every(forward_periods)
    result = df.filter(pl.col("date").is_in(sampled_dates.implode()))
    n_after = len(sampled_dates)
    logger = get_metrics_logger()
    logger.debug(
        "non_overlap_sample: forward_periods=%d n_dates_before=%d n_after=%d",
        forward_periods,
        df["date"].n_unique(),
        n_after,
    )
    # WARNING: post-sampling series shorter than 1.5x the usual minimum is
    # a red flag — downstream t-tests either short-circuit or operate on
    # a frail sample that silently caller-doesn't-notice.
    min_safe = int(MIN_ASSETS_PER_DATE_IC * 1.5)
    if 0 < n_after < min_safe:
        logger.warning(
            "non_overlap_sample shrunk to n=%d (< %d = MIN_ASSETS_PER_DATE_IC*1.5); "
            "downstream significance tests may be unreliable. "
            "forward_periods=%d",
            n_after,
            min_safe,
            forward_periods,
        )
    return result


def _scaled_min_periods(base: int, forward_periods: int) -> int:
    """Raw-sample minimum for a metric that will sub-sample at stride h.

    ``MIN_*_PERIODS`` constants are calibrated for the *effective*
    sample size the downstream t-test operates on. When the metric
    first runs ``_sample_non_overlapping(df, h)`` the effective n
    shrinks to ``raw_n / h``, so the pre-sampling guard needs
    ``raw_n ≥ base · h`` to land with ≥ ``base`` independent
    observations after sampling. Clamps ``h ≥ 1`` so ``h = 1`` is a
    no-op.
    """
    return base * max(forward_periods, 1)


def _lag_within_asset(
    df: pl.DataFrame,
    col: str,
    *,
    periods: int = 1,
    by: str = "asset_id",
) -> pl.DataFrame:
    """Replace ``col`` with its per-asset lag; drop rows where the lag is null.

    Common post-sampling pattern: after ``_sample_non_overlapping`` sorts
    the panel to the rebalance schedule, we want each row's ``col`` to
    carry the value observed one sampled period earlier on the same
    asset (weight[t-1], rank[t-1], ...). Single helper so the whole
    codebase lags the same way — sort by (asset, date), shift within
    asset, drop the first row per asset.
    """
    return (
        df.sort([by, "date"])
        .with_columns(pl.col(col).shift(periods).over(by).alias(col))
        .drop_nulls([col])
    )


def _assign_quantile_groups(
    df: pl.DataFrame,
    factor_col: str = "factor",
    n_groups: int = 5,
    tie_policy: str = "ordinal",
) -> pl.DataFrame:
    """Assign quantile group labels (0 = bottom, n_groups-1 = top) per date.

    ``tie_policy="ordinal"`` (default): break ties deterministically by
    row order → balanced group sizes, but tied assets end up in different
    buckets (arbitrary but consistent).

    ``tie_policy="average"``: tied assets share an average rank → same
    bucket → honest signal resolution, group sizes may be unbalanced.
    Prefer this for low-cardinality factors (binary, bucketed, or
    categorical signals) where ordinal tie-breaking would inject
    sorting-artifact noise indistinguishable from alpha.

    Returns:
        DataFrame with ``_group`` column appended.
    """
    rank_expr = pl.col(factor_col).rank(method=tie_policy).over("date").alias("_rank")
    return (
        df.with_columns(
            rank_expr,
            pl.len().over("date").alias("_n"),
        )
        .with_columns(
            ((pl.col("_rank") - 1) * n_groups / pl.col("_n"))
            .cast(pl.Int32)
            .clip(0, n_groups - 1)
            .alias("_group")
        )
        .drop("_rank", "_n")
    )


def _compute_tie_ratio(
    df: pl.DataFrame,
    factor_col: str = "factor",
) -> float:
    """Median-across-dates tie ratio ``1 - n_unique / n`` for ``factor_col``.

    A float in [0, 1]: 0 means every per-date cross-section has unique
    factor values (no ties); 1 means every cross-section is fully
    degenerate. Returns ``nan`` when the panel is empty (no dates).

    Used as a diagnostic on quantile-bucketing metrics — callers log a
    warning when the return exceeds ``TIE_RATIO_WARN_THRESHOLD`` and
    stash the value in ``MetricOutput.metadata["tie_ratio"]`` for
    downstream inspection.
    """
    if df.is_empty():
        return float("nan")
    per_date = (
        df.group_by("date")
        .agg(
            pl.col(factor_col).n_unique().alias("_u"),
            pl.len().alias("_n"),
        )
        .with_columns(
            (1.0 - pl.col("_u") / pl.col("_n")).alias("_tr"),
        )
    )
    med = per_date["_tr"].median()
    return float("nan") if med is None else float(med)


def _warn_high_tie_ratio(
    ratio: float,
    metric_name: str,
    tie_policy: str,
) -> None:
    """Emit a ``UserWarning`` when median tie_ratio exceeds the threshold.

    No-op for ``tie_policy="average"`` (the policy already handles ties
    honestly — warning would be noise) or NaN ratios. Uses ``warnings.warn``
    not ``logger`` so the advisory surfaces in notebooks where root logger
    defaults to WARNING. Python's default ``"default"`` filter dedupes
    by (module, lineno, message) so sweep loops naturally emit once.
    """
    if math.isnan(ratio) or ratio <= TIE_RATIO_WARN_THRESHOLD:
        return
    if tie_policy != "ordinal":
        return
    warnings.warn(
        f"{metric_name}: median tie_ratio={ratio:.3f} exceeds "
        f"{TIE_RATIO_WARN_THRESHOLD:.2f}. Ordinal tie-breaking on a "
        f"low-cardinality factor injects sorting-artifact noise. "
        f"Consider tie_policy='average' on the Config, or a coarser "
        f"n_groups.",
        UserWarning,
        stacklevel=3,
    )


def _median_universe_size(df: pl.DataFrame) -> int:
    """Median number of unique assets per date."""
    return int(
        df.group_by("date").agg(pl.col("asset_id").n_unique().alias("n"))["n"].median()
    )


def _signed_car(
    df: pl.DataFrame,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> np.ndarray:
    """Compute signed CAR for event rows (factor ≠ 0).

    ``signed_car = return × sign(factor)``

    Args:
        df: Event-filtered DataFrame (factor ≠ 0 rows only).

    Returns:
        1-D numpy array of signed abnormal returns.
    """
    return df[return_col].to_numpy() * np.sign(df[factor_col].to_numpy())
