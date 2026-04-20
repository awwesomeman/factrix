"""Shared helpers used across multiple tool modules.

These are internal utilities — not part of the public API.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import MetricOutput


def _short_circuit_output(
    name: str,
    reason: str,
    **extra_metadata: object,
) -> MetricOutput:
    """Canonical short-circuit ``MetricOutput`` for "cannot compute".

    Reason vocabulary (matches ``_insufficient_metrics`` prefixes):
        - ``insufficient_<thing>`` — data shortage (dropped from BHY)
        - ``no_<thing>`` — missing input / missing config / missing data

    ``p_value=1.0`` is the conservative default so BHY treats short-circuited
    metrics as rejected rather than crashing; ``_pv`` reads the same key.

    Use this instead of hand-rolling ``MetricOutput(value=0.0, stat=None,
    significance="", metadata={"reason": ..., "p_value": 1.0, ...})``.
    """
    return MetricOutput(
        name=name,
        value=0.0,
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
    return (
        "abnormal_return"
        if "abnormal_return" in df.columns
        else "forward_return"
    )


def _sample_non_overlapping(
    df: pl.DataFrame,
    forward_periods: int,
) -> pl.DataFrame:
    """Keep every N-th date to avoid overlapping forward returns.

    Args:
        df: DataFrame with a ``date`` column.
        forward_periods: Sampling interval.

    Returns:
        Filtered DataFrame containing only sampled dates.
    """
    sampled = df["date"].unique().sort().gather_every(forward_periods)
    return df.filter(pl.col("date").is_in(sampled.implode()))


def _assign_quantile_groups(
    df: pl.DataFrame,
    factor_col: str = "factor",
    n_groups: int = 5,
) -> pl.DataFrame:
    """Assign quantile group labels (0 = bottom, n_groups-1 = top) per date.

    Uses ``rank(method="ordinal")`` to break ties deterministically,
    ensuring balanced group sizes even when many assets share the same
    factor value. Tie-breaking order is arbitrary but consistent.

    Returns:
        DataFrame with ``_group`` column appended.
    """
    # WHY: "ordinal" assigns unique ranks even for tied values,
    # preventing multiple assets from clustering in the same group.
    # "average" would give tied assets the same rank → unbalanced groups.
    return (
        df.with_columns(
            pl.col(factor_col).rank(method="ordinal").over("date").alias("_rank"),
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


def _median_universe_size(df: pl.DataFrame) -> int:
    """Median number of unique assets per date."""
    return int(
        df.group_by("date")
        .agg(pl.col("asset_id").n_unique().alias("n"))
        ["n"].median()
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


