"""Resolver helpers for per-metric capability attributes.

Each metric module declares optional capability attributes at module
top-level (parallel to the metric callable itself):

- ``per_date_series(input_df) -> pl.DataFrame``: returns ``(date,
  value)`` long-form per-date series. Required for slice-test
  consumers (#176).
- ``min_assets_per_group: int | None``: minimum cross-section bucket
  size for slice-test ``n_groups`` downscale (#153 §5).

This module centralizes the lookup from a metric callable to its
declaring module's capabilities so consumers do not grovel
``sys.modules`` directly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Protocol

import polars as pl


class PerDateSeries(Protocol):
    """Callable contract for ``module.per_date_series``.

    Implementations consume the metric's canonical input DataFrame
    and return a two-column long-form frame with ``date`` and
    ``value``. ``value`` is the per-date scalar that the slice tests
    treat as the metric's time series (per-date information coefficient
    (IC), per-date Fama-MacBeth lambda, per-date hit indicator, etc.).
    """

    def __call__(self, df: pl.DataFrame, /) -> pl.DataFrame: ...

def per_date_series_rename(source_col: str) -> PerDateSeries:
    """Factory: ``per_date_series`` for metrics whose input is already
    a ``(date, source_col)`` per-date frame.

    Returns a callable that projects ``date`` plus ``source_col``
    aliased to ``value`` and drops null rows. Metrics whose per-date
    series needs computation (binary cast, cross-section aggregation,
    etc.) define their own ``per_date_series`` instead.
    """

    def _per_date(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            [pl.col("date"), pl.col(source_col).alias("value")]
        ).drop_nulls()

    return _per_date

def resolve_per_date_series(metric: Callable) -> PerDateSeries:
    """Look up ``per_date_series`` on the module that defines ``metric``.

    Raises ``TypeError`` when the metric's module does not declare the
    attribute — the metric is not eligible for slice tests / other
    consumers that need a per-date series.
    """
    mod = sys.modules[metric.__module__]
    fn = getattr(mod, "per_date_series", None)
    if fn is None:
        raise TypeError(
            f"metric {metric.__name__!r} is not slice-test-eligible: "
            f"module {metric.__module__!r} does not declare a top-level "
            f"`per_date_series(df) -> pl.DataFrame[(date, value)]` callable."
        )
    return fn

def resolve_min_assets_per_group(metric: Callable) -> int | None:
    """Look up ``min_assets_per_group`` on the module that defines ``metric``.

    Returns ``None`` when the attribute is absent or explicitly
    ``None`` — slice tests treat both as "metric does not bucket
    cross-section assets, skip ``n_groups`` downscale".
    """
    mod = sys.modules[metric.__module__]
    return getattr(mod, "min_assets_per_group", None)
