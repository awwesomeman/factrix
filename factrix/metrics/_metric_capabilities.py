"""Resolver helpers for per-metric capability attributes.

Each metric module declares optional capability attributes at module
top-level (parallel to the metric callable itself):

- ``per_date_series(input_df) -> pl.DataFrame``: returns ``(date,
  value)`` long-form per-date series. Required for slice-test
  consumers.
- ``min_assets_per_group: int | None``: minimum cross-section bucket
  size for slice-test ``n_groups`` downscale.

This module centralizes the lookup from a metric callable to its
declaring module's capabilities so consumers do not grovel
``sys.modules`` directly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import polars as pl

if TYPE_CHECKING:
    from factrix.inference import NeweyWest, NonOverlapping


class PerDateSeries(Protocol):
    """Callable contract for ``module.per_date_series``.

    Implementations consume the metric's canonical input DataFrame
    and return a two-column long-form frame with ``date`` and
    ``value``. ``value`` is the per-date scalar that the slice tests
    treat as the metric's time series (per-date information coefficient
    (IC), per-date Fama-MacBeth lambda, per-date hit indicator, etc.).
    """

    def __call__(self, data: pl.DataFrame, /) -> pl.DataFrame: ...


def per_date_series_rename(source_col: str) -> PerDateSeries:
    """Factory: ``per_date_series`` for metrics whose input is already
    a ``(date, source_col)`` per-date frame.

    Returns a callable that projects ``date`` plus ``source_col``
    aliased to ``value`` and drops null rows. Metrics whose per-date
    series needs computation (binary cast, cross-section aggregation,
    etc.) define their own ``per_date_series`` instead.
    """

    def _per_date(data: pl.DataFrame) -> pl.DataFrame:
        return data.select(
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
            f"`per_date_series(data) -> pl.DataFrame[(date, value)]` callable."
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


def resolve_applicable_inference(
    metric: Callable,
) -> frozenset[NonOverlapping | NeweyWest] | None:
    """Look up the inference allowlist for ``metric``.

    Returns the frozenset of inference methods the metric accepts at
    ``inference=``, or ``None`` when the metric exposes no ``inference=``
    knob (a singleton-inference metric). The allowlist is declared once per
    module (``applicable_inference``), but a module may host both an
    ``inference=``-bearing metric and a singleton-inference sibling
    (``quantile_spread`` / ``quantile_spread_vw``) — so the result is gated
    on the callable actually carrying an ``inference`` parameter, not just
    on the module attribute existing.
    """
    if "inference" not in getattr(metric, "_param_names", ()):
        return None
    mod = sys.modules[metric.__module__]
    return getattr(mod, "applicable_inference", None)
