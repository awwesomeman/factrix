"""Shared helpers for ``run_metrics`` / ``run_metrics_chunked`` /
``run_metrics_iter`` tests.

The three test files all need the same multi-factor panel factory,
factor-column selector, and bundle equality check. Centralised here
so a schema or fixture tweak lands in one place.
"""

from __future__ import annotations

import polars as pl
from factrix._run_metrics import MetricsBundle
from factrix.datasets import make_multi_factor_panel
from factrix.preprocess import compute_forward_return


def make_multi_panel(
    *, n_factors: int = 6, n_dates: int = 200, n_assets: int = 40, seed: int = 0
) -> pl.DataFrame:
    """Multi-factor panel with `forward_return` attached, ready for run_metrics."""
    raw = make_multi_factor_panel(
        n_factors=n_factors, n_dates=n_dates, n_assets=n_assets, seed=seed
    )
    return compute_forward_return(raw, forward_periods=5)


def factor_cols(panel: pl.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("factor_")]


def bundle_equals(a: MetricsBundle, b: MetricsBundle) -> bool:
    """Compare two MetricsBundle instances by identity + per-metric output."""
    if a.identity != b.identity:
        return False
    if set(a.metrics) != set(b.metrics):
        return False
    return all(repr(a.metrics[name]) == repr(b.metrics[name]) for name in a.metrics)
