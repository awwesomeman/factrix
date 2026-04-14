"""Shared fixtures for factorlib unit tests."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def tiny_panel() -> pl.DataFrame:
    """3 dates × 5 assets, perfect monotonic factor-return alignment.

    IC = 1.0 per date, monotonicity = 1.0, spread hand-computable.
    """
    dates = [datetime(2024, 1, 1) + timedelta(weeks=i) for i in range(3)]
    assets = ["A", "B", "C", "D", "E"]
    factors = [1.0, 2.0, 3.0, 4.0, 5.0]
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]

    rows = []
    for d in dates:
        for a, f, r in zip(assets, factors, returns):
            rows.append({"date": d, "asset_id": a, "factor": f, "forward_return": r})

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def noisy_panel() -> pl.DataFrame:
    """20 dates × 30 assets, seeded random with positive IC.

    return = 0.3 * factor + 0.7 * noise → positive but noisy IC.
    """
    rng = np.random.default_rng(42)
    n_dates, n_assets = 20, 30
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        r = 0.3 * f + 0.7 * noise
        for i, a in enumerate(assets):
            rows.append({
                "date": d,
                "asset_id": a,
                "factor": float(f[i]),
                "forward_return": float(r[i]),
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def ic_series_positive() -> pl.DataFrame:
    """50-period positive IC series (date + value)."""
    rng = np.random.default_rng(42)
    values = rng.normal(0.05, 0.02, 50)
    values = np.abs(values)  # ensure all positive
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
    return pl.DataFrame({
        "date": dates,
        "value": values,
    }).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.fixture
def ic_series_sign_flip() -> pl.DataFrame:
    """40-period IC: first 20 positive, last 20 negative."""
    rng = np.random.default_rng(42)
    pos = rng.normal(0.05, 0.01, 20)
    pos = np.abs(pos)
    neg = -np.abs(rng.normal(0.03, 0.01, 20))
    values = np.concatenate([pos, neg])
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    return pl.DataFrame({
        "date": dates,
        "value": values,
    }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
