"""Identical calls return bit-identical results.

An eager ``group_by`` has no fixed output order, so a float reduction over its
rows (a ``mean`` or numpy ``sum``) adds the same values in a different order
from call to call and differs in the last bits. Every reduction over grouped
rows sorts first. These compare with ``==`` on purpose: a tolerance would hide
exactly the drift being guarded.

The order is decided by the thread scheduler, so a repeat that happens to agree
proves nothing; each case repeats enough times that the unsorted code differed
in every run it was tried on.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from factrix.metrics._helpers import _estimate_within_date_icc
from factrix.metrics._primitives._group_returns import compute_group_returns
from factrix.metrics.event_quality import signal_density

N_ASSETS = 300
N_DATES = 500
REPEATS = 30


def _panel(seed: int, *, event_rate: float) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_ASSETS * N_DATES
    factor = np.where(rng.random(n) < event_rate, rng.choice([-1.0, 1.0], size=n), 0.0)
    return pl.DataFrame(
        {
            "date": np.tile(np.arange(N_DATES), N_ASSETS),
            "asset_id": np.repeat([f"A{i:04d}" for i in range(N_ASSETS)], N_DATES),
            "factor": factor,
            "forward_return": rng.standard_normal(n) * 0.02,
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def _distinct(produce, data: pl.DataFrame) -> set:
    return {produce(data) for _ in range(REPEATS)}


def test_signal_density_value_is_bit_identical() -> None:
    data = _panel(0, event_rate=0.03)
    assert len(_distinct(lambda d: signal_density(d).value, data)) == 1


def test_within_date_icc_is_bit_identical() -> None:
    rng = np.random.default_rng(1)
    p_date = np.repeat(rng.uniform(0.2, 0.8, N_DATES), N_ASSETS)
    hits = pl.DataFrame(
        {
            "date": np.repeat(np.arange(N_DATES), N_ASSETS),
            "_hit": (rng.random(N_DATES * N_ASSETS) < p_date).astype(float),
        }
    ).sample(fraction=0.5, seed=3)
    assert len(_distinct(lambda d: _estimate_within_date_icc(d, "_hit"), hits)) == 1


def test_group_returns_are_bit_identical() -> None:
    data = _panel(3, event_rate=0.5).with_columns(
        pl.col("factor")
        + pl.Series(np.random.default_rng(4).random(N_ASSETS * N_DATES))
    )

    def produce(d: pl.DataFrame) -> tuple:
        out = compute_group_returns(d, overlap_periods=1, n_groups=5)
        return tuple(out["mean_return"].to_list())

    assert len(_distinct(produce, data)) == 1
