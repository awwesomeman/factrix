"""Null size of ``ic_trend``'s Mann-Kendall (Hamed-Rao) trend test.

The series is the per-period Spearman IC of
``make_cs_panel(ic_target=0.0)`` through ``compute_ic``, so the IC has no
time trend to find; ``factor_persistence`` supplies the second null. The
test runs on the non-overlapping subsample at stride ``overlap_periods``.
Measured over 300 replications per cell at seed 20260830 + rep, at a
nominal 5%:

| T | h | phi = 0 | phi = 0.9 |
|---|---|---|---|
| 60 | 1 | 4.0% | 3.0% |
| 60 | 5 | 1.7% | 4.7% |
| 120 | 1 | 4.3% | 5.0% |
| 120 | 5 | 3.3% | 5.3% |
| 240 | 1 | 1.7% | 4.7% |
| 240 | 5 | 3.3% | 4.7% |

Calibrated-to-conservative throughout (1.7-5.3%), and the persistent
factor moves nothing outside the iid band: factor persistence carries
into the *level* of the per-period IC, not into a drift in it, and the
Hamed-Rao variance correction absorbs what serial dependence the IC
series does carry.

Replication count here is cut to keep the module fast, so the bounds
characterise rather than pin.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest
from factrix.datasets import make_cs_panel
from factrix.metrics.ic import compute_ic
from factrix.metrics.trend import ic_trend
from factrix.preprocess import compute_forward_return

SEED = 20260830
N_REPS = 60
N_ASSETS = 50


def _rejection_rate(n_periods: int, horizon: int, persistence: float) -> float:
    rejected = kept = 0
    for rep in range(N_REPS):
        raw = make_cs_panel(
            n_assets=N_ASSETS,
            n_dates=n_periods + horizon + 1,
            ic_target=0.0,
            factor_persistence=persistence,
            rng=SEED + rep,
        )
        panel = compute_forward_return(raw, forward_periods=horizon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = compute_ic(panel)["factor"].select(
                "date", pl.col("ic").alias("value")
            )
            out = ic_trend(series, overlap_periods=horizon)
        if out.p_value is None:
            continue
        kept += 1
        rejected += out.p_value < 0.05
    assert kept, "every replication short-circuited; the null grid is wrong"
    return rejected / kept


@pytest.mark.parametrize("persistence", [0.0, 0.9])
@pytest.mark.parametrize(
    ("n_periods", "horizon"),
    [(60, 1), (60, 5), (120, 1), (120, 5), (240, 1), (240, 5)],
)
def test_mann_kendall_is_calibrated_on_a_true_null(n_periods, horizon, persistence):
    # Upper bound only: at the reduced replication count a conservative cell
    # (1.7% at the full 300 reps) legitimately returns 0/60, so a floor here
    # would pin sampling noise rather than the path.
    assert _rejection_rate(n_periods, horizon, persistence) <= 0.12
