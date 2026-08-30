"""Null size of ``common_beta``'s calendar-time-SE cross-asset t.

The null is COMMON-scope, matching the metric's cell: one AR(phi) factor
series broadcast to every asset, drawn from an RNG stream independent of
the ``make_cs_panel(ic_target=0.0)`` prices, so every true per-asset beta
is zero while the assets still share the cross-sectional return
correlation the panel generates. Measured over 300 replications per cell
at seed 20260830 + rep (factor stream 20260830 + 10000 + rep), 50 assets,
at a nominal 5%:

| T | h | phi = 0 | phi = 0.9 |
|---|---|---|---|
| 60 | 1 | 6.7% | 5.7% |
| 60 | 5 | 7.3% | 3.3% |
| 120 | 1 | 6.0% | 6.7% |
| 120 | 5 | 6.7% | 2.7% |
| 240 | 1 | 5.3% | 4.7% |
| 240 | 5 | 5.3% | 2.0% |

Calibrated across the grid (2.0-7.3%), with the worst cells at the short
end where the Newey-West variance of the equal-weight portfolio slope is
estimated from fewest periods. The persistent factor is the *conservative*
direction at ``h = 5`` (3.3 / 2.7 / 2.0% against 7.3 / 6.7 / 5.3%): a
phi = 0.9 regressor read through an overlapping forward return leaves more
serial dependence for the Newey-West kernel behind ``V_EW`` to pick up, so
the SE widens where the iid cross-asset t would not have.

Replication count here is cut to keep the module fast, so the bounds
characterise rather than pin.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix.datasets import make_cs_panel
from factrix.metrics._primitives import compute_common_betas
from factrix.metrics.common_beta import common_beta
from factrix.preprocess import compute_forward_return

SEED = 20260830
FACTOR_SEED_OFFSET = 10_000
N_REPS = 60
N_ASSETS = 50


def _common_scope_panel(
    n_periods: int, horizon: int, persistence: float, rep: int
) -> pl.DataFrame:
    """One AR(phi) factor shared by every asset, independent of the returns."""
    raw = make_cs_panel(
        n_assets=N_ASSETS,
        n_dates=n_periods + horizon + 1,
        ic_target=0.0,
        rng=SEED + rep,
    )
    dates = raw["date"].unique().sort()
    rng = np.random.default_rng(SEED + FACTOR_SEED_OFFSET + rep)
    innovations = rng.standard_normal(len(dates))
    factor = np.empty(len(dates))
    factor[0] = innovations[0]
    scale = np.sqrt(1 - persistence**2)
    for i in range(1, len(dates)):
        factor[i] = persistence * factor[i - 1] + scale * innovations[i]
    common = pl.DataFrame({"date": dates, "_common_factor": factor})
    panel = (
        raw.drop("factor")
        .join(common, on="date", how="inner")
        .rename({"_common_factor": "factor"})
    )
    return compute_forward_return(panel, forward_periods=horizon)


def _rejection_rate(n_periods: int, horizon: int, persistence: float) -> float:
    rejected = kept = calendar = 0
    for rep in range(N_REPS):
        panel = _common_scope_panel(n_periods, horizon, persistence, rep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            betas = compute_common_betas(panel, overlap_periods=horizon)["factor"]
            out = common_beta(betas)
        if out.p_value is None:
            continue
        kept += 1
        rejected += out.p_value < 0.05
        calendar += bool(out.metadata.get("calendar_time_se_applied"))
    assert kept, "every replication short-circuited; the null grid is wrong"
    # The calendar-time SE is load-bearing for the size above, so pin that it
    # ran rather than the iid fallback.
    assert calendar == kept
    return rejected / kept


@pytest.mark.parametrize("persistence", [0.0, 0.9])
@pytest.mark.parametrize(
    ("n_periods", "horizon"),
    [(60, 1), (60, 5), (120, 1), (120, 5), (240, 1), (240, 5)],
)
def test_calendar_time_se_is_calibrated_on_a_common_scope_null(
    n_periods, horizon, persistence
):
    # Upper bound only: at the reduced replication count a conservative cell
    # (2.0% at the full 300 reps) legitimately returns 0/60, so a floor here
    # would pin sampling noise rather than the path.
    assert _rejection_rate(n_periods, horizon, persistence) <= 0.12
