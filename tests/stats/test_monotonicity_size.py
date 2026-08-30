"""Null size of ``monotonicity``'s Patton-Timmermann MR bootstrap p.

The panel is ``make_cs_panel(ic_target=0.0)``, so no bucket ordering
exists; ``factor_persistence`` supplies the second null. Measured over
300 replications per cell at seed 20260830 + rep, ``n_groups=5``,
``n_resamples=499``, at a nominal 5%:

| T | h | phi = 0 | phi = 0.9 |
|---|---|---|---|
| 120 | 1 | 3.7% | 3.3% |
| 120 | 5 | 2.0% | 2.7% |
| 240 | 1 | 7.3% | 6.0% |
| 240 | 5 | 5.0% | 3.3% |

Calibrated-to-conservative throughout (2.0-7.3%); the persistent factor
moves nothing outside the iid band. The conservatism at ``h = 5`` is the
block bootstrap's block-length floor at ``overlap_periods`` absorbing the
MA(h-1) overlap.

Replication count here is cut to keep the module fast, so the bounds
characterise rather than pin.
"""

from __future__ import annotations

import warnings

import pytest
from factrix.datasets import make_cs_panel
from factrix.metrics.monotonicity import monotonicity
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
            out = monotonicity(
                panel,
                overlap_periods=horizon,
                n_groups=5,
                n_resamples=499,
                rng=SEED + rep,
            )["factor"]
        if out.p_value is None:
            continue
        kept += 1
        rejected += out.p_value < 0.05
    assert kept, "every replication short-circuited; the null grid is wrong"
    return rejected / kept


@pytest.mark.parametrize("persistence", [0.0, 0.9])
@pytest.mark.parametrize(
    ("n_periods", "horizon"), [(120, 1), (120, 5), (240, 1), (240, 5)]
)
def test_mr_bootstrap_is_calibrated_on_a_true_null(n_periods, horizon, persistence):
    size = _rejection_rate(n_periods, horizon, persistence)
    # Never liberal beyond the short-sample band every per-period path in
    # this repo sits in. No lower bound: at the reduced replication count a
    # conservative cell (2.0% at the full 300 reps) legitimately returns
    # 0/60, so a floor here would pin sampling noise rather than the path.
    assert size <= 0.12
