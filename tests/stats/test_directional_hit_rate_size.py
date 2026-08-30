"""Null size of ``directional_hit_rate``'s Pesaran-Timmermann test.

The panel is ``make_cs_panel(ic_target=0.0)``, so ``sign(factor)`` carries
no information about ``sign(forward_return)``; ``factor_persistence``
supplies the second null. The Kolari-Pynnonen within-period deflation
fires on every cell — the null panel is a cross-section, so the pooled
``(date, asset)`` directional trials are not independent. Measured over
300 replications per cell at seed 20260830 + rep, at a nominal 5%:

| T | h | phi = 0 | phi = 0.9 |
|---|---|---|---|
| 120 | 1 | 4.3% | 5.3% |
| 120 | 5 | 3.0% | 4.7% |
| 240 | 1 | 4.0% | 6.7% |
| 240 | 5 | 6.0% | 7.7% |

Calibrated across the grid (3.0-7.7%). The deflation is what keeps it
there: the raw pooled S_n treats 50 same-date names as 50 independent
trials.

Replication count here is cut to keep the module fast, so the bounds
characterise rather than pin.
"""

from __future__ import annotations

import warnings

import pytest
from factrix.datasets import make_cs_panel
from factrix.metrics.directional_hit_rate import directional_hit_rate
from factrix.preprocess import compute_forward_return

SEED = 20260830
N_REPS = 60
N_ASSETS = 50


def _rejection_rate(n_periods: int, horizon: int, persistence: float) -> float:
    rejected = kept = deflated = 0
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
            out = directional_hit_rate(panel, overlap_periods=horizon)
        if out.p_value is None:
            continue
        kept += 1
        rejected += out.p_value < 0.05
        deflated += bool(out.metadata.get("kolari_pynnonen_applied"))
    assert kept, "every replication short-circuited; the null grid is wrong"
    # The deflation is load-bearing for the size above, so pin that it ran.
    assert deflated == kept
    return rejected / kept


@pytest.mark.parametrize("persistence", [0.0, 0.9])
@pytest.mark.parametrize(
    ("n_periods", "horizon"), [(120, 1), (120, 5), (240, 1), (240, 5)]
)
def test_pt_test_is_calibrated_on_a_true_null(n_periods, horizon, persistence):
    assert _rejection_rate(n_periods, horizon, persistence) <= 0.12
