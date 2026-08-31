"""Null size of ``STATIONARY_BOOTSTRAP`` on a long-short spread series.

``ic``'s bootstrap size table measures the IC series; a spread series is a
cross-sectional bucket difference with a different distribution, so
admitting the member to ``quantile_spread`` / ``quantile_spread_vw`` /
``k_spread`` rests on its own measurement. The full 3x3 grid
(``T`` in {60, 120, 240} x ``h`` in {1, 5, 21}, 500 replications per cell,
``n_resamples=499``, base seed 20260830) is recorded in
``reference/inference-calibration``; measured there, the bootstrap
rejects 4.8-9.0% at a nominal 5% across every measurable cell and
``NEWEY_WEST`` 2.4-7.6%.

This module re-runs two of those cells at a cut replication count so the
suite stays fast: the bounds characterise the result rather than pin it,
and the Monte-Carlo standard error at 120 replications is ~2pp.
"""

from __future__ import annotations

import warnings

import factrix as fx
import pytest
from factrix.inference import NEWEY_WEST, StationaryBootstrap
from factrix.metrics.quantile import quantile_spread
from factrix.preprocess import compute_forward_return

N_ASSETS = 50
N_REPS = 120
N_RESAMPLES = 499
BASE_SEED = 20260830
NOMINAL = 0.05


def _rejection_rates(n_periods: int, overlap_periods: int) -> tuple[float, float]:
    """(Newey-West, stationary-bootstrap) rejection rate on a zero-signal panel."""
    rejected = {"nw": 0, "sb": 0}
    tested = 0
    for rep in range(N_REPS):
        seed = BASE_SEED + 1000 * rep + 7 * overlap_periods + n_periods
        raw = fx.datasets.make_cs_panel(
            n_assets=N_ASSETS,
            n_dates=n_periods + overlap_periods,
            ic_target=0.0,
            rng=seed,
        )
        panel = compute_forward_return(raw, forward_periods=overlap_periods)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nw = quantile_spread(
                panel, overlap_periods=overlap_periods, inference=NEWEY_WEST
            )["factor"]
            sb = quantile_spread(
                panel,
                overlap_periods=overlap_periods,
                inference=StationaryBootstrap(n_resamples=N_RESAMPLES, rng=seed),
            )["factor"]
        tested += 1
        rejected["nw"] += nw.p_value < NOMINAL
        rejected["sb"] += sb.p_value < NOMINAL
    return rejected["nw"] / tested, rejected["sb"] / tested


@pytest.mark.parametrize(
    ("n_periods", "overlap_periods"),
    [(120, 5), (240, 21)],
)
def test_bootstrap_size_is_not_wildly_liberal(n_periods, overlap_periods):
    """The bootstrap stays in the same neighbourhood as the recorded table."""
    nw_rate, sb_rate = _rejection_rates(n_periods, overlap_periods)
    # Loose: 120 replications carry a ~2pp Monte-Carlo standard error, and the
    # recorded cells sit at 7.2% / 6.8%.
    assert 0.01 <= sb_rate <= 0.16
    # Newey-West is conservative at the long horizon; it must not blow up.
    assert nw_rate <= 0.16


def test_long_horizon_short_panel_is_refused_not_measured():
    """``T = 60, h = 21`` has no rejection rate: the metric refuses the panel."""
    raw = fx.datasets.make_cs_panel(
        n_assets=N_ASSETS, n_dates=81, ic_target=0.0, rng=BASE_SEED
    )
    panel = compute_forward_return(raw, forward_periods=21)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = quantile_spread(
            panel,
            overlap_periods=21,
            inference=StationaryBootstrap(n_resamples=N_RESAMPLES, rng=1),
        )["factor"]
    assert "metric_unavailable" in result.warning_codes
