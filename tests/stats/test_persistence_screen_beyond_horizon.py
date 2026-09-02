"""``serial_correlation_detected`` fires on persistence *beyond* the overlap horizon.

The screen behind the code reads lag-1 autocorrelation on the series
strided at ``overlap_periods``, not on the full overlapping series.
Overlapping h-period forward returns carry an MA(h-1) structure by
construction — lag-1 near ``1 - 1/h``, lag-h near zero — which the HAC
bandwidth floor and the bootstrap block-length floor absorb, and which an
unstrided lag-1 read would flag as a calibration failure it is not.

Two nulls pin the two halves of that claim, at cut replication counts
against the full grids recorded in ``reference/inference-calibration``:

* a per-asset AR(0.9) factor with overlapping returns — persistent factor,
  persistent *unstrided* IC series, nothing left after the stride: the
  screen must stay quiet and the paths must stay calibrated;
* an AR(0.6) per-period signal level at ``overlap_periods=1`` — genuinely
  persistent beyond any horizon: the screen must fire and the plain t must
  be visibly oversized, which is what the code is warning about.
"""

from __future__ import annotations

import datetime as dt
import warnings

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._stats.constants import PERSISTENT_SERIES_AUTOCORR
from factrix._stats.diagnostics import _lag1_autocorr
from factrix._types import MIN_SERIES_PERIODS_HARD
from factrix.inference import NEWEY_WEST, NON_OVERLAPPING, StationaryBootstrap
from factrix.inference.series_mean import (
    _persistent_beyond_horizon,
    _persistent_sample,
)
from factrix.metrics._helpers import _stride_dates
from factrix.metrics._primitives import compute_ic
from factrix.metrics.fm_beta import compute_fm_betas, fm_beta
from factrix.metrics.ic import ic
from factrix.preprocess import compute_forward_return

N_ASSETS = 50
N_REPS = 60
N_RESAMPLES = 199
BASE_SEED = 20260830
NOMINAL = 0.05
SCREEN = "serial_correlation_detected"


def _persistent_factor_ic(
    n_periods: int, overlap_periods: int, seed: int
) -> pl.DataFrame:
    """Per-period IC series of a panel whose factor is a per-asset AR(0.9)."""
    raw = fx.datasets.make_cs_panel(
        n_assets=N_ASSETS,
        n_dates=n_periods + overlap_periods,
        ic_target=0.0,
        signal_horizon=overlap_periods,
        factor_persistence=0.9,
        rng=seed,
    )
    panel = compute_forward_return(raw, forward_periods=overlap_periods)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_ic(panel)["factor"]


def _ar_signal_ic(n_periods: int, phi: float, seed: int) -> pl.DataFrame:
    """Per-period IC series that is AR(phi) in its own right, with no overlap.

    The factor→return signal *level* follows a zero-mean AR(phi) across the
    period grid, so ``E[IC] = 0`` and the IC series inherits phi. At
    ``overlap_periods=1`` the stride is a no-op, so nothing can remove it.
    """
    rng = np.random.default_rng(seed)
    level = np.empty(n_periods)
    level[0] = rng.normal(scale=0.15 / np.sqrt(1.0 - phi * phi))
    for t in range(1, n_periods):
        level[t] = phi * level[t - 1] + rng.normal(scale=0.15)
    factor = rng.normal(size=(n_periods, N_ASSETS))
    fwd = level[:, None] * factor + rng.normal(scale=0.3, size=(n_periods, N_ASSETS))
    dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_periods)]
    panel = pl.DataFrame(
        {
            "date": np.repeat(np.array(dates, dtype="datetime64[ms]"), N_ASSETS),
            "asset_id": np.tile([f"a{i:03d}" for i in range(N_ASSETS)], n_periods),
            "factor": factor.reshape(-1),
            "forward_return": fwd.reshape(-1),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_ic(panel)["factor"]


def _members(seed: int) -> dict[str, object]:
    return {
        "non_overlapping": NON_OVERLAPPING,
        "newey_west": NEWEY_WEST,
        "stationary_bootstrap": StationaryBootstrap(n_resamples=N_RESAMPLES, rng=seed),
    }


def _rates(ic_frames, overlap_periods: int, member_key: str):
    """(firing rate, rejection rate) of one member over a sequence of IC series."""
    fired = rejected = tested = 0
    for seed, ic_df in ic_frames:
        member = _members(seed)[member_key]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ic(ic_df, overlap_periods=overlap_periods, inference=member)
        codes = set(result.warning_codes)
        if "metric_unavailable" in codes:
            continue
        tested += 1
        fired += SCREEN in codes
        rejected += result.p_value is not None and result.p_value < NOMINAL
    assert tested == len(ic_frames)
    return fired / tested, rejected / tested


# --------------------------------------------------------------------------
# Overlap-only persistence: the screen must stay quiet.
# --------------------------------------------------------------------------


def test_overlapping_series_is_persistent_at_lag_one_but_not_after_the_stride():
    """The MA(h-1) signature: high unstrided lag-1, near-zero once strided."""
    ic_df = _persistent_factor_ic(240, 5, BASE_SEED).sort("date")
    full = ic_df["ic"].drop_nulls().drop_nans().to_numpy()
    strided = _stride_dates(ic_df, 5)["ic"].drop_nulls().drop_nans().to_numpy()
    assert _lag1_autocorr(full) > PERSISTENT_SERIES_AUTOCORR
    assert _lag1_autocorr(strided) < PERSISTENT_SERIES_AUTOCORR


@pytest.mark.parametrize(
    "member_key", ["non_overlapping", "newey_west", "stationary_bootstrap"]
)
def test_screen_is_quiet_on_the_persistent_factor_null(member_key):
    """Persistent factor, h=5: the recorded grid fires on 2-7% of draws."""
    frames = [
        (BASE_SEED + rep, _persistent_factor_ic(240, 5, BASE_SEED + rep))
        for rep in range(N_REPS)
    ]
    fire_rate, _ = _rates(frames, 5, member_key)
    assert fire_rate < 0.10


def test_fm_beta_screens_the_strided_beta_series() -> None:
    """Mechanical overlap persistence no longer triggers the FM warning."""
    panel = fx.datasets.make_cs_panel(
        n_assets=N_ASSETS,
        n_dates=240,
        ic_target=0.0,
        signal_horizon=5,
        factor_persistence=0.6,
        rng=0,
    )
    panel = compute_forward_return(panel, forward_periods=5)
    beta_df = compute_fm_betas(panel)["factor"]
    betas = beta_df["beta"].drop_nulls().to_numpy()
    assert _lag1_autocorr(betas) > PERSISTENT_SERIES_AUTOCORR
    assert _lag1_autocorr(betas[::5]) < PERSISTENT_SERIES_AUTOCORR

    result = fm_beta(beta_df, overlap_periods=5)
    assert WarningCode.SERIAL_CORRELATION_DETECTED.value not in result.warning_codes


def test_fm_beta_warning_reports_the_measured_strided_autocorrelation() -> None:
    rng = np.random.default_rng(1026)
    betas = np.empty(120)
    betas[0] = rng.standard_normal()
    for idx in range(1, len(betas)):
        betas[idx] = 0.9 * betas[idx - 1] + rng.standard_normal()
    beta_df = pl.DataFrame({"date": np.arange(len(betas)), "beta": betas})
    measured = _lag1_autocorr(betas)

    with pytest.warns(UserWarning) as caught:
        result = fm_beta(beta_df, overlap_periods=1)

    code = WarningCode.SERIAL_CORRELATION_DETECTED.value
    messages = [str(item.message) for item in caught if code in str(item.message)]
    assert messages
    assert f"lag-1 autocorrelation {measured:.2f}" in messages[0]
    assert f"> {PERSISTENT_SERIES_AUTOCORR}" in messages[0]
    assert code in result.warning_codes


def test_persistent_factor_null_stays_calibrated_without_the_warning():
    """The quiet is honest: size on that null sits near nominal, not above it.

    Recorded at 300 replications: 0.080 / 0.087 / 0.090 across the three
    members at ``T = 240, h = 5``. Bounds are loose — the Monte-Carlo
    standard error at 60 replications is ~2.8pp.
    """
    frames = [
        (BASE_SEED + rep, _persistent_factor_ic(240, 5, BASE_SEED + rep))
        for rep in range(N_REPS)
    ]
    for member_key in ("non_overlapping", "newey_west", "stationary_bootstrap"):
        _, size = _rates(frames, 5, member_key)
        assert size <= 0.20, member_key


# --------------------------------------------------------------------------
# Persistence the stride cannot remove: the screen must fire.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "member_key", ["non_overlapping", "newey_west", "stationary_bootstrap"]
)
def test_screen_fires_on_a_series_persistent_beyond_the_horizon(member_key):
    """AR(0.6) IC series at h=1: the recorded grid fires on 94.7-100% of draws."""
    frames = [
        (BASE_SEED + rep, _ar_signal_ic(240, 0.6, BASE_SEED + rep))
        for rep in range(N_REPS)
    ]
    fire_rate, _ = _rates(frames, 1, member_key)
    assert fire_rate > 0.80


def test_the_warned_regime_really_is_oversized():
    """The code is not cosmetic: the plain t rejects far above nominal there.

    Recorded at 300 replications: 0.347 (plain t), 0.073 (NW), 0.073
    (bootstrap) at ``phi = 0.6, T = 240``.
    """
    frames = [
        (BASE_SEED + rep, _ar_signal_ic(240, 0.6, BASE_SEED + rep))
        for rep in range(N_REPS)
    ]
    _, plain_t_size = _rates(frames, 1, "non_overlapping")
    assert plain_t_size > 0.15


def test_stride_of_one_leaves_the_screen_unchanged():
    """At ``overlap_periods=1`` the stride is a no-op, so the screen is plain lag-1."""
    ic_df = _ar_signal_ic(240, 0.6, BASE_SEED)
    strided = _stride_dates(ic_df.sort("date"), 1)
    assert strided.equals(ic_df.sort("date"))


# --------------------------------------------------------------------------
# The series-periods floor: below it the screen is withheld, not guessed at.
# --------------------------------------------------------------------------


def _ramp_ic_frame(n_periods: int) -> pl.DataFrame:
    """An IC series of ``n_periods`` points with lag-1 autocorrelation near 1.

    A monotone ramp — as persistent as a series can be, so the only thing
    that can keep the screen quiet is the periods floor itself.
    """
    dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_periods)]
    return pl.DataFrame(
        {
            "date": np.array(dates, dtype="datetime64[ms]"),
            "ic": np.arange(n_periods, dtype=float),
            "n_assets": np.full(n_periods, N_ASSETS),
        }
    )


@pytest.mark.parametrize(
    ("n_periods", "expected"),
    [(MIN_SERIES_PERIODS_HARD - 1, False), (MIN_SERIES_PERIODS_HARD, True)],
)
def test_screen_is_withheld_below_the_series_periods_floor(n_periods, expected):
    """9 strongly autocorrelated observations do not fire; 10 do."""
    ic_df = _ramp_ic_frame(n_periods)
    values = ic_df["ic"].to_numpy()
    # The input really is persistent — only the floor can be keeping it quiet.
    assert _lag1_autocorr(values) > PERSISTENT_SERIES_AUTOCORR
    assert _persistent_sample(values) is expected
    assert _persistent_beyond_horizon(ic_df, "ic", 1) is expected


def test_the_floor_is_the_stride_survivor_count_not_the_input_length():
    """A long input strided below the floor is withheld all the same."""
    # 27 periods strided at 21 leaves 2 survivors, below the floor.
    ic_df = _ramp_ic_frame(27)
    assert _lag1_autocorr(ic_df["ic"].to_numpy()) > PERSISTENT_SERIES_AUTOCORR
    assert _persistent_beyond_horizon(ic_df, "ic", 21) is False
    # The same input at stride 1 keeps all 27 and fires.
    assert _persistent_beyond_horizon(ic_df, "ic", 1) is True
