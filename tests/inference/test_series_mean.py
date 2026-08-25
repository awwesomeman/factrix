"""Series-mean inference methods — ``factrix.inference`` members.

Verifies:
- The df-based ``compute(df, *, value_col, forward_periods)`` contract
  matches the underlying ``factrix._stats`` kernels bit-for-bit.
- ``min_periods`` soft floor surfaces ``UNRELIABLE_SE_SHORT_PERIODS``;
  Hansen-Hodrick clamp surfaces ``RECT_KERNEL_NEGATIVE_VARIANCE``.
- Identity ClassVars (``test`` / ``se`` / ``summary``) and the
  ``Inference`` runtime-checkable protocol.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._stats import (
    _hansen_hodrick_t_test,
    _newey_west_t_test,
    _p_value_from_t,
    _resolve_nw_lags,
    _t_stat_from_array,
)
from factrix._stats.bootstrap import _block_bootstrap_diff_p
from factrix._stats.constants import MIN_PERIODS_WARN, auto_bartlett
from factrix.inference import (
    NEWEY_WEST,
    NON_OVERLAPPING,
    STATIONARY_BOOTSTRAP,
    HansenHodrick,
    Inference,
    NeweyWest,
    NonOverlapping,
    StationaryBootstrap,
)


def _series_df(values: np.ndarray) -> pl.DataFrame:
    """Wrap a 1-D array as a date-indexed ``(date, ic)`` DataFrame."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "ic": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestProtocolIdentity:
    @pytest.mark.parametrize(
        "member", [NON_OVERLAPPING, NEWEY_WEST, HansenHodrick(), STATIONARY_BOOTSTRAP]
    )
    def test_satisfies_inference_protocol(self, member: object) -> None:
        assert isinstance(member, Inference)

    def test_test_and_se_labels(self) -> None:
        assert NonOverlapping.test == "t"
        assert NonOverlapping.se == "ols"
        assert NeweyWest.se == "hac"
        assert HansenHodrick.se == "hac"
        assert StationaryBootstrap.test == "bootstrap-mean"
        assert StationaryBootstrap.se == "bootstrap"

    def test_curated_instances_are_singletons_of_their_type(self) -> None:
        assert isinstance(NON_OVERLAPPING, NonOverlapping)
        assert isinstance(NEWEY_WEST, NeweyWest)
        assert isinstance(STATIONARY_BOOTSTRAP, StationaryBootstrap)


def _same(a: float, b: float) -> bool:
    """Equality that treats NaN as a value (the not-computable sentinel)."""
    return (math.isnan(a) and math.isnan(b)) or a == b


class TestNonOverlapping:
    @pytest.mark.parametrize("forward_periods", [1, 5, 10])
    def test_bit_equal_to_kernel(self, forward_periods: int) -> None:
        rng = np.random.default_rng(0)
        series = rng.standard_normal(120) + 0.1
        df = _series_df(series)
        result = NON_OVERLAPPING.compute(
            df, value_col="ic", forward_periods=forward_periods
        )
        sampled = series[::forward_periods]
        assert result.stat == _t_stat_from_array(sampled)
        assert result.p_value == _p_value_from_t(
            _t_stat_from_array(sampled), len(sampled)
        )
        assert result.metadata["stride"] == forward_periods
        assert result.metadata["n_obs_sampled"] == len(sampled)

    def test_sorts_by_date_before_striding(self) -> None:
        # Shuffled rows must produce the same result as sorted input —
        # compute owns the date-ordering.
        series = np.arange(40.0)
        df = _series_df(series)
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=1)
        a = NON_OVERLAPPING.compute(df, value_col="ic", forward_periods=5)
        b = NON_OVERLAPPING.compute(shuffled, value_col="ic", forward_periods=5)
        assert a.stat == b.stat
        assert a.p_value == b.p_value

    def test_drops_nulls(self) -> None:
        series = np.arange(20.0)
        df = _series_df(series).with_columns(
            pl.when(pl.col("ic") == 3.0).then(None).otherwise(pl.col("ic")).alias("ic")
        )
        result = NON_OVERLAPPING.compute(df, value_col="ic", forward_periods=1)
        assert result.metadata["n_obs_original"] == 19

    def test_short_sample_warns(self) -> None:
        result = NON_OVERLAPPING.compute(
            _series_df(np.arange(10.0)), value_col="ic", forward_periods=1
        )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_min_input_periods_scales_with_stride(self) -> None:
        assert NON_OVERLAPPING.min_input_periods(
            5
        ) == 5 * NON_OVERLAPPING.min_input_periods(1)


class TestNeweyWest:
    @pytest.mark.parametrize("forward_periods", [1, 5, 10])
    def test_bit_equal_to_kernel_nw1994(self, forward_periods: int) -> None:
        rng = np.random.default_rng(42)
        series = rng.standard_normal(60)
        df = _series_df(series)
        result = NEWEY_WEST.compute(df, value_col="ic", forward_periods=forward_periods)
        nw_lags = _resolve_nw_lags(
            len(series), auto_bartlett(len(series)), forward_periods
        )
        t_direct, p_direct, _ = _newey_west_t_test(series, lags=nw_lags)
        assert result.stat == t_direct
        assert result.p_value == p_direct
        assert result.metadata["nw_lags"] == nw_lags
        assert result.metadata["prewhitened"] is True
        assert -1.0 < result.metadata["ar1_phi_hat"] < 1.0

    def test_short_series_warns(self) -> None:
        series = np.random.default_rng(0).standard_normal(MIN_PERIODS_WARN - 5)
        result = NEWEY_WEST.compute(
            _series_df(series), value_col="ic", forward_periods=5
        )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_single_observation_is_a_shortage_not_a_degeneracy(self) -> None:
        # NaN, not the former (0.0, 1.0): the kernel cannot run below three
        # observations, and p=1 would have read that shortage as "no signal".
        # It is *not* DEGENERATE_VARIANCE — nothing here shows a collapsed
        # SE — so only the short-sample code fires.
        result = NEWEY_WEST.compute(
            _series_df(np.array([0.0])), value_col="ic", forward_periods=5
        )
        assert math.isnan(result.stat)
        assert math.isnan(result.p_value)
        assert WarningCode.DEGENERATE_VARIANCE not in result.warnings
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings
        assert result.metadata["nw_lags"] == 0

    def test_constant_non_zero_series_is_flagged_not_nulled(self) -> None:
        # The reported shape: identical, non-zero observations. Evidence is
        # maximal, so the one outcome that must NOT appear is p=1.
        result = NEWEY_WEST.compute(
            _series_df(np.full(40, 0.03)), value_col="ic", forward_periods=5
        )
        assert result.p_value != 1.0
        assert math.isnan(result.p_value)
        assert WarningCode.DEGENERATE_VARIANCE in result.warnings


class TestHansenHodrick:
    @pytest.mark.parametrize("forward_periods", [2, 5, 10])
    def test_bit_equal_to_kernel(self, forward_periods: int) -> None:
        rng = np.random.default_rng(42)
        series = rng.standard_normal(60)
        result = HansenHodrick().compute(
            _series_df(series), value_col="ic", forward_periods=forward_periods
        )
        t_direct, p_direct, _, clamped = _hansen_hodrick_t_test(
            series, forward_periods=forward_periods
        )
        # ``==`` is not enough: a clamped kernel makes both sides NaN, and the
        # point of the test is that compute delegates rather than recomputes.
        assert _same(result.stat, t_direct)
        assert _same(result.p_value, p_direct)
        assert result.metadata == {"kernel": "rectangular", "variance_clamped": clamped}

    def test_clamp_surfaces_warning(self) -> None:
        series = np.random.default_rng(0).standard_normal(10)
        result = HansenHodrick().compute(
            _series_df(series), value_col="ic", forward_periods=4
        )
        assert result.metadata["variance_clamped"] is True
        assert WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE in result.warnings


class TestStationaryBootstrap:
    def test_stat_is_observed_mean(self) -> None:
        series = np.random.default_rng(0).standard_normal(80) + 0.2
        result = STATIONARY_BOOTSTRAP.compute(
            _series_df(series), value_col="ic", forward_periods=5
        )
        assert result.stat == pytest.approx(float(series.mean()))
        assert 0.0 < result.p_value <= 1.0

    def test_metadata_reports_reproducible_seed(self) -> None:
        series = np.random.default_rng(1).standard_normal(80)
        result = STATIONARY_BOOTSTRAP.compute(
            _series_df(series), value_col="ic", forward_periods=5
        )
        assert set(result.metadata) == {
            "block_length",
            "n_resamples",
            "scheme",
            "rng_seed",
        }
        p_replay, _ = _block_bootstrap_diff_p(
            series, rng_seed=result.metadata["rng_seed"]
        )
        assert result.p_value == p_replay

    def test_short_sample_warns(self) -> None:
        result = STATIONARY_BOOTSTRAP.compute(
            _series_df(np.arange(10.0)), value_col="ic", forward_periods=1
        )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_min_input_periods_matches_newey_west(self) -> None:
        assert STATIONARY_BOOTSTRAP.min_input_periods(
            5
        ) == NEWEY_WEST.min_input_periods(5)


class TestCleanSeriesDropsNaN:
    def test_nan_dropped_alongside_null(self) -> None:
        from factrix.inference.series_mean import _clean_series

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        df = pl.DataFrame(
            {"date": dates, "v": [1.0, float("nan"), None, 2.0, 3.0]}
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        assert _clean_series(df, "v").to_list() == [1.0, 2.0, 3.0]


class TestPersistenceScreen:
    """Every series-mean member flags a persistent input series.

    Above ``PERSISTENT_SERIES_AUTOCORR`` none of the members is calibrated
    (NW 13–17%, bootstrap 12–19%, plain t 32–34% at nominal 5% for phi=0.6),
    so the code is raised regardless of which member ran; it is advice to
    raise the hurdle, not a reason to switch member.
    """

    @staticmethod
    def _ar1(phi: float, n: int = 240, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        e = np.empty(n)
        e[0] = rng.standard_normal()
        for t in range(1, n):
            e[t] = phi * e[t - 1] + rng.standard_normal() * np.sqrt(1 - phi * phi)
        return e

    @pytest.mark.parametrize(
        "member", [NON_OVERLAPPING, NEWEY_WEST, HansenHodrick(), STATIONARY_BOOTSTRAP]
    )
    def test_persistent_series_is_flagged(self, member) -> None:
        result = member.compute(
            _series_df(self._ar1(0.85)), value_col="ic", forward_periods=1
        )
        assert WarningCode.SERIAL_CORRELATION_DETECTED in result.warnings

    @pytest.mark.parametrize(
        "member", [NON_OVERLAPPING, NEWEY_WEST, HansenHodrick(), STATIONARY_BOOTSTRAP]
    )
    def test_iid_series_is_not_flagged(self, member) -> None:
        result = member.compute(
            _series_df(np.random.default_rng(3).standard_normal(240)),
            value_col="ic",
            forward_periods=1,
        )
        assert WarningCode.SERIAL_CORRELATION_DETECTED not in result.warnings

    @pytest.mark.parametrize(
        ("phi", "h", "flagged"),
        [
            # Reviewer's table: lag-1 of the strided sample vs realised size.
            (0.6, 1, True),  # sample lag-1 0.585, size 32.9%
            (0.6, 5, False),  # 0.051, size 7.9%
            (0.6, 21, False),  # -0.098, size 4.5% — a flag here is a false alarm
            (0.85, 5, True),  # 0.386, size 24.3%
            (0.85, 21, False),  # -0.070, size 5.9%
        ],
    )
    def test_non_overlapping_screens_the_strided_sample(self, phi, h, flagged):
        """``NonOverlapping`` exists to buy independence by striding; the
        screen must judge the subsample the t runs on, not the full series,
        or it fires exactly when the member has done its job."""
        result = NON_OVERLAPPING.compute(
            # 240 * h input rows -> ~240 strided points, so the sample lag-1
            # estimate is stable at every h.
            _series_df(self._ar1(phi, n=240 * h)),
            value_col="ic",
            forward_periods=h,
        )
        assert (WarningCode.SERIAL_CORRELATION_DETECTED in result.warnings) is flagged
