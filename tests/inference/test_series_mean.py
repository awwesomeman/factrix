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
from factrix._stats.constants import MIN_PERIODS_WARN, auto_bartlett
from factrix.inference import (
    NEWEY_WEST,
    NON_OVERLAPPING,
    HansenHodrick,
    Inference,
    NeweyWest,
    NonOverlapping,
)


def _series_df(values: np.ndarray) -> pl.DataFrame:
    """Wrap a 1-D array as a date-indexed ``(date, ic)`` DataFrame."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "ic": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestProtocolIdentity:
    @pytest.mark.parametrize("member", [NON_OVERLAPPING, NEWEY_WEST, HansenHodrick()])
    def test_satisfies_inference_protocol(self, member: object) -> None:
        assert isinstance(member, Inference)

    def test_test_and_se_labels(self) -> None:
        assert NonOverlapping.test == "t"
        assert NonOverlapping.se == "ols"
        assert NeweyWest.se == "hac"
        assert HansenHodrick.se == "hac"

    def test_curated_instances_are_singletons_of_their_type(self) -> None:
        assert isinstance(NON_OVERLAPPING, NonOverlapping)
        assert isinstance(NEWEY_WEST, NeweyWest)


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
        assert result.metadata == {"nw_lags": nw_lags}

    def test_short_series_warns(self) -> None:
        series = np.random.default_rng(0).standard_normal(MIN_PERIODS_WARN - 5)
        result = NEWEY_WEST.compute(
            _series_df(series), value_col="ic", forward_periods=5
        )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_degenerate_series_does_not_crash(self) -> None:
        result = NEWEY_WEST.compute(
            _series_df(np.array([0.0])), value_col="ic", forward_periods=5
        )
        assert result.stat == 0.0
        assert result.p_value == 1.0
        assert result.metadata["nw_lags"] == 0


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
        assert result.stat == t_direct
        assert result.p_value == p_direct
        assert result.metadata == {"kernel": "rectangular", "variance_clamped": clamped}

    def test_clamp_surfaces_warning(self) -> None:
        series = np.random.default_rng(0).standard_normal(10)
        result = HansenHodrick().compute(
            _series_df(series), value_col="ic", forward_periods=4
        )
        assert result.metadata["variance_clamped"] is True
        assert WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE in result.warnings
