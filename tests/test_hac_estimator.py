"""Tests for ``HACEstimator`` sub-protocol + ``InferenceResult`` (#163).

Verifies:
- ``HACEstimator`` runtime_checkable identity (NW / HH satisfy;
  slice-test estimators on base ``Estimator`` only).
- ``NeweyWest.compute`` / ``HansenHodrick.compute`` are bit-equal to
  the underlying ``factrix._stats`` primitives the v0.11 cell
  procedures call (no behavior drift from the dispatch refactor's
  Protocol layer).
- ``InferenceResult`` carries the ``StatCode`` keys + flat metadata +
  warning frozenset the cell procedure will stitch into ``FactorProfile``.
- ``min_periods`` soft floor surfaces ``UNRELIABLE_SE_SHORT_PERIODS``.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._codes import StatCode, WarningCode
from factrix._stats import (
    _hansen_hodrick_t_test,
    _newey_west_t_test,
    _resolve_nw_lags,
)
from factrix._stats.constants import MIN_PERIODS_WARN, auto_bartlett
from factrix.stats import (
    BlockBootstrap,
    Estimator,
    HACEstimator,
    HansenHodrick,
    InferenceResult,
    NeweyWest,
    WaldNWCluster,
    WaldTwoWayCluster,
)


class TestProtocolIdentity:
    """Layered ``isinstance`` semantics for base vs HAC sub-protocol."""

    def test_nw_satisfies_both(self) -> None:
        nw = NeweyWest()
        assert isinstance(nw, Estimator)
        assert isinstance(nw, HACEstimator)

    def test_hh_satisfies_both(self) -> None:
        hh = HansenHodrick()
        assert isinstance(hh, Estimator)
        assert isinstance(hh, HACEstimator)

    @pytest.mark.parametrize(
        "estimator", [WaldNWCluster(), WaldTwoWayCluster(), BlockBootstrap()]
    )
    def test_slice_estimators_base_only(self, estimator: object) -> None:
        assert isinstance(estimator, Estimator)
        assert not isinstance(estimator, HACEstimator)


class TestNeweyWestCompute:
    """``NeweyWest.compute`` matches ``_newey_west_t_test`` bit-for-bit."""

    @pytest.mark.parametrize("forward_periods", [1, 5, 10])
    def test_bit_equal_to_primitive(self, forward_periods: int) -> None:
        rng = np.random.default_rng(42)
        series = rng.standard_normal(60)
        result = NeweyWest().compute(series, forward_periods=forward_periods)

        n = len(series)
        nw_lags = _resolve_nw_lags(n, auto_bartlett(n), forward_periods)
        t_direct, p_direct, _ = _newey_west_t_test(series, lags=nw_lags)

        assert result.stat == t_direct
        assert result.p_value == p_direct
        assert result.metadata == {"nw_lags": nw_lags}

    def test_inference_result_keys(self) -> None:
        result = NeweyWest().compute(
            np.random.default_rng(0).standard_normal(40), forward_periods=5
        )
        assert isinstance(result, InferenceResult)
        assert result.stat_name is StatCode.T_NW
        assert result.p_name is StatCode.P_NW

    def test_short_series_emits_warning(self) -> None:
        series = np.random.default_rng(0).standard_normal(MIN_PERIODS_WARN - 5)
        result = NeweyWest().compute(series, forward_periods=5)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_long_series_no_warning(self) -> None:
        series = np.random.default_rng(0).standard_normal(MIN_PERIODS_WARN + 10)
        result = NeweyWest().compute(series, forward_periods=5)
        assert result.warnings == frozenset()

    def test_degenerate_series_does_not_crash(self) -> None:
        # ``n < 2`` is the existing call-site guard; compute must not
        # raise, and the primitive returns the conservative (0, 1).
        result = NeweyWest().compute(np.array([0.0]), forward_periods=5)
        assert result.stat == 0.0
        assert result.p_value == 1.0
        assert result.metadata["nw_lags"] == 0


class TestHansenHodrickCompute:
    """``HansenHodrick.compute`` matches ``_hansen_hodrick_t_test`` bit-for-bit."""

    @pytest.mark.parametrize("forward_periods", [2, 5, 10])
    def test_bit_equal_to_primitive(self, forward_periods: int) -> None:
        rng = np.random.default_rng(42)
        series = rng.standard_normal(60)
        result = HansenHodrick().compute(series, forward_periods=forward_periods)

        t_direct, p_direct, _, clamped = _hansen_hodrick_t_test(
            series, forward_periods=forward_periods
        )

        assert result.stat == t_direct
        assert result.p_value == p_direct
        assert result.metadata == {
            "kernel": "rectangular",
            "variance_clamped": clamped,
        }

    def test_inference_result_keys(self) -> None:
        result = HansenHodrick().compute(
            np.random.default_rng(0).standard_normal(40), forward_periods=5
        )
        assert isinstance(result, InferenceResult)
        assert result.stat_name is StatCode.T_HH
        assert result.p_name is StatCode.P_HH

    def test_clamp_surfaces_warning(self) -> None:
        series = np.random.default_rng(0).standard_normal(10)
        result = HansenHodrick().compute(series, forward_periods=4)
        assert result.metadata["variance_clamped"] is True
        assert WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE in result.warnings


class TestMinPeriodsContract:
    """``min_periods`` advertises the SE-validity soft floor."""

    @pytest.mark.parametrize("estimator", [NeweyWest(), HansenHodrick()])
    def test_min_periods_is_warn_floor(self, estimator: HACEstimator) -> None:
        assert estimator.min_periods == MIN_PERIODS_WARN
