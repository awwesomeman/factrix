"""Tests for factrix.metrics.corrado_rank."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.corrado_rank import corrado_rank


def _panel(returns: np.ndarray, factor: np.ndarray) -> pl.DataFrame:
    n = len(returns)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "asset_id": ["A"] * n,
            "factor": factor,
            "forward_return": returns,
        }
    )


def _directional_panel(sign: float, n: int = 300, seed: int = 0) -> pl.DataFrame:
    """Baseline noise returns with a handful of events whose own return is
    shifted well into the tail in the direction ``sign * factor``, so the
    event ranks are genuinely extreme rather than incidentally so.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=n)
    factor = np.zeros(n)
    event_idx = np.arange(10, n, 30)
    factor[event_idx] = sign
    returns[event_idx] = sign * 5.0  # push into the extreme tail
    return _panel(returns, factor)


class TestOneSidedPValue:
    def test_anti_predictive_factor_is_not_significant(self):
        """Events sit at the top of the return distribution (rank ≈ +0.5)
        while ``factor`` points the wrong way (-1), so the direction-signed
        rank is negative: z < 0, and a one-sided p should be large.
        """
        rng = np.random.default_rng(0)
        n = 300
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(10, n, 30)
        returns[event_idx] = 5.0  # events are the largest returns...
        factor[event_idx] = -1.0  # ...but the factor calls them down

        result = corrado_rank(_panel(returns, factor))

        assert result.stat < 0
        assert result.p_value > 0.5

    def test_predictive_factor_is_significant(self):
        """Mirror case: factor direction matches the extreme rank at each
        event, so z > 0 and the one-sided p should be small.
        """
        result = corrado_rank(_directional_panel(sign=1.0))

        assert result.stat > 0
        assert result.p_value < 0.05

    def test_p_value_is_one_sided_sf(self):
        """p should equal the upper-tail normal survival function of z,
        not the two-sided 2*sf(|z|)."""
        from scipy import stats as sp_stats

        rng = np.random.default_rng(1)
        n = 150
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(5, n, 15)
        factor[event_idx] = rng.choice([-1.0, 1.0], size=len(event_idx))

        result = corrado_rank(_panel(returns, factor))

        assert result.p_value == sp_stats.norm.sf(result.stat)


class TestNonFiniteReturns:
    """Ranks must be formed over finite returns only."""

    def _panel_with_hole(self, hole, n: int = 300, seed: int = 0) -> pl.DataFrame:
        """Directional panel whose *first non-event* return is replaced."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(10, n, 30)
        factor[event_idx] = 1.0
        returns[event_idx] = 5.0
        returns[0] = hole  # a non-event row
        return _panel(returns, factor)

    def test_null_return_does_not_nan_out_the_statistic(self):
        # Old behaviour: a null return produced a null rank, std(u_all)
        # became NaN, and _calc_t_stat silently returned z=0 -> p=0.5.
        result = corrado_rank(self._panel_with_hole(None))
        assert np.isfinite(result.stat)
        assert result.stat > 0
        assert result.p_value < 0.05

    def test_nan_return_is_not_ranked_as_the_largest_value(self):
        # polars ranks float NaN as the largest value, so an unmasked NaN
        # silently entered the sample as a top-rank observation and shifted
        # every other rank down by one.
        nan_result = corrado_rank(self._panel_with_hole(float("nan")))
        null_result = corrado_rank(self._panel_with_hole(None))
        assert nan_result.stat == pytest.approx(null_result.stat)
        assert nan_result.value == pytest.approx(null_result.value)
        # The non-finite cell is excluded from the pooled-std denominator.
        assert nan_result.metadata["n_total_obs"] == 299

    @pytest.mark.parametrize("hole", [float("nan"), None])
    def test_non_finite_event_row_is_dropped_and_counted(self, hole):
        rng = np.random.default_rng(3)
        n = 300
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(10, n, 30)
        factor[event_idx] = 1.0
        returns[event_idx] = 5.0
        returns[event_idx[0]] = hole  # poison one *event*

        result = corrado_rank(_panel(returns, factor))

        assert result.metadata["n_events_dropped_non_finite"] == 1
        assert result.metadata["n_events"] == len(event_idx) - 1
        assert result.n_obs == len(event_idx) - 1
        assert np.isfinite(result.stat)

    def test_nan_factor_survives_the_event_filter_and_is_dropped(self):
        # `NaN != 0` is True in polars, so a NaN factor reaches the event
        # sample and sign(NaN) would poison u_event.
        rng = np.random.default_rng(5)
        n = 300
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(10, n, 30)
        factor[event_idx] = 1.0
        returns[event_idx] = 5.0
        factor[1] = float("nan")

        result = corrado_rank(_panel(returns, factor))

        assert result.metadata["n_events_dropped_non_finite"] == 1
        assert result.metadata["n_events"] == len(event_idx)
        assert np.isfinite(result.stat)

    def test_clean_panel_reports_zero_drops(self):
        result = corrado_rank(_directional_panel(sign=1.0))
        assert result.metadata["n_events_dropped_non_finite"] == 0
        assert result.metadata["n_total_obs"] == 300
