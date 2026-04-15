"""Tests for factorlib.tools.series.significance."""

import numpy as np
import pytest

from factorlib.tools.series.significance import (
    _calc_t_stat,
    _significance_marker,
    _t_stat_from_array,
)


class TestCalcTStat:
    def test_basic(self):
        # mean=1.0, std=0.5, n=100 → t = 1.0 / (0.5/10) = 20.0
        assert _calc_t_stat(1.0, 0.5, 100) == pytest.approx(20.0)

    def test_zero_std(self):
        assert _calc_t_stat(1.0, 0.0, 100) == 0.0

    def test_near_zero_std(self):
        assert _calc_t_stat(1.0, 1e-12, 100) == 0.0

    def test_zero_n(self):
        assert _calc_t_stat(1.0, 0.5, 0) == 0.0

    def test_negative_mean(self):
        assert _calc_t_stat(-2.0, 0.5, 100) < 0


class TestTStatFromArray:
    def test_known_values(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t = _t_stat_from_array(arr)
        # mean=3.0, std=sqrt(2.5)≈1.581, n=5 → t = 3.0/(1.581/sqrt(5))
        expected = 3.0 / (np.std(arr, ddof=1) / np.sqrt(5))
        assert t == pytest.approx(expected)

    def test_single_element(self):
        assert _t_stat_from_array(np.array([1.0])) == 0.0

    def test_empty(self):
        assert _t_stat_from_array(np.array([])) == 0.0


class TestSignificanceMarker:
    @pytest.mark.parametrize("p, expected", [
        (0.001, "***"), (0.005, "***"), (0.009, "***"),
        (0.01, "**"), (0.03, "**"), (0.049, "**"),
        (0.05, "*"), (0.08, "*"), (0.099, "*"),
        (0.10, ""), (0.5, ""), (1.0, ""),
        (None, ""),
    ])
    def test_markers(self, p, expected):
        assert _significance_marker(p) == expected
