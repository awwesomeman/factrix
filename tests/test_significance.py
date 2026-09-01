"""Tests for factrix._stats."""

import math

import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix._stats import (
    _calc_t_stat,
    _p_value_from_t,
    _significance_marker,
    _t_stat_from_array,
)


class TestCalcTStat:
    def test_basic(self):
        # mean=1.0, std=0.5, n=100 → t = 1.0 / (0.5/10) = 20.0
        assert _calc_t_stat(1.0, 0.5, 100) == pytest.approx(20.0)

    def test_zero_std(self):
        # NOT 0.0: a zero-dispersion sample with a non-zero mean is degenerate
        # in the maximum-evidence direction (t -> +inf), so reporting 0 would
        # have downstream read p=1 -> "no predictive power". NaN says
        # "not computable", the direction R's t.test refuses in.
        assert math.isnan(_calc_t_stat(1.0, 0.0, 100))

    def test_zero_std_zero_mean_is_also_not_computable(self):
        # The 0/0 branch: undefined rather than maximal, but equally not a
        # statistic. One value for both keeps callers on one code path.
        assert math.isnan(_calc_t_stat(0.0, 0.0, 100))

    def test_near_zero_std(self):
        assert math.isnan(_calc_t_stat(1.0, 1e-12, 100))

    def test_zero_n(self):
        assert math.isnan(_calc_t_stat(1.0, 0.5, 0))

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
        assert math.isnan(_t_stat_from_array(np.array([1.0])))

    def test_empty(self):
        assert math.isnan(_t_stat_from_array(np.array([])))

    def test_constant_non_zero_array(self):
        # The reported shape: every observation identical and non-zero.
        assert math.isnan(_t_stat_from_array(np.full(10, 0.03)))


def test_p_value_rejects_unknown_alternative_before_short_sample_fallback():
    with pytest.raises(UserInputError, match="greater"):
        _p_value_from_t(0.0, 1, "grater")  # type: ignore[arg-type]


class TestSignificanceMarker:
    @pytest.mark.parametrize(
        "p, expected",
        [
            (0.001, "***"),
            (0.005, "***"),
            (0.009, "***"),
            (0.01, "**"),
            (0.03, "**"),
            (0.049, "**"),
            (0.05, "*"),
            (0.08, "*"),
            (0.099, "*"),
            (0.10, ""),
            (0.5, ""),
            (1.0, ""),
            (None, ""),
        ],
    )
    def test_markers(self, p, expected):
        assert _significance_marker(p) == expected


class TestZeroVarianceIsNotTheNull:
    """A zero-dispersion sample must never surface as t=0 / p=1.

    That collapse reported "no predictive power" for a sample carrying either
    maximal evidence (every observation identical and non-zero, t -> ±inf) or
    none at all (an undefined 0/0). ``scipy.stats.ttest_1samp`` propagates
    instead and R's ``t.test`` refuses outright ("data are essentially
    constant"); neither lands on the null, and neither does factrix.
    """

    @pytest.mark.parametrize("mean", [0.03, -0.03, 1e-6])
    def test_constant_non_zero_sample_yields_no_statistic(self, mean):
        assert math.isnan(_t_stat_from_array(np.full(30, mean)))

    def test_p_value_of_a_degenerate_t_is_never_one(self):
        # The downstream half of the bug: a 0.0 t fed a p of exactly 1.0.
        p = _p_value_from_t(_t_stat_from_array(np.full(30, 0.03)), 30)
        assert math.isnan(p)
        assert p != 1.0

    def test_dispersion_makes_the_same_mean_testable(self):
        # Control: the identical mean with real dispersion still tests fine,
        # so the NaN above is about degeneracy, not about the mean's size.
        rng = np.random.default_rng(0)
        values = 0.03 + rng.standard_normal(30) * 0.01
        assert math.isfinite(_t_stat_from_array(values))
