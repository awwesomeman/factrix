"""Overlapping-horizon calibration of the Newey-West HAR t-test.

The previous convention — bandwidth ``max(auto_bartlett(T), h - 1)``, SE
``sqrt(LRV / T)``, reference ``t_{T-1}`` — rejects 10-34% at a nominal 5%
on an overlapping MA(h-1) null, i.e. 2-7x its stated size. The three
pieces now in force are the LLSW (2018) ``1.3*sqrt(T)`` bandwidth floored
at ``3(h-1)``, the ``T/(T - L - 1)`` finite-sample scale, and the
fixed-b effective df ``min(1.5*T/L - 1, T/h - 1)``.

Grid and replication count are kept small so the module stays fast; the
full 4000-replication sweep behind the table in
``_newey_west_t_test``'s Notes is not re-run here.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.constants import har_bandwidth
from factrix._stats.hac import (
    _bartlett_long_run_variance,
    _hac_bandwidth_ill_conditioned,
    _har_dof,
    _newey_west_se,
    _newey_west_t_test,
    _resolve_har_lags,
)
from scipy import stats as sp_stats


def _overlapping_null(n_periods: int, horizon: int, rng) -> np.ndarray:
    """Null series of overlapping ``horizon``-period sums of iid normals.

    Exactly the MA(h-1) structure an h-period forward return induces, with
    a true mean of zero, scaled so the marginal variance is 1.
    """
    e = rng.normal(size=n_periods + horizon)
    cumulative = np.cumsum(np.concatenate([[0.0], e]))
    return (
        cumulative[horizon : horizon + n_periods] - cumulative[:n_periods]
    ) / np.sqrt(horizon)


class TestBandwidthRule:
    def test_base_rule_is_llsw(self) -> None:
        # 1.3 * sqrt(120) = 14.24 -> 15; well inside the T/3 = 40 cap.
        assert har_bandwidth(120) == 15
        assert _resolve_har_lags(120, None, None) == 15

    def test_overlap_floor_is_three_h_minus_one(self) -> None:
        # h = 21 at T = 240: 3 * 20 = 60 beats the LLSW 21, cap is 80.
        assert _resolve_har_lags(240, None, 21) == 60
        # h = 5 at T = 240: 3 * 4 = 12 loses to the LLSW 21.
        assert _resolve_har_lags(240, None, 5) == 21

    def test_capped_at_one_third_of_the_sample(self) -> None:
        # 3 * 20 = 60 would exceed T/3 = 20 at T = 60.
        assert _resolve_har_lags(60, None, 21) == 20

    def test_explicit_lags_still_floored_and_capped(self) -> None:
        assert _resolve_har_lags(240, 2, 21) == 60
        assert _resolve_har_lags(60, 500, None) == 20


class TestFiniteSampleScale:
    def test_se_carries_the_dof_scale(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(120)
        lags = _resolve_har_lags(120, None, None)
        hand = np.sqrt(_bartlett_long_run_variance(x - x.mean(), lags) / 120)
        assert _newey_west_se(x) == pytest.approx(
            hand * np.sqrt(120 / (120 - lags - 1)), rel=1e-12
        )

    def test_t_is_read_against_the_effective_df(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(240) + 0.15
        t, p, _ = _newey_west_t_test(x, overlap_periods=5)
        lags = _resolve_har_lags(240, None, 5)
        dof = _har_dof(240, lags, 5)
        assert dof == pytest.approx(min(1.5 * 240 / lags - 1, 240 / 5 - 1))
        assert p == pytest.approx(2 * sp_stats.t.sf(abs(t), dof), rel=1e-12)

    def test_effective_df_caps_at_the_non_overlapping_count(self) -> None:
        # T=240, h=21 -> L=60, so 1.5*T/L - 1 = 5.0 and T/h - 1 = 10.43.
        assert _har_dof(240, 60, 21) == pytest.approx(5.0)
        # T=500, h=21 -> L=60, so 1.5*T/L - 1 = 11.5 beats T/h - 1 = 22.8.
        assert _har_dof(500, 60, 21) == pytest.approx(11.5)


class TestIllConditionedBandwidth:
    def test_fires_when_bandwidth_exceeds_a_fifth_of_the_sample(self) -> None:
        # h = 21 at T = 60 resolves L = 20 and 60 < 100.
        assert _hac_bandwidth_ill_conditioned(60, _resolve_har_lags(60, None, 21))

    def test_silent_in_the_ordinary_regime(self) -> None:
        assert not _hac_bandwidth_ill_conditioned(240, _resolve_har_lags(240, None, 5))


class TestNullSize:
    """Rejection frequency on a true null must sit at or below ~7%."""

    @pytest.mark.parametrize(
        ("n_periods", "horizon"),
        [(120, 1), (120, 5), (240, 5), (240, 21)],
    )
    def test_size_at_nominal_five_percent(self, n_periods: int, horizon: int) -> None:
        n_reps = 400
        rng = np.random.default_rng(20240607)
        rejects = 0
        for _ in range(n_reps):
            x = _overlapping_null(n_periods, horizon, rng)
            _, p, _ = _newey_west_t_test(x, overlap_periods=horizon)
            rejects += p < 0.05
        rate = rejects / n_reps
        # Upper bound is the claim under test. The lower bound only guards
        # against a change that makes the test vacuous (never rejects);
        # 400 reps put the binomial sd at ~1.1pp.
        assert 0.005 <= rate <= 0.09, f"size {rate:.3f} at T={n_periods}, h={horizon}"

    def test_the_old_convention_was_oversized(self) -> None:
        """Pin the regression: the pre-fix recipe over-rejects on this DGP."""
        n_periods, horizon, n_reps = 240, 21, 400
        rng = np.random.default_rng(20240607)
        rejects = 0
        for _ in range(n_reps):
            x = _overlapping_null(n_periods, horizon, rng)
            # Old recipe: bandwidth h-1, no finite-sample scale, t_{T-1}.
            lags = horizon - 1
            se = np.sqrt(_bartlett_long_run_variance(x - x.mean(), lags) / n_periods)
            t = x.mean() / se
            rejects += 2 * sp_stats.t.sf(abs(t), n_periods - 1) < 0.05
        assert rejects / n_reps > 0.12

    def test_power_survives_the_correction(self) -> None:
        """The size fix must not turn the test into a no-op."""
        n_periods, horizon, n_reps = 240, 5, 400
        mu = 2.5 / np.sqrt(n_periods / horizon)
        rng = np.random.default_rng(99)
        rejects = 0
        for _ in range(n_reps):
            x = _overlapping_null(n_periods, horizon, rng) + mu
            _, p, _ = _newey_west_t_test(x, overlap_periods=horizon)
            rejects += p < 0.05
        assert rejects / n_reps > 0.55
