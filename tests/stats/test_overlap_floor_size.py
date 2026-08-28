"""Size cost of the overlap floor: ``L = h - 1`` against ``L = 3(h - 1)``.

Isolates the first of the three departures in ``_resolve_har_lags``: every
other piece of the HAR recipe (the ``T / (T - L - 1)`` finite-sample scale,
the ``min(1.5*T/L - 1, T/h - 1)`` effective df) is held at what factrix
does today, and only the overlap floor moves. This is a narrower contrast
than the table in ``_newey_west_t_test``'s Notes, whose "before" column
swaps all three pieces at once.

At ``T = 240, h = 21`` the ``h - 1`` floor does not even bind — the LLSW
base rule ``1.3*sqrt(T)`` resolves 21 lags against a floor of 20 — so the
measured contrast is ``L = 21`` against ``L = 60``, both inside the
``ceil(T / 3) = 80`` cap.

Measured on a null MA(h-1) series of overlapping h-period sums of iid
normals, 40000 replications, seed 20260828: the rejection frequency at a
nominal 5% falls from 10.1% at the ``h - 1`` floor to 6.0% at
``3(h - 1)``. Monte-Carlo standard error is 0.15pp / 0.12pp; a second
sweep at seed 7 gives 10.1% / 6.1%. The replication count here is cut to
keep the module fast, so the bounds characterise rather than pin.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.constants import har_bandwidth
from factrix._stats.hac import (
    _MAX_BANDWIDTH_FRACTION,
    _bartlett_long_run_variance,
    _har_dof,
    _newey_west_t_test,
    _resolve_har_lags,
)
from scipy import stats as sp_stats

T = 240
H = 21
N_REPS = 800
SEED = 20260828


def _overlapping_null(rng: np.random.Generator) -> np.ndarray:
    """Null series of overlapping ``H``-period sums of iid normals, unit variance."""
    e = rng.normal(size=T + H)
    cumulative = np.cumsum(np.concatenate([[0.0], e]))
    return (cumulative[H : H + T] - cumulative[:T]) / np.sqrt(H)


def _bandwidth_under_single_floor() -> int:
    """``_resolve_har_lags`` with the Hansen-Hodrick ``h - 1`` floor instead of ``3(h - 1)``."""
    base = max(har_bandwidth(T), H - 1)
    cap = min(T - 1, max(1, -(-T // _MAX_BANDWIDTH_FRACTION)))
    return max(0, min(base, cap))


def _p_value(values: np.ndarray, lags: int) -> float:
    """The current HAR t-test at an explicit bandwidth, floor bypassed.

    Same SE scale and effective df as :func:`_newey_west_t_test`; passing
    ``lags`` to it directly would be re-floored at ``3(h - 1)``.
    """
    demeaned = values - float(np.mean(values))
    lrv = _bartlett_long_run_variance(demeaned, lags)
    se = np.sqrt(max(lrv / T, 0.0) * (T / max(T - lags - 1, 1)))
    t = float(np.mean(values)) / se
    return float(2 * sp_stats.t.sf(abs(t), _har_dof(T, lags, H)))


class TestBartlettWeightAcrossTheMaBand:
    """The mechanism the floor exists for: weight retained on lags 1..h-1."""

    @staticmethod
    def _mean_weight(lags: int) -> float:
        j = np.arange(1, H)
        return float(np.mean(np.clip(1.0 - j / (lags + 1), 0.0, None)))

    def test_consistency_floor_keeps_about_half_the_band(self) -> None:
        assert self._mean_weight(H - 1) == pytest.approx(0.50, abs=0.005)

    def test_tripled_floor_keeps_about_five_sixths(self) -> None:
        assert self._mean_weight(3 * (H - 1)) == pytest.approx(0.83, abs=0.005)


class TestOverlapFloorSize:
    def test_the_contrast_is_twenty_one_against_sixty_lags(self) -> None:
        assert _bandwidth_under_single_floor() == 21
        assert _resolve_har_lags(T, None, H) == 60

    def test_p_value_helper_matches_the_library_at_the_resolved_bandwidth(
        self,
    ) -> None:
        values = _overlapping_null(np.random.default_rng(1))
        _, p, _ = _newey_west_t_test(values, overlap_periods=H)
        assert _p_value(values, _resolve_har_lags(T, None, H)) == pytest.approx(
            p, rel=1e-12
        )

    def test_tripled_floor_restores_size_at_nominal_five_percent(self) -> None:
        lags_before = _bandwidth_under_single_floor()
        lags_after = _resolve_har_lags(T, None, H)
        rng = np.random.default_rng(SEED)
        rejects_before = rejects_after = 0
        for _ in range(N_REPS):
            values = _overlapping_null(rng)
            rejects_before += _p_value(values, lags_before) < 0.05
            rejects_after += _p_value(values, lags_after) < 0.05
        before = rejects_before / N_REPS
        after = rejects_after / N_REPS
        # 800 reps put the binomial sd at ~1.1pp; bands are deliberately
        # loose so the check characterises rather than pins the sweep.
        assert before >= 0.08, f"before {before:.3f} at L={lags_before}"
        assert after <= 0.09, f"after {after:.3f} at L={lags_after}"
        assert before - after >= 0.02, f"{before:.3f} -> {after:.3f}"
