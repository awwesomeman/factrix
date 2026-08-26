"""``spanning_alpha``'s HAC bandwidth honours the overlap horizon.

The alpha t-stat used ``auto_bartlett(T)`` alone, with no ``h - 1`` floor.
Spread series built from ``h``-period overlapping forward returns carry
MA(``h-1``) residual autocorrelation (Hansen-Hodrick 1980), so a Bartlett
kernel run at fewer than ``h - 1`` lags leaves that autocorrelation in the
residual and understates the standard error. Every other HAC path in factrix
floors at ``h - 1`` via ``_resolve_nw_lags``; this one now does too.

The reference values here are computed by hand from the Newey-West (1987)
definition, independently of the implementation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._ols import ols_alpha
from factrix._stats.constants import auto_bartlett
from factrix.metrics.spanning import spanning_alpha


def _hand_hac_alpha_t(y: np.ndarray, f: np.ndarray, lags: int) -> float:
    """Newey-West HAC t on the intercept, written out from the definition."""
    n = len(y)
    X = np.column_stack([np.ones(n), f])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta

    scores = X * resid[:, None]
    # Omega_0 + sum_j w_j (Omega_j + Omega_j'), w_j = 1 - j / (lags + 1).
    s = scores.T @ scores
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        omega_j = scores[j:].T @ scores[:-j]
        s = s + w * (omega_j + omega_j.T)

    xtx_inv = np.linalg.inv(X.T @ X)
    cov = xtx_inv @ s @ xtx_inv
    return float(beta[0] / np.sqrt(cov[0, 0]))


def _overlapping_pair(n: int = 200, h: int = 5, seed: int = 17):
    """A candidate/base pair whose residual is an MA(h-1) by construction."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    shocks = rng.normal(0.0, 1.0, n + h)
    # Rolling h-sum of iid shocks: exactly the MA(h-1) an h-period
    # overlapping forward return induces.
    overlap = np.array([shocks[i : i + h].sum() for i in range(n)])
    candidate = 0.4 * base + 0.05 * overlap + 0.02
    return base, candidate


class TestOlsAlphaBandwidth:
    def test_matches_a_hand_computed_newey_west_t(self):
        base, candidate = _overlapping_pair()
        n = len(candidate)
        h = 5
        expected_lags = max(auto_bartlett(n), h - 1)

        out = ols_alpha(candidate, base[:, None], forward_periods=h)
        assert out.alpha_t == pytest.approx(
            _hand_hac_alpha_t(candidate, base, expected_lags), rel=1e-10
        )

    def test_default_is_the_unfloored_auto_rule(self):
        base, candidate = _overlapping_pair()
        n = len(candidate)
        out = ols_alpha(candidate, base[:, None])
        assert out.alpha_t == pytest.approx(
            _hand_hac_alpha_t(candidate, base, auto_bartlett(n)), rel=1e-10
        )

    def test_the_floor_binds_only_when_the_horizon_exceeds_the_auto_rule(self):
        base, candidate = _overlapping_pair(n=200)
        auto = auto_bartlett(200)
        # h - 1 below the auto rule: nothing changes.
        below = ols_alpha(candidate, base[:, None], forward_periods=auto)
        assert below.alpha_t == pytest.approx(
            ols_alpha(candidate, base[:, None]).alpha_t, rel=1e-12
        )
        # h - 1 above it: the wider kernel must move the t.
        above = ols_alpha(candidate, base[:, None], forward_periods=auto + 20)
        assert above.alpha_t != pytest.approx(below.alpha_t, rel=1e-6)
        assert above.alpha_t == pytest.approx(
            _hand_hac_alpha_t(candidate, base, auto + 19), rel=1e-10
        )

    def test_positive_residual_autocorrelation_widens_the_se(self):
        # The point of the floor: absorbing the MA(h-1) must not shrink the SE.
        base, candidate = _overlapping_pair(n=160, h=20)
        unfloored = abs(ols_alpha(candidate, base[:, None]).alpha_t)
        floored = abs(ols_alpha(candidate, base[:, None], forward_periods=20).alpha_t)
        assert floored < unfloored


class TestSpanningAlphaThreadsTheHorizon:
    @staticmethod
    def _frame(values: np.ndarray) -> pl.DataFrame:
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
        return pl.DataFrame({"date": dates, "spread": values}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )

    def test_forward_periods_reaches_the_hac_bandwidth(self):
        base, candidate = _overlapping_pair(n=160, h=20)
        cand_df, base_df = self._frame(candidate), self._frame(base)

        default = spanning_alpha(cand_df, base_spreads={"base": base_df})
        floored = spanning_alpha(
            cand_df, base_spreads={"base": base_df}, forward_periods=20
        )

        assert default.value == pytest.approx(floored.value)  # same point estimate
        assert abs(floored.stat) < abs(default.stat)  # wider kernel, wider SE
        assert floored.stat == pytest.approx(
            _hand_hac_alpha_t(candidate, base, max(auto_bartlett(160), 19)), rel=1e-10
        )
