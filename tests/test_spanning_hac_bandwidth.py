"""``spanning_alpha``'s HAC reference honours the overlap horizon.

The alpha t-stat is a **single restriction**, so it takes the scalar HAR
recipe (``_resolve_scalar_wald_hac``): the ``1.3*sqrt(T)`` base, the
``3(h - 1)`` overlap floor and the ``ceil(T / 3)`` cap for the bandwidth,
the ``T / (T - L - 1)`` finite-sample variance scale, and the fixed-``b``
effective degrees of freedom. Spread series built from ``h``-period
overlapping forward returns carry MA(``h-1``) residual autocorrelation
(Hansen-Hodrick 1980); the narrow ``max(auto_bartlett(T), h - 1)`` rule this
path used before left it 13.7-46.0% oversized at ``h > 1``.

The reference values here are computed by hand from the Newey-West (1987)
definition, independently of the implementation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._ols import ols_alpha
from factrix._stats.constants import har_bandwidth
from factrix._stats.hac import _resolve_scalar_wald_hac
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
    # The HAR finite-sample variance scale the resolver hands the caller.
    return float(beta[0] / np.sqrt(cov[0, 0] * n / (n - lags - 1)))


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
        expected_lags, _, _ = _resolve_scalar_wald_hac(n, None, h)
        assert expected_lags == 3 * (h - 1) or expected_lags == har_bandwidth(n)

        out = ols_alpha(candidate, base[:, None], overlap_periods=h)
        assert out.alpha_t == pytest.approx(
            _hand_hac_alpha_t(candidate, base, expected_lags), rel=1e-10
        )

    def test_default_is_the_unfloored_har_base_rule(self):
        base, candidate = _overlapping_pair()
        n = len(candidate)
        out = ols_alpha(candidate, base[:, None])
        assert out.alpha_t == pytest.approx(
            _hand_hac_alpha_t(candidate, base, har_bandwidth(n)), rel=1e-10
        )

    def test_the_floor_binds_only_when_the_horizon_exceeds_the_base_rule(self):
        base, candidate = _overlapping_pair(n=200)
        base_lags = har_bandwidth(200)
        # 3(h - 1) below the base rule: nothing changes.
        small_h = base_lags // 3
        below = ols_alpha(candidate, base[:, None], overlap_periods=small_h)
        assert below.alpha_t == pytest.approx(
            ols_alpha(candidate, base[:, None]).alpha_t, rel=1e-12
        )
        # 3(h - 1) above it: the wider kernel must move the t.
        big_h = base_lags
        above = ols_alpha(candidate, base[:, None], overlap_periods=big_h)
        assert above.alpha_t != pytest.approx(below.alpha_t, rel=1e-6)
        assert above.alpha_t == pytest.approx(
            _hand_hac_alpha_t(candidate, base, 3 * (big_h - 1)), rel=1e-10
        )

    def test_absorbing_the_overlap_does_not_sharpen_the_test(self):
        # The point of the floor. The invariant is the p-value, not the SE:
        # at L = 54 on 160 periods the Bartlett long-run variance can come out
        # *smaller* than at L = 16, and the raw |t| with it, but the fixed-b
        # effective df falls faster, so the test does not get sharper.
        from scipy import stats as sp_stats

        base, candidate = _overlapping_pair(n=160, h=20)
        unfloored = ols_alpha(candidate, base[:, None])
        floored = ols_alpha(candidate, base[:, None], overlap_periods=20)

        def _p(out) -> float:
            return float(2 * sp_stats.t.sf(abs(out.alpha_t), out.alpha_dof))

        assert floored.alpha_dof < unfloored.alpha_dof
        assert _p(floored) > _p(unfloored)


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
            cand_df, base_spreads={"base": base_df}, overlap_periods=20
        )

        assert default.value == pytest.approx(floored.value)  # same point estimate
        # Wider kernel, fewer effective degrees of freedom: a weaker test.
        assert floored.p_value > default.p_value
        expected_lags, _, _ = _resolve_scalar_wald_hac(160, None, 20)
        assert floored.stat == pytest.approx(
            _hand_hac_alpha_t(candidate, base, expected_lags), rel=1e-10
        )
