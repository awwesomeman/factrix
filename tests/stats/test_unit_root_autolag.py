"""ADF: auto-lag selection and the corrected upper-tail critical points."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.unit_root import (
    _ADF_CRITS_CONSTANT,
    _adf,
    _adf_pvalue_interp,
    _schwert_maxlag,
)


def test_upper_tail_points_are_fullers_tau_mu():
    # Fuller / MacKinnon tau_mu upper quantiles: 90% -0.44, 95% -0.07,
    # 97.5% +0.23, 99% +0.60. The old table had 0.23 labelled as the 95% point.
    d = dict(_ADF_CRITS_CONSTANT)
    assert d[-0.07] == 0.95
    assert d[0.23] == 0.975
    assert d[0.60] == 0.99
    assert _adf_pvalue_interp(-0.07) == pytest.approx(0.95)
    assert _adf_pvalue_interp(0.23) == pytest.approx(0.975)


def test_schwert_maxlag_matches_statsmodels_rule():
    assert _schwert_maxlag(100) == 12
    assert _schwert_maxlag(250) == 15
    assert _schwert_maxlag(50) == 10


def test_autolag_rejects_stationary_and_keeps_random_walk():
    rng = np.random.default_rng(0)
    n = 400
    e = rng.normal(size=n)
    # MA(4) errors (what an h=5 overlapping series carries): the lag-0
    # regression is mis-sized; auto-lag must still separate the two cases.
    ma = e + 0.6 * np.roll(e, 1) + 0.4 * np.roll(e, 2) + 0.2 * np.roll(e, 3)
    stationary = 0.3 * ma
    walk = np.cumsum(ma)
    _, p_s = _adf(stationary)
    _, p_w = _adf(walk)
    assert p_s < 0.01
    assert p_w > 0.10
    # explicit lags still honoured
    tau_0, _ = _adf(stationary, lags=0)
    assert np.isfinite(tau_0)


def test_short_series_returns_cannot_reject():
    assert _adf(np.arange(8.0)) == (0.0, 1.0)
