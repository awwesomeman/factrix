"""Minimal ADF + macro_common.factor_persistent rule."""

from __future__ import annotations

import numpy as np
import pytest

from factorlib._stats import _adf, _adf_pvalue_interp


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_adf_stationary_series_rejects_unit_root(rng):
    y = rng.standard_normal(300)
    tau, p = _adf(y)
    assert tau < -2.86, f"white noise should reject unit root at 5%, got τ={tau}"
    assert p < 0.05


def test_adf_random_walk_fails_to_reject(rng):
    y = np.cumsum(rng.standard_normal(300))
    tau, p = _adf(y)
    assert tau > -2.86, f"random walk should not reject unit root, got τ={tau}"
    assert p > 0.10


def test_adf_highly_persistent_ar1_flags_as_unit_root_ish(rng):
    rho = 0.98
    eps = rng.standard_normal(400)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = rho * y[t - 1] + eps[t]
    tau, p = _adf(y)
    assert p > 0.05, f"ρ=0.98 should not reject at 5%, got p={p}"


def test_adf_short_series_returns_degenerate():
    tau, p = _adf(np.array([1.0, 2.0, 3.0]))
    assert (tau, p) == (0.0, 1.0)


@pytest.mark.parametrize(
    "tau,expected_range",
    [
        (-5.0, (0.0, 0.005)),
        (-3.0, (0.02, 0.08)),
        (-2.0, (0.10, 0.45)),
        (0.5, (0.90, 1.0)),
    ],
)
def test_adf_pvalue_interpolation_ranges(tau, expected_range):
    lo, hi = expected_range
    p = _adf_pvalue_interp(tau)
    assert lo <= p <= hi, f"τ={tau}: p={p} outside {expected_range}"


def test_macro_common_factor_persistent_rule_fires(rng):
    """End-to-end: build a MacroCommonProfile from a persistent factor and
    confirm the macro_common.factor_persistent diagnostic fires."""
    from datetime import datetime, timedelta

    import polars as pl

    from factorlib.config import MacroCommonConfig
    from factorlib.evaluation.pipeline import build_artifacts
    from factorlib.evaluation.profiles import MacroCommonProfile

    n_dates = 300
    factor_ts = np.cumsum(rng.standard_normal(n_dates))

    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"a{i}" for i in range(8)]
    rows = []
    for i, d in enumerate(dates):
        for a in assets:
            rows.append({
                "date": d,
                "asset_id": a,
                "factor": float(factor_ts[i]),
                "forward_return": float(
                    0.1 * factor_ts[i] + rng.standard_normal() * 0.5
                ),
            })
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    art = build_artifacts(df, MacroCommonConfig())
    art.factor_name = "persistent_macro"
    profile, _ = MacroCommonProfile.from_artifacts(art)

    assert profile.factor_adf_p > 0.10
    codes = {d.code for d in profile.diagnose()}
    assert "macro_common.factor_persistent" in codes


def test_macro_common_stationary_factor_does_not_flag(rng):
    from datetime import datetime, timedelta

    import polars as pl

    from factorlib.config import MacroCommonConfig
    from factorlib.evaluation.pipeline import build_artifacts
    from factorlib.evaluation.profiles import MacroCommonProfile

    n_dates = 300
    factor_ts = rng.standard_normal(n_dates)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"a{i}" for i in range(8)]
    rows = []
    for i, d in enumerate(dates):
        for a in assets:
            rows.append({
                "date": d,
                "asset_id": a,
                "factor": float(factor_ts[i]),
                "forward_return": float(rng.standard_normal()),
            })
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    art = build_artifacts(df, MacroCommonConfig())
    art.factor_name = "stationary_macro"
    profile, _ = MacroCommonProfile.from_artifacts(art)

    assert profile.factor_adf_p < 0.10
    codes = {d.code for d in profile.diagnose()}
    assert "macro_common.factor_persistent" not in codes
