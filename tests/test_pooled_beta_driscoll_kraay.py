"""Tests for ``pooled_beta(driscoll_kraay=True)``.

The DK SE path: metadata marks the SE method, the point estimate is
unchanged vs the clustered path, DK inflates SE relative to a one-way
date cluster when the score carries a persistent common factor (the
divergence case), short period series short-circuit / warn, and
``two_way_cluster_col`` is rejected.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics.fm_beta import pooled_beta


def _common_factor_panel(
    *, n_dates: int, n_assets: int, rho: float, seed: int = 7
) -> pl.DataFrame:
    """Panel where factor and return both load on a persistent common factor.

    The shared (optionally AR(1)) component ``g_t`` enters both the
    regressor and the return, so the per-observation score carries a
    persistent cross-sectional common term — exactly the structure a
    one-way date cluster misses and DK corrects.
    """
    rng = np.random.default_rng(seed)
    g = 0.0
    rows = []
    for d in range(n_dates):
        g = rho * g + rng.normal(0, 1)
        u = rng.normal(0, 0.5, n_assets)
        v = rng.normal(0, 0.5, n_assets)
        f = g + u
        r = g + v
        for a in range(n_assets):
            rows.append((d, a, float(f[a]), float(r[a])))
    return pl.DataFrame(
        rows,
        schema=["date", "asset_id", "factor", "forward_return"],
        orient="row",
    )


def _se(result) -> float:
    return abs(result.value / result.stat)


class TestDriscollKraayPath:
    def test_metadata_marks_se_method(self):
        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        res = pooled_beta(df, driscoll_kraay=True)
        assert res.metadata["se_method"] == "driscoll_kraay"
        assert "Driscoll-Kraay" in res.metadata["method"]
        assert res.metadata["n_periods"] == 60
        assert isinstance(res.metadata["driscoll_kraay_lags"], int)

    def test_point_estimate_matches_clustered_path(self):
        # SE method does not change the OLS slope — only its variance.
        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        cl = pooled_beta(df)
        dk = pooled_beta(df, driscoll_kraay=True)
        assert dk.value == pytest.approx(cl.value)

    def test_inflates_se_vs_date_cluster_under_common_factor(self):
        # Persistent common factor in the score → date-clustering treats
        # periods as independent and understates SE; DK is robust to the
        # serial dependence and reports a larger SE.
        df = _common_factor_panel(n_dates=90, n_assets=12, rho=0.85)
        cl = pooled_beta(df)
        dk = pooled_beta(df, driscoll_kraay=True)
        assert _se(dk) > _se(cl)

    def test_short_period_series_warns(self):
        df = _common_factor_panel(n_dates=12, n_assets=15, rho=0.2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = pooled_beta(df, driscoll_kraay=True)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in res.warning_codes
        assert any("Driscoll-Kraay" in str(w.message) for w in caught)

    def test_long_period_series_silent(self):
        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.2)
        res = pooled_beta(df, driscoll_kraay=True)
        assert res.warning_codes == ()

    def test_too_few_periods_short_circuits(self):
        # < 3 distinct periods → cross-sectional HAC undefined.
        df = _common_factor_panel(n_dates=2, n_assets=30, rho=0.0)
        res = pooled_beta(df, driscoll_kraay=True)
        assert res.stat is None
        assert res.metadata["reason"] == "insufficient_periods"
        assert res.p_value == 1.0

    def test_mutually_exclusive_with_two_way_cluster(self):
        df = _common_factor_panel(n_dates=40, n_assets=10, rho=0.2)
        with pytest.raises(ValueError, match="mutually exclusive"):
            pooled_beta(df, driscoll_kraay=True, two_way_cluster_col="asset_id")

    def test_explicit_lags_recorded(self):
        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        res = pooled_beta(df, driscoll_kraay=True, driscoll_kraay_lags=1)
        assert res.metadata["driscoll_kraay_lags"] == 1

    def test_default_path_unchanged(self):
        # driscoll_kraay defaults to False → clustered SE, no DK metadata.
        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        res = pooled_beta(df)
        assert "se_method" not in res.metadata
        assert "clustered SE" in res.metadata["method"]

    def test_driscoll_kraay_p_value_degrees_of_freedom(self):
        import scipy.stats as sp_stats

        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        res = pooled_beta(df, driscoll_kraay=True)
        # Manually compute the p-value with dof = n_periods - 1
        n_periods = res.metadata["n_periods"]
        assert n_periods == 60
        dof = n_periods - 1
        t_stat = res.stat
        expected_p = float(2 * sp_stats.t.sf(abs(t_stat), dof))
        assert res.p_value == pytest.approx(expected_p)
