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
from factrix._stats import _resolve_scalar_wald_hac
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
        res = pooled_beta(df, driscoll_kraay=True, overlap_periods=5)
        assert res.metadata["se_method"] == "driscoll_kraay"
        assert "Driscoll-Kraay" in res.metadata["method"]
        assert res.metadata["n_periods"] == 60
        assert isinstance(res.metadata["driscoll_kraay_lags"], int)
        lags, scale, dof = _resolve_scalar_wald_hac(60, None, 5)
        assert res.metadata["driscoll_kraay_lags"] == lags
        assert res.metadata["hac_scale"] == pytest.approx(scale)
        assert res.metadata["hac_dof"] == pytest.approx(dof)
        assert res.metadata["overlap_periods"] == 5
        assert res.metadata["overlap_periods_consumed"] is True

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
        assert np.isnan(res.value)
        assert res.stat is None
        assert res.metadata["reason"] == "insufficient_periods"
        assert res.metadata["n_periods"] == 2
        assert res.p_value == 1.0
        assert WarningCode.METRIC_UNAVAILABLE.value in res.warning_codes

    def test_singular_design_has_its_own_reason(self):
        df = _common_factor_panel(n_dates=12, n_assets=10, rho=0.2).with_columns(
            pl.lit(1.0).alias("factor")
        )

        res = pooled_beta(df, driscoll_kraay=True)

        assert np.isnan(res.value)
        assert res.stat is None
        assert res.p_value == 1.0
        assert res.metadata["reason"] == "singular_pooled_design_matrix"
        assert WarningCode.METRIC_UNAVAILABLE.value in res.warning_codes

    def test_mutually_exclusive_with_two_way_cluster(self):
        df = _common_factor_panel(n_dates=40, n_assets=10, rho=0.2)
        with pytest.raises(ValueError, match="mutually exclusive"):
            pooled_beta(df, driscoll_kraay=True, two_way_cluster_col="asset_id")

    def test_explicit_lags_recorded(self):
        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        res = pooled_beta(
            df,
            driscoll_kraay=True,
            driscoll_kraay_lags=1,
            overlap_periods=1,
        )
        assert res.metadata["driscoll_kraay_lags"] == 1

    def test_explicit_lags_cannot_undercut_overlap_floor(self):
        df = _common_factor_panel(n_dates=120, n_assets=12, rho=0.3)
        res = pooled_beta(
            df,
            driscoll_kraay=True,
            driscoll_kraay_lags=1,
            overlap_periods=21,
            expected_warnings=("hac_bandwidth_ill_conditioned",),
        )
        expected_lags, _, _ = _resolve_scalar_wald_hac(120, 1, 21)
        assert res.metadata["driscoll_kraay_lags"] == expected_lags

    def test_default_path_unchanged(self):
        # driscoll_kraay defaults to False → clustered SE, no DK metadata.
        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        res = pooled_beta(df)
        assert "se_method" not in res.metadata
        assert "clustered SE" in res.metadata["method"]
        assert res.metadata["overlap_periods"] is None
        assert res.metadata["overlap_periods_consumed"] is False

    def test_clustered_path_records_but_does_not_consume_overlap(self):
        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        res = pooled_beta(df, overlap_periods=21)

        assert res.metadata["overlap_periods"] == 21
        assert res.metadata["overlap_periods_consumed"] is False

    def test_driscoll_kraay_p_value_degrees_of_freedom(self):
        import scipy.stats as sp_stats

        df = _common_factor_panel(n_dates=60, n_assets=12, rho=0.3)
        res = pooled_beta(df, driscoll_kraay=True)
        # The scalar restriction uses the calibrated effective degrees of
        # freedom carried in metadata.
        n_periods = res.metadata["n_periods"]
        assert n_periods == 60
        dof = res.metadata["hac_dof"]
        t_stat = res.stat
        expected_p = float(2 * sp_stats.t.sf(abs(t_stat), dof))
        assert res.p_value == pytest.approx(expected_p)


class TestPooledBetaNullPairs:
    """Null factor/return rows must not poison the pooled OLS slope."""

    def test_null_factor_does_not_nan_the_slope(self):
        # Factor undefined for some names (common in real research). The slope
        # must be estimated on the complete (factor, return) pairs, and n_obs
        # must count those pairs — not the raw rows including nulls.
        rng = np.random.default_rng(5)
        rows = []
        for d in range(120):
            for a in range(20):
                f = rng.normal()
                r = 1.5 * f + 0.5 * rng.normal()
                if rng.random() < 0.05:  # factor undefined here
                    f = None
                if rng.random() < 0.02:  # return unobserved here
                    r = None
                rows.append(
                    {
                        "date": d,
                        "asset_id": a,
                        "factor": None if f is None else float(f),
                        "forward_return": None if r is None else float(r),
                    }
                )
        df = pl.DataFrame(rows)

        res = pooled_beta(df)
        complete = df.drop_nulls(["factor", "forward_return"])
        ref_slope = float(
            np.polyfit(
                complete["factor"].to_numpy(), complete["forward_return"].to_numpy(), 1
            )[0]
        )
        assert np.isfinite(res.value)
        assert res.value == pytest.approx(ref_slope, abs=1e-9)
        assert res.n_obs == complete.height
