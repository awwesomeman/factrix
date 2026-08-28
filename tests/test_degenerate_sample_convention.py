"""Degenerate samples report NaN and a warning code, never the null.

The repo convention (``_stats/core.py``, ``_stats/hac.py``,
``_degenerate_test_fields``) is that a zero-SE sample is degenerate in the
*maximum*-evidence direction, so ``p = 1.0`` inverts the reading. Three call
sites still reported ``t = 0, p = 1.0`` or a best-case score of ``0.0``.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics.fm_beta import pooled_beta
from factrix.metrics.predictive_beta import predictive_beta


def _perfect_fit_panel(n_dates: int = 60, n_assets: int = 8) -> pl.DataFrame:
    """Panel whose forward return is EXACTLY ``0.01 * factor`` — zero residual."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_dates):
        date = datetime(2024, 1, 1) + timedelta(days=d)
        for a in range(n_assets):
            f = float(rng.standard_normal())
            rows.append(
                {
                    "date": date,
                    "asset_id": f"A{a}",
                    "factor": f,
                    "forward_return": 0.01 * f,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestPooledBetaPerfectFit:
    """`pooled_beta` was the only metric in the library skipping the
    degenerate-sample convention: value=0.01, stat=0.0, p=1.0, no code."""

    def test_clustered_path_withholds_the_test(self):
        result = pooled_beta(_perfect_fit_panel())
        assert result.value == pytest.approx(0.01)
        assert result.stat is None
        assert result.p_value is None
        assert WarningCode.DEGENERATE_VARIANCE.value in result.warning_codes
        assert result.metadata["signal_status"] == "degenerate_zero_variance"

    def test_driscoll_kraay_path_withholds_the_test(self):
        result = pooled_beta(_perfect_fit_panel(), driscoll_kraay=True)
        assert result.value == pytest.approx(0.01)
        assert result.stat is None
        assert result.p_value is None
        assert WarningCode.DEGENERATE_VARIANCE.value in result.warning_codes

    def test_ordinary_panel_still_reports_a_test(self):
        rng = np.random.default_rng(1)
        rows = []
        for d in range(60):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for a in range(8):
                f = float(rng.standard_normal())
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{a}",
                        "factor": f,
                        "forward_return": 0.01 * f + 0.05 * rng.standard_normal(),
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = pooled_beta(panel)
        assert result.stat is not None
        assert result.p_value is not None
        assert WarningCode.DEGENERATE_VARIANCE.value not in result.warning_codes


class TestPredictiveBetaPerfectFit:
    """`_ols_nw_slope_t` returned (beta, 0.0, 1.0) on a zero-residual fit and
    `predictive_beta` passed it to MetricResult with no flag."""

    @staticmethod
    def _series(n: int = 60) -> pl.DataFrame:
        rng = np.random.default_rng(3)
        x = rng.standard_normal(n)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        return pl.DataFrame(
            {
                "date": dates,
                "asset_id": ["A"] * n,
                "factor": x,
                "forward_return": 2.0 * x + 1.0,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_exact_linear_relation_withholds_the_test(self):
        result = predictive_beta(self._series(), adf_threshold=None)
        assert result.value == pytest.approx(2.0)
        assert result.stat is None
        assert result.p_value is None
        assert WarningCode.DEGENERATE_VARIANCE.value in result.warning_codes

    def test_noisy_relation_still_reports_a_test(self):
        rng = np.random.default_rng(4)
        n = 60
        x = rng.standard_normal(n)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pl.DataFrame(
            {
                "date": dates,
                "asset_id": ["A"] * n,
                "factor": x,
                "forward_return": 2.0 * x + rng.standard_normal(n),
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = predictive_beta(df, adf_threshold=None)
        assert result.stat is not None
        assert result.p_value is not None


class TestOlsSlopeDegenerateBranches:
    def test_constant_regressor_is_not_computable(self):
        from factrix._stats.ols import _ols_nw_slope_t

        y = np.arange(40, dtype=float)
        x = np.full(40, 2.0)
        beta, t, p, _ = _ols_nw_slope_t(y, x, lags=1)
        assert math.isnan(beta) and math.isnan(t) and math.isnan(p)

    def test_perfect_fit_keeps_beta_but_withholds_the_test(self):
        from factrix._stats.ols import _ols_nw_slope_t

        x = np.linspace(-1.0, 1.0, 40)
        beta, t, p, _ = _ols_nw_slope_t(2.0 * x + 1.0, x, lags=1)
        assert beta == pytest.approx(2.0)
        assert math.isnan(t) and math.isnan(p)


class TestHansenHodrickReportsEstimateAndNObs:
    """The only member of the family that dropped them."""

    def test_estimate_and_n_obs_are_populated(self):
        from factrix.inference.series_mean import (
            HANSEN_HODRICK,
            NEWEY_WEST,
            NON_OVERLAPPING,
            STATIONARY_BOOTSTRAP,
        )

        series = np.random.default_rng(0).standard_normal(80) + 0.1
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(80)]
        df = pl.DataFrame({"date": dates, "ic": series}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        for member in (
            NON_OVERLAPPING,
            NEWEY_WEST,
            HANSEN_HODRICK,
            STATIONARY_BOOTSTRAP,
        ):
            result = member.compute(df, value_col="ic", overlap_periods=5)
            assert result.n_obs is not None, member.summary
            assert result.estimate is not None, member.summary
        assert HANSEN_HODRICK.compute(
            df, value_col="ic", overlap_periods=5
        ).estimate == pytest.approx(float(series.mean()))
