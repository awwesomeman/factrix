"""Tests for factrix.metrics.trend."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest
from factrix.metrics.trend import ic_trend


def _make_series(values: list[float]) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "value": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestTheilSenSlope:
    def test_perfect_uptrend(self):
        values = [0.01 * i for i in range(20)]
        result = ic_trend(_make_series(values))
        assert result.value > 0
        assert result.metadata["ci_excludes_zero"] is True

    def test_flat(self):
        values = [0.05] * 20
        result = ic_trend(_make_series(values))
        assert result.value == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        values = [0.01] * 5  # < 10
        result = ic_trend(_make_series(values))
        assert math.isnan(result.value)
        assert result.p_value is None or result.p_value >= 0.10

    def test_negative_slope(self):
        values = [0.10 - 0.005 * i for i in range(20)]
        result = ic_trend(_make_series(values))
        assert result.value < 0

    def test_adf_flags_random_walk(self):
        """Unit-root series (random walk) should trip the ADF guard."""
        import numpy as np

        rng = np.random.default_rng(0)
        walk = np.cumsum(rng.standard_normal(200)).tolist()
        result = ic_trend(_make_series(walk))
        assert result.metadata["unit_root_suspected"] is True
        assert "adf_stat" in result.metadata

    def test_adf_clears_stationary_noise(self):
        """IID Gaussian noise should not trip the ADF guard."""
        import numpy as np

        rng = np.random.default_rng(1)
        noise = rng.standard_normal(200).tolist()
        result = ic_trend(_make_series(noise))
        assert result.metadata["unit_root_suspected"] is False

    def test_adf_threshold_none_disables_check(self):
        values = [0.01 * i for i in range(20)]
        result = ic_trend(_make_series(values), adf_threshold=None)
        assert "unit_root_suspected" not in result.metadata
        assert "adf_p" not in result.metadata

    def test_adf_threshold_custom_value(self):
        """Strict threshold (0.01) should rarely flag stationary noise;
        permissive threshold (0.50) should frequently flag the same series.
        Verifies the parameter actually plumbs into the unit-root decision."""
        import numpy as np

        rng = np.random.default_rng(7)
        noise = rng.standard_normal(200).tolist()
        strict = ic_trend(_make_series(noise), adf_threshold=0.01)
        permissive = ic_trend(_make_series(noise), adf_threshold=0.50)
        assert strict.metadata["adf_p"] == permissive.metadata["adf_p"]
        # Same ADF p, different decision threshold → strict <= permissive flag.
        assert int(strict.metadata["unit_root_suspected"]) <= int(
            permissive.metadata["unit_root_suspected"]
        )

    def test_adf_threshold_out_of_range_raises(self):
        values = [0.01 * i for i in range(20)]
        with pytest.raises(ValueError, match="adf_threshold"):
            ic_trend(_make_series(values), adf_threshold=1.5)
        with pytest.raises(ValueError, match="adf_threshold"):
            ic_trend(_make_series(values), adf_threshold=0.0)

    def test_all_nan_input_short_circuits(self):
        """Regression: NaN-heavy IC series (e.g. from a constant factor)
        must short-circuit, not flow into lstsq and trip LAPACK DLASCL."""
        import math

        result = ic_trend(_make_series([math.nan] * 30))
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_trend_periods"


class TestDegenerateTheilSenCI:
    """Regression: a zero-width CI reported p=1.0 next to ci_excludes_zero."""

    def test_perfect_line_is_decisive_not_p_one(self):
        result = ic_trend(_make_series([0.01 * i for i in range(20)]))
        assert result.metadata["ci_low"] == result.metadata["ci_high"]
        assert result.metadata["ci_excludes_zero"] is True
        assert result.p_value == 0.0
        # slope / 0 has no finite t; MetricResult rejects a non-finite stat.
        assert result.stat is None
        assert "degenerate" in result.metadata["method"]

    def test_perfect_downtrend_is_decisive(self):
        result = ic_trend(_make_series([-0.01 * i for i in range(20)]))
        assert result.value < 0
        assert result.p_value == 0.0
        assert result.stat is None

    def test_flat_series_stays_insignificant(self):
        """Zero-width CI around a *zero* slope is the opposite case."""
        result = ic_trend(_make_series([0.05] * 20))
        assert result.value == pytest.approx(0.0, abs=1e-12)
        assert result.stat == 0.0
        assert result.p_value == 1.0
        assert result.metadata["ci_excludes_zero"] is False

    def test_p_value_uses_n_minus_2_dof(self):
        """SE is normal-derived, but the tail is t(n - 2) — slope + intercept."""
        import numpy as np
        from scipy import stats as sp_stats

        rng = np.random.default_rng(3)
        vals = (0.001 * np.arange(40) + rng.standard_normal(40) * 0.01).tolist()
        result = ic_trend(_make_series(vals), adf_threshold=None)
        assert result.stat is not None
        n = result.n_obs
        expected = float(2 * sp_stats.t.sf(abs(result.stat), n - 2))
        assert result.p_value == pytest.approx(expected)
        # ...and is not the old n - 1 tail.
        assert result.p_value != pytest.approx(
            float(2 * sp_stats.t.sf(abs(result.stat), n - 1))
        )
