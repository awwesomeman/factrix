"""Tests for factrix.metrics.trend."""

import math
import polars as pl
import pytest
from datetime import datetime, timedelta

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
        assert result.significance == ""

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

    def test_adf_check_opt_out(self):
        values = [0.01 * i for i in range(20)]
        result = ic_trend(_make_series(values), adf_check=False)
        assert "unit_root_suspected" not in result.metadata
        assert "adf_p" not in result.metadata
