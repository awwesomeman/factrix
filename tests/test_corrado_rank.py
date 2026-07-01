"""Tests for factrix.metrics.corrado_rank."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from factrix.metrics.corrado_rank import corrado_rank


def _panel(returns: np.ndarray, factor: np.ndarray) -> pl.DataFrame:
    n = len(returns)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "asset_id": ["A"] * n,
            "factor": factor,
            "forward_return": returns,
        }
    )


def _directional_panel(sign: float, n: int = 300, seed: int = 0) -> pl.DataFrame:
    """Baseline noise returns with a handful of events whose own return is
    shifted well into the tail in the direction ``sign * factor``, so the
    event ranks are genuinely extreme rather than incidentally so.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=n)
    factor = np.zeros(n)
    event_idx = np.arange(10, n, 30)
    factor[event_idx] = sign
    returns[event_idx] = sign * 5.0  # push into the extreme tail
    return _panel(returns, factor)


class TestOneSidedPValue:
    def test_anti_predictive_factor_is_not_significant(self):
        """Events sit at the top of the return distribution (rank ≈ +0.5)
        while ``factor`` points the wrong way (-1), so the direction-signed
        rank is negative: z < 0, and a one-sided p should be large.
        """
        rng = np.random.default_rng(0)
        n = 300
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(10, n, 30)
        returns[event_idx] = 5.0  # events are the largest returns...
        factor[event_idx] = -1.0  # ...but the factor calls them down

        result = corrado_rank(_panel(returns, factor))

        assert result.stat < 0
        assert result.p_value > 0.5

    def test_predictive_factor_is_significant(self):
        """Mirror case: factor direction matches the extreme rank at each
        event, so z > 0 and the one-sided p should be small.
        """
        result = corrado_rank(_directional_panel(sign=1.0))

        assert result.stat > 0
        assert result.p_value < 0.05

    def test_p_value_is_one_sided_sf(self):
        """p should equal the upper-tail normal survival function of z,
        not the two-sided 2*sf(|z|)."""
        from scipy import stats as sp_stats

        rng = np.random.default_rng(1)
        n = 150
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(5, n, 15)
        factor[event_idx] = rng.choice([-1.0, 1.0], size=len(event_idx))

        result = corrado_rank(_panel(returns, factor))

        assert result.p_value == sp_stats.norm.sf(result.stat)
