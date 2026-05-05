"""Tests for ``factrix.metrics.ts_quantile_spread`` (issue #5)."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix.metrics import ts_quantile_spread


def _series_panel(
    factor: np.ndarray,
    forward_return: np.ndarray,
) -> pl.DataFrame:
    T = len(factor)
    return pl.DataFrame(
        {
            "date": list(range(T)),
            "asset_id": [0] * T,
            "factor": factor,
            "forward_return": forward_return,
        }
    )


class TestLinearDgp:
    def test_recovers_positive_spread(self):
        rng = np.random.default_rng(42)
        T = 600
        f = rng.standard_normal(T)
        r = 0.10 * f + rng.standard_normal(T) * 0.5
        out = ts_quantile_spread(_series_panel(f, r), n_groups=5)
        assert out.value > 0
        assert out.metadata["p_value"] < 0.05
        assert out.metadata["spearman_rho"] > 0.5

    def test_top_bucket_mean_exceeds_bottom(self):
        rng = np.random.default_rng(7)
        T = 600
        f = rng.standard_normal(T)
        r = 0.10 * f + rng.standard_normal(T) * 0.5
        out = ts_quantile_spread(_series_panel(f, r), n_groups=5)
        buckets = out.metadata["buckets"]
        assert buckets[-1]["mean_return"] > buckets[0]["mean_return"]
        # All five buckets accounted for, sums to T
        assert sum(b["n"] for b in buckets) == T


class TestNullDgp:
    def test_pure_noise_rejection_rate_at_size(self):
        # Single-seed p > 0.05 is a flaky test — under H0, by definition
        # 5% of draws reject. Check empirical rejection across many seeds
        # stays close to nominal 5% instead.
        T = 400
        rejects = 0
        n_trials = 100
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            f = rng.standard_normal(T)
            r = rng.standard_normal(T) * 0.5
            out = ts_quantile_spread(_series_panel(f, r), n_groups=5)
            if out.metadata["p_value"] < 0.05:
                rejects += 1
        rate = rejects / n_trials
        # Wide-but-meaningful band: 0–13% covers binomial noise at n=100
        # while still flagging gross bias (e.g. 30% rejection under null).
        assert 0.0 <= rate <= 0.13, f"empirical size {rate} out of band"


class TestNonLinearDgp:
    def test_u_shape_caught_by_buckets_not_by_sign(self):
        # E[r | f] = c·f² — symmetric U-shape. β_OLS ≈ 0 but the
        # extreme buckets carry the largest mean returns, the middle
        # the smallest. Spearman across bucket means is NOT monotone
        # (so |rho| should be small) but the per-bucket pattern is
        # clearly non-trivial — top and bottom buckets both above
        # middle buckets.
        rng = np.random.default_rng(3)
        T = 800
        f = rng.standard_normal(T)
        r = 0.30 * (f * f) + rng.standard_normal(T) * 0.4
        out = ts_quantile_spread(_series_panel(f, r), n_groups=5)
        means = [b["mean_return"] for b in out.metadata["buckets"]]
        # Extremes higher than middle bucket
        assert means[0] > means[2]
        assert means[-1] > means[2]


class TestGateAFactorVariation:
    def test_binary_factor_short_circuits(self):
        rng = np.random.default_rng(0)
        T = 200
        f = rng.choice([0.0, 1.0], size=T)
        r = rng.standard_normal(T)
        out = ts_quantile_spread(_series_panel(f, r), n_groups=5)
        assert out.metadata["reason"] == "insufficient_factor_variation"
        assert out.metadata["n_distinct"] == 2
        assert "event_quality" in out.metadata["hint"]

    def test_signed_binary_factor_short_circuits(self):
        rng = np.random.default_rng(0)
        T = 200
        f = rng.choice([-1.0, 0.0, 1.0], size=T)
        r = rng.standard_normal(T)
        out = ts_quantile_spread(_series_panel(f, r), n_groups=5)
        assert out.metadata["reason"] == "insufficient_factor_variation"

    def test_lower_n_groups_passes_when_distinct_allows(self):
        # 6 distinct values, n_groups=2 → 6 >= 4 passes Gate A.
        rng = np.random.default_rng(0)
        T = 200
        f = rng.integers(0, 6, size=T).astype(float)
        r = rng.standard_normal(T)
        out = ts_quantile_spread(_series_panel(f, r), n_groups=2)
        assert "reason" not in out.metadata or out.metadata.get("reason") is None


class TestSampleFloor:
    def test_short_series_short_circuits(self):
        # MIN_PORTFOLIO_PERIODS_HARD = 3; T=2 trips the floor.
        rng = np.random.default_rng(0)
        T = 2
        f = rng.standard_normal(T)
        r = rng.standard_normal(T)
        out = ts_quantile_spread(_series_panel(f, r), n_groups=5)
        assert out.metadata["reason"] == "insufficient_portfolio_periods"


class TestPerBucketWarning:
    def test_thin_buckets_emit_warning(self):
        # T=20, n_groups=5 → 4 per bucket → triggers <5 warning.
        rng = np.random.default_rng(0)
        T = 20
        f = rng.standard_normal(T)
        r = rng.standard_normal(T)
        with pytest.warns(UserWarning, match="periods per bucket"):
            ts_quantile_spread(_series_panel(f, r), n_groups=5)

    def test_fat_buckets_silent(self):
        rng = np.random.default_rng(0)
        T = 200
        f = rng.standard_normal(T)
        r = rng.standard_normal(T)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ts_quantile_spread(_series_panel(f, r), n_groups=5)


class TestMissingColumns:
    def test_missing_date_short_circuits(self):
        df = pl.DataFrame(
            {"asset_id": [0, 0], "factor": [1.0, 2.0], "forward_return": [0.1, 0.2]}
        )
        out = ts_quantile_spread(df)
        assert out.metadata["reason"] == "no_date_column"

    def test_missing_factor_short_circuits(self):
        df = pl.DataFrame(
            {"date": [0, 1], "asset_id": [0, 0], "forward_return": [0.1, 0.2]}
        )
        out = ts_quantile_spread(df)
        assert out.metadata["reason"] == "no_factor_column"
