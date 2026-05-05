"""Tests for ``factrix.metrics.ts_asymmetry`` (issue #5)."""

from __future__ import annotations

import numpy as np
import polars as pl
from factrix.metrics import ts_asymmetry


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


class TestSymmetricDgp:
    def test_linear_dgp_does_not_reject_symmetry(self):
        # r = 0.05*f → both sides have equal-magnitude conditional means.
        rng = np.random.default_rng(11)
        T = 600
        f = rng.standard_normal(T)
        r = 0.05 * f + rng.standard_normal(T) * 0.5
        out = ts_asymmetry(_series_panel(f, r), forward_periods=1)
        # H0: β_long + β_short = 0; under symmetric DGP we should not reject.
        assert out.metadata["p_value"] > 0.05
        # Method B should run (continuous f → both sides have variation).
        assert "p_wald_slopes" in out.metadata
        assert out.metadata["p_wald_slopes"] > 0.05


class TestAsymmetricDgp:
    def test_long_only_alpha_detected(self):
        # Edge only on the positive side: r = 0.20*max(f,0) + ε.
        # E[r|f<0] ≈ 0, E[r|f>0] > 0 → β_long + β_short > 0, reject H0.
        rng = np.random.default_rng(2)
        T = 800
        f = rng.standard_normal(T)
        r = 0.20 * np.maximum(f, 0) + rng.standard_normal(T) * 0.4
        out = ts_asymmetry(_series_panel(f, r), forward_periods=1)
        assert out.value > 0
        assert out.metadata["p_value"] < 0.05
        assert out.metadata["beta_long"] > out.metadata["beta_short"]
        # Slope test should also reject β_pos = β_neg.
        assert out.metadata["p_wald_slopes"] < 0.05
        assert out.metadata["beta_pos"] > out.metadata["beta_neg"]


class TestGateBNoTwoSides:
    def test_unsigned_factor_short_circuits(self):
        # All f >= 0 → no negative side; asymmetry undefined.
        rng = np.random.default_rng(0)
        T = 200
        f = rng.uniform(0, 1, size=T)
        r = rng.standard_normal(T)
        out = ts_asymmetry(_series_panel(f, r))
        assert out.metadata["reason"] == "no_two_sided_factor"
        assert out.metadata["n_neg"] == 0
        assert "event_quality" in out.metadata["hint"]

    def test_zero_only_factor_short_circuits(self):
        T = 200
        f = np.zeros(T)
        r = np.random.default_rng(0).standard_normal(T)
        out = ts_asymmetry(_series_panel(f, r))
        assert out.metadata["reason"] == "no_two_sided_factor"


class TestGateCMethodBSkip:
    def test_signed_binary_skips_method_b(self):
        # Signed binary {-1, +1} passes Gate B but fails Gate C
        # (each side has 1 unique value → cannot identify a slope).
        rng = np.random.default_rng(0)
        T = 200
        f = rng.choice([-1.0, 1.0], size=T)
        r = 0.10 * f + rng.standard_normal(T) * 0.5
        out = ts_asymmetry(_series_panel(f, r), forward_periods=1)
        assert "p_wald_slopes" not in out.metadata
        assert "method_b_skipped" in out.metadata
        # Method A still ran.
        assert "beta_long" in out.metadata
        assert "beta_short" in out.metadata

    def test_three_state_signed_skips_method_b(self):
        # {-1, 0, +1} sparse signal: each side has 1 unique value too.
        rng = np.random.default_rng(0)
        T = 300
        f = rng.choice([-1.0, 0.0, 1.0], size=T)
        r = 0.10 * f + rng.standard_normal(T) * 0.5
        out = ts_asymmetry(_series_panel(f, r), forward_periods=1)
        assert "method_b_skipped" in out.metadata
        # n_zero accounted for and zero column added to design.
        assert out.metadata["n_zero"] > 0
        assert "beta_zero" in out.metadata


class TestSampleFloor:
    def test_short_series_short_circuits(self):
        T = 4
        rng = np.random.default_rng(0)
        f = rng.standard_normal(T)
        r = rng.standard_normal(T)
        out = ts_asymmetry(_series_panel(f, r))
        assert out.metadata["reason"] == "insufficient_portfolio_periods"


class TestRatioDiagnostic:
    def test_short_dominant_ratio_above_one(self):
        rng = np.random.default_rng(4)
        T = 600
        f = rng.standard_normal(T)
        # Negative side carries 3x the magnitude.
        r = (
            0.05 * np.maximum(f, 0)
            + 0.15 * np.minimum(f, 0)
            + rng.standard_normal(T) * 0.3
        )
        out = ts_asymmetry(_series_panel(f, r), forward_periods=1)
        assert out.metadata["abs_short_over_long"] > 1.0


class TestMissingColumns:
    def test_missing_date_short_circuits(self):
        df = pl.DataFrame(
            {"asset_id": [0, 0], "factor": [1.0, -1.0], "forward_return": [0.1, 0.2]}
        )
        out = ts_asymmetry(df)
        assert out.metadata["reason"] == "no_date_column"
