"""Tests for factrix.metrics.fama_macbeth and macro_panel pipeline."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factrix._types import FactorType
from factrix.config import MacroPanelConfig
from factrix.metrics.fama_macbeth import (
    compute_fm_betas,
    fama_macbeth,
    pooled_ols,
    beta_sign_consistency,
)
from factrix.metrics.quantile import quantile_spread, compute_spread_series
import factrix as fl
from factrix.evaluation.pipeline import build_artifacts
from factrix.evaluation.profiles import MacroPanelProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_macro_panel(
    n_countries: int = 15,
    n_months: int = 120,
    signal_strength: float = 0.3,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2015, 1, 1) + timedelta(days=30 * i) for i in range(n_months)]
    assets = [f"country_{i}" for i in range(n_countries)]

    rows = []
    for d in dates:
        signal = rng.standard_normal(n_countries)
        noise = rng.standard_normal(n_countries)
        ret = signal_strength * signal + (1 - signal_strength) * noise
        for i, a in enumerate(assets):
            rows.append({
                "date": d,
                "asset_id": a,
                "factor": float(signal[i]),
                "forward_return": float(ret[i]),
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def strong_macro() -> pl.DataFrame:
    return _make_macro_panel(signal_strength=0.5)


@pytest.fixture
def noise_macro() -> pl.DataFrame:
    return _make_macro_panel(signal_strength=0.0, seed=99)


@pytest.fixture
def tiny_macro() -> pl.DataFrame:
    return _make_macro_panel(n_countries=5, signal_strength=0.5)


# ---------------------------------------------------------------------------
# compute_fm_betas
# ---------------------------------------------------------------------------

class TestComputeFmBetas:
    def test_returns_date_beta_columns(self, strong_macro):
        result = compute_fm_betas(strong_macro)
        assert "date" in result.columns
        assert "beta" in result.columns
        assert len(result) > 0

    def test_strong_signal_positive_betas(self, strong_macro):
        result = compute_fm_betas(strong_macro)
        mean_beta = result["beta"].mean()
        assert mean_beta > 0


# ---------------------------------------------------------------------------
# fama_macbeth
# ---------------------------------------------------------------------------

class TestFamaMacbeth:
    def test_strong_signal_significant(self, strong_macro):
        betas = compute_fm_betas(strong_macro)
        result = fama_macbeth(betas)
        assert result.name == "fm_beta"
        assert result.value > 0
        assert abs(result.stat) > 2.0
        assert result.significance in ("***", "**")

    def test_noise_not_significant(self, noise_macro):
        betas = compute_fm_betas(noise_macro)
        result = fama_macbeth(betas)
        assert abs(result.stat) < 2.0

    def test_insufficient_periods(self):
        df = pl.DataFrame({
            "date": pl.Series([], dtype=pl.Datetime("ms")),
            "beta": pl.Series([], dtype=pl.Float64),
        })
        result = fama_macbeth(df)
        assert math.isnan(result.value)

    def test_shanken_correction_reduces_t(self, strong_macro):
        """EIV flag inflates SE by √c exactly; t scaled by 1/√c."""
        betas = compute_fm_betas(strong_macro)
        raw = fama_macbeth(betas, is_estimated_factor=False)
        corrected = fama_macbeth(betas, is_estimated_factor=True)
        c = corrected.metadata["shanken_c"]
        assert c > 1.0
        assert corrected.stat == pytest.approx(raw.stat / math.sqrt(c), rel=1e-10)
        assert corrected.metadata["stat_uncorrected"] == pytest.approx(raw.stat)
        assert corrected.metadata["method"].endswith("Shanken (1992) EIV")
        assert corrected.metadata["shanken_factor_return_var_source"] == (
            "betas_timeseries_proxy"
        )

    def test_shanken_with_user_supplied_factor_var(self, strong_macro):
        """User-supplied σ²_f overrides the proxy; source field reflects it."""
        betas = compute_fm_betas(strong_macro)
        corrected = fama_macbeth(
            betas, is_estimated_factor=True, factor_return_var=0.01,
        )
        assert corrected.metadata["shanken_factor_return_var"] == 0.01
        assert corrected.metadata["shanken_factor_return_var_source"] == (
            "user_supplied"
        )

    def test_shanken_skipped_on_flat_factor(self):
        """σ²_f ≈ 0 → correction skipped; uncorrected values kept."""
        from datetime import datetime, timedelta
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)],
            "beta": [0.5] * 20,
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = fama_macbeth(df, is_estimated_factor=True)
        assert result.metadata["shanken_correction"] == (
            "skipped_zero_factor_variance"
        )
        assert "shanken_c" not in result.metadata


# ---------------------------------------------------------------------------
# pooled_ols
# ---------------------------------------------------------------------------

class TestPooledOls:
    def test_strong_signal_significant(self, strong_macro):
        result = pooled_ols(strong_macro)
        assert result.name == "pooled_beta"
        assert result.value > 0
        assert abs(result.stat) > 2.0

    def test_noise_not_significant(self, noise_macro):
        result = pooled_ols(noise_macro)
        assert abs(result.stat) < 2.0

    def test_two_way_metadata_shape(self, strong_macro):
        """Opt-in two-way clustering reports both cluster counts + intersection."""
        result = pooled_ols(
            strong_macro, two_way_cluster_col="asset_id",
        )
        md = result.metadata
        assert md["method"].startswith(
            "Pooled OLS + two-way clustered SE",
        )
        assert "n_clusters_a" in md
        assert "n_clusters_b" in md
        assert "n_clusters_intersection" in md
        # Intersection cells ≤ product of the two margins (balanced panel
        # gives equality, unbalanced gives strict <).
        assert md["n_clusters_intersection"] <= (
            md["n_clusters_a"] * md["n_clusters_b"]
        )

    def test_two_way_and_one_way_differ(self, strong_macro):
        """Two-way SE ≠ single-way SE on data with asset-level persistence."""
        single = pooled_ols(strong_macro)
        two_way = pooled_ols(
            strong_macro, two_way_cluster_col="asset_id",
        )
        # Both share the same point estimate (same OLS).
        assert single.value == pytest.approx(two_way.value)
        # But t-stats differ because SE differs (usually two-way SE > single).
        assert abs(single.stat - two_way.stat) > 1e-6

    def test_two_way_non_psd_falls_back_to_oneway(self, strong_macro, monkeypatch):
        """Non-PSD V → fall back to single-way V_a; metadata flags the swap.

        Trigger by patching ``_cluster_meat`` to return a meat-i dominating
        a+b, which guarantees ``meat_a + meat_b − meat_i`` has a negative
        [1,1] along the slope coordinate. The single-way test above
        establishes the happy-path baseline; this one covers the
        Cameron-Miller (2015) fallback branch.
        """
        import sys
        fm_mod = sys.modules["factrix.metrics.fama_macbeth"]
        real_meat = fm_mod._cluster_meat
        call_count = {"n": 0}

        def patched_meat(X, resid, clusters):
            call_count["n"] += 1
            meat, g = real_meat(X, resid, clusters)
            # Third call is meat_i (after meat_a, meat_b). Inflate it so
            # c_a·meat_a + c_b·meat_b − c_i·meat_i has negative [1,1].
            if call_count["n"] == 3:
                return meat * 10.0, g
            return meat, g

        monkeypatch.setattr(fm_mod, "_cluster_meat", patched_meat)
        result = pooled_ols(strong_macro, two_way_cluster_col="asset_id")
        md = result.metadata
        assert md.get("variance_non_psd_fallback") == "one_way_date"
        # After fallback, SE equals single-way SE on the a-dimension.
        single = pooled_ols(strong_macro)
        assert abs(result.stat) == pytest.approx(abs(single.stat), rel=1e-10)


# ---------------------------------------------------------------------------
# beta_sign_consistency
# ---------------------------------------------------------------------------

class TestBetaSignConsistency:
    def test_strong_signal_high_consistency(self, strong_macro):
        betas = compute_fm_betas(strong_macro)
        result = beta_sign_consistency(betas)
        assert result.name == "beta_sign_consistency"
        assert result.value > 0.6


# ---------------------------------------------------------------------------
# quantile_spread with small N (macro_panel uses n_groups=3)
# ---------------------------------------------------------------------------

class TestQuantileSpreadSmallN:
    def test_strong_signal_positive_spread(self, strong_macro):
        result = quantile_spread(strong_macro, forward_periods=1, n_groups=3)
        assert result.name == "quantile_spread"
        assert result.value > 0


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestMacroPanelPipeline:
    def test_evaluate_returns_profile(self, strong_macro):
        profile = fl.evaluate(
            strong_macro, "test_macro",
            config=MacroPanelConfig(),
        )
        assert isinstance(profile, MacroPanelProfile)
        assert profile.factor_name == "test_macro"

    def test_strong_signal_passes(self, strong_macro):
        profile = fl.evaluate(
            strong_macro, "test_macro",
            config=MacroPanelConfig(),
        )
        assert profile.verdict() == "PASS"
        assert profile.fm_beta_p < 0.05

    def test_noise_fails(self, noise_macro):
        profile = fl.evaluate(
            noise_macro, "noise_macro",
            config=MacroPanelConfig(),
        )
        assert profile.verdict() == "FAILED"

    def test_tiny_n_diagnoses(self, tiny_macro):
        profile = fl.evaluate(
            tiny_macro, "tiny_macro",
            config=MacroPanelConfig(min_cross_section=10),
        )
        codes = {d.code for d in profile.diagnose()}
        assert any("cross_section" in c or "cross-section" in c for c in codes)

    def test_artifacts_keys(self, strong_macro):
        artifacts = build_artifacts(strong_macro, MacroPanelConfig())
        assert "beta_series" in artifacts.intermediates
        assert "beta_values" in artifacts.intermediates
        assert "spread_series" in artifacts.intermediates
