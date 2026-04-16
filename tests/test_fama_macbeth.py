"""Tests for factorlib.metrics.fama_macbeth and macro_panel pipeline."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib._types import FactorType
from factorlib.config import MacroPanelConfig
from factorlib.metrics.fama_macbeth import (
    compute_fm_betas,
    fama_macbeth,
    pooled_ols,
    beta_sign_consistency,
)
from factorlib.metrics.quantile import quantile_spread, compute_spread_series
from factorlib.evaluation.pipeline import evaluate, build_artifacts
from factorlib.evaluation.profile import compute_profile


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
        assert result.value == 0.0


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
        assert result.name == "q1_q5_spread"
        assert result.value > 0


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestMacroPanelPipeline:
    def test_evaluate_returns_result(self, strong_macro):
        result = evaluate(
            strong_macro, "test_macro",
            config=MacroPanelConfig(),
        )
        assert result.factor_name == "test_macro"
        assert result.status in ("PASS", "CAUTION", "FAILED", "VETOED")
        assert result.artifacts is not None

    def test_profile_has_fm_metrics(self, strong_macro):
        result = evaluate(
            strong_macro, "test_macro",
            config=MacroPanelConfig(), gates=[],
        )
        assert result.profile is not None
        assert result.profile.get("fm_beta") is not None
        assert result.profile.get("pooled_beta") is not None
        assert result.profile.get("beta_sign_consistency") is not None
        assert result.profile.get("q1_q5_spread") is not None
        assert result.profile.get("oos_decay") is not None
        assert result.profile.get("beta_trend") is not None

    def test_no_ic_metrics_in_profile(self, strong_macro):
        result = evaluate(
            strong_macro, "test_macro",
            config=MacroPanelConfig(), gates=[],
        )
        assert result.profile.get("ic") is None
        assert result.profile.get("ic_ir") is None
        assert result.profile.get("hit_rate") is None

    def test_repr_works(self, strong_macro):
        result = evaluate(
            strong_macro, "test_macro",
            config=MacroPanelConfig(), gates=[],
        )
        text = repr(result)
        assert "Factor: test_macro" in text
        assert "fm_beta" in text

    def test_to_dataframe(self, strong_macro):
        result = evaluate(
            strong_macro, "test_macro",
            config=MacroPanelConfig(), gates=[],
        )
        df = result.to_dataframe()
        assert len(df) > 0
        assert "fm_beta" in df["metric"].to_list()

    def test_noise_fails_gate(self, noise_macro):
        result = evaluate(
            noise_macro, "noise_macro",
            config=MacroPanelConfig(),
        )
        assert result.status == "FAILED"

    def test_tiny_n_caution(self, tiny_macro):
        result = evaluate(
            tiny_macro, "tiny_macro",
            config=MacroPanelConfig(min_cross_section=10), gates=[],
        )
        assert any("cross-section" in r.lower() for r in result.caution_reasons)

    def test_artifacts_keys(self, strong_macro):
        artifacts = build_artifacts(strong_macro, MacroPanelConfig())
        assert "beta_series" in artifacts.intermediates
        assert "beta_values" in artifacts.intermediates
        assert "spread_series" in artifacts.intermediates
