"""Tests for factorlib.metrics.ts_beta and macro_common pipeline."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.config import MacroCommonConfig
from factorlib.metrics.ts_beta import (
    compute_ts_betas,
    ts_beta,
    mean_r_squared,
    compute_rolling_mean_beta,
    ts_beta_sign_consistency,
)
from factorlib.evaluation.pipeline import evaluate, build_artifacts
from factorlib.evaluation.profile import compute_profile


def _make_macro_common(
    n_assets: int = 20,
    n_dates: int = 200,
    signal_strength: float = 0.3,
    seed: int = 42,
) -> pl.DataFrame:
    """Common factor shared across all assets, per-asset β varies."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2015, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    # WHY: each asset has its own true β to the common factor
    true_betas = rng.uniform(0.5, 1.5, n_assets)
    factor_ts = rng.standard_normal(n_dates)

    rows = []
    for t, d in enumerate(dates):
        for i, a in enumerate(assets):
            ret = true_betas[i] * signal_strength * factor_ts[t] + rng.standard_normal()
            rows.append({
                "date": d,
                "asset_id": a,
                "factor": float(factor_ts[t]),
                "forward_return": float(ret),
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def strong_common() -> pl.DataFrame:
    return _make_macro_common(signal_strength=0.8)


@pytest.fixture
def noise_common() -> pl.DataFrame:
    return _make_macro_common(signal_strength=0.0, seed=99)


class TestComputeTsBetas:
    def test_returns_per_asset_betas(self, strong_common):
        result = compute_ts_betas(strong_common)
        assert "asset_id" in result.columns
        assert "beta" in result.columns
        assert "r_squared" in result.columns
        assert len(result) == 20

    def test_strong_signal_positive_betas(self, strong_common):
        result = compute_ts_betas(strong_common)
        assert result["beta"].mean() > 0


class TestTsBeta:
    def test_strong_signal_significant(self, strong_common):
        betas = compute_ts_betas(strong_common)
        result = ts_beta(betas)
        assert result.name == "ts_beta"
        assert result.value > 0
        assert abs(result.stat) > 2.0

    def test_noise_not_significant(self, noise_common):
        betas = compute_ts_betas(noise_common)
        result = ts_beta(betas)
        assert abs(result.stat) < 2.0


class TestMeanRSquared:
    def test_strong_signal_higher_r2(self, strong_common):
        betas = compute_ts_betas(strong_common)
        result = mean_r_squared(betas)
        assert result.name == "mean_r_squared"
        assert result.value > 0.05


class TestTsBetaSignConsistency:
    def test_strong_signal_high_consistency(self, strong_common):
        betas = compute_ts_betas(strong_common)
        result = ts_beta_sign_consistency(betas)
        assert result.value > 0.8


class TestMacroCommonPipeline:
    def test_evaluate_returns_result(self, strong_common):
        result = evaluate(
            strong_common, "VIX",
            config=MacroCommonConfig(ts_window=60),
        )
        assert result.factor_name == "VIX"
        assert result.status in ("PASS", "CAUTION", "FAILED", "VETOED")
        assert result.artifacts is not None

    def test_profile_has_ts_metrics(self, strong_common):
        result = evaluate(
            strong_common, "VIX",
            config=MacroCommonConfig(ts_window=60), gates=[],
        )
        assert result.profile is not None
        assert result.profile.get("ts_beta") is not None
        assert result.profile.get("mean_r_squared") is not None
        assert result.profile.get("ts_beta_sign_consistency") is not None
        assert result.profile.get("oos_decay") is not None
        assert result.profile.get("beta_trend") is not None

    def test_no_ic_metrics(self, strong_common):
        result = evaluate(
            strong_common, "VIX",
            config=MacroCommonConfig(ts_window=60), gates=[],
        )
        assert result.profile.get("ic") is None
        assert result.profile.get("fm_beta") is None

    def test_repr_works(self, strong_common):
        result = evaluate(
            strong_common, "VIX",
            config=MacroCommonConfig(ts_window=60), gates=[],
        )
        text = repr(result)
        assert "Factor: VIX" in text

    def test_noise_fails_gate(self, noise_common):
        result = evaluate(
            noise_common, "noise",
            config=MacroCommonConfig(ts_window=60),
        )
        assert result.status == "FAILED"

    def test_artifacts_keys(self, strong_common):
        artifacts = build_artifacts(strong_common, MacroCommonConfig(ts_window=60))
        assert "beta_series" in artifacts.intermediates
        assert "beta_values" in artifacts.intermediates
