"""Integration tests for event_signal full pipeline."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import factorlib as fl
from factorlib._types import FactorType
from factorlib.config import EventConfig
from factorlib.evaluation.pipeline import build_artifacts
from factorlib.evaluation.profiles import EventProfile
from factorlib.metrics.clustering import clustering_diagnostic
from factorlib.metrics.corrado import corrado_rank_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event_signal(
    n_assets: int = 50,
    n_dates: int = 500,
    event_prob: float = 0.02,
    signal_strength: float = 0.02,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for a in assets:
        price = 100.0
        for d in dates:
            is_event = rng.random() < event_prob
            direction = rng.choice([-1.0, 1.0]) if is_event else 0.0
            daily_ret = rng.normal(0, 0.015)
            if is_event:
                daily_ret += signal_strength * direction
            price *= (1 + daily_ret)

            rows.append({
                "date": d,
                "asset_id": a,
                "factor": direction,
                "forward_return": daily_ret,
                "price": price,
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def strong_event() -> pl.DataFrame:
    return _make_event_signal(signal_strength=0.03)


@pytest.fixture
def noise_event() -> pl.DataFrame:
    return _make_event_signal(signal_strength=0.0, seed=99)


@pytest.fixture
def single_asset_event() -> pl.DataFrame:
    return _make_event_signal(
        n_assets=1, n_dates=1000, event_prob=0.05,
        signal_strength=0.03, seed=77,
    )


@pytest.fixture
def high_clustering() -> pl.DataFrame:
    """Most events on the same few dates."""
    rng = np.random.default_rng(123)
    n_assets = 30
    n_dates = 200
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    # Cluster events: 80% of assets fire on the same 5 dates
    cluster_dates = set(dates[50:55])

    rows = []
    for a in assets:
        price = 100.0
        for d in dates:
            if d in cluster_dates:
                is_event = rng.random() < 0.8
            else:
                is_event = rng.random() < 0.005
            direction = rng.choice([-1.0, 1.0]) if is_event else 0.0
            daily_ret = rng.normal(0, 0.015)
            if is_event:
                daily_ret += 0.02 * direction
            price *= (1 + daily_ret)

            rows.append({
                "date": d,
                "asset_id": a,
                "factor": direction,
                "forward_return": daily_ret,
                "price": price,
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestEventPipeline:
    def test_evaluate_returns_profile(self, strong_event):
        profile = fl.evaluate(
            strong_event, "test_event",
            config=EventConfig(),
        )
        assert isinstance(profile, EventProfile)
        assert profile.factor_name == "test_event"

    def test_profile_has_event_metrics(self, strong_event):
        profile = fl.evaluate(
            strong_event, "test_event",
            config=EventConfig(),
        )
        assert profile.caar_mean != 0.0
        assert profile.bmp_sar_mean != 0.0
        assert profile.event_hit_rate >= 0.0
        assert profile.oos_survival_ratio >= 0.0
        assert profile.profit_factor > 0.0

    def test_clustering_in_multi_asset_profile(self, strong_event):
        profile = fl.evaluate(
            strong_event, "test_event",
            config=EventConfig(),
        )
        assert profile.clustering_hhi is not None

    def test_noise_fails_verdict(self, noise_event):
        profile = fl.evaluate(
            noise_event, "noise_event",
            config=EventConfig(),
        )
        assert profile.verdict() == "FAILED"

    def test_artifacts_keys(self, strong_event):
        artifacts = build_artifacts(strong_event, EventConfig())
        assert "caar_series" in artifacts.intermediates
        assert "caar_values" in artifacts.intermediates


# ---------------------------------------------------------------------------
# Single asset (N=1)
# ---------------------------------------------------------------------------

class TestSingleAsset:
    def test_no_clustering_in_single_asset(self, single_asset_event):
        profile = fl.evaluate(
            single_asset_event, "single_asset",
            config=EventConfig(),
        )
        assert profile.clustering_hhi is None

    def test_core_metrics_still_present(self, single_asset_event):
        profile = fl.evaluate(
            single_asset_event, "single_asset",
            config=EventConfig(),
        )
        assert profile.caar_mean != 0.0
        assert profile.event_hit_rate >= 0.0
        assert profile.profit_factor > 0.0


# ---------------------------------------------------------------------------
# High clustering
# ---------------------------------------------------------------------------

class TestHighClustering:
    def test_clustering_diagnostic_high_hhi(self, high_clustering):
        result = clustering_diagnostic(high_clustering)
        assert result.value > 0.01
        assert result.metadata["hhi_normalized"] > 0

    def test_diagnose_flags_high_clustering(self, high_clustering):
        profile = fl.evaluate(
            high_clustering, "cluster_test",
            config=EventConfig(),
        )
        if profile.clustering_hhi_normalized and profile.clustering_hhi_normalized > 0.3:
            codes = {d.code for d in profile.diagnose()}
            assert any("clustering" in c for c in codes)


# ---------------------------------------------------------------------------
# Continuous signal (event_ic auto-appears)
# ---------------------------------------------------------------------------

class TestContinuousSignal:
    def test_event_ic_appears_for_continuous(self):
        """When factor has magnitude variance, event_ic is populated."""
        rng = np.random.default_rng(42)
        n_assets, n_dates = 30, 300
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        assets = [f"a_{i}" for i in range(n_assets)]
        rows = []
        for a in assets:
            price = 100.0
            for d in dates:
                is_event = rng.random() < 0.03
                if is_event:
                    mag = rng.uniform(0.5, 5.0)
                    direction = rng.choice([-1.0, 1.0])
                    factor_val = direction * mag
                    daily_ret = 0.005 * mag * direction + rng.normal(0, 0.015)
                else:
                    factor_val = 0.0
                    daily_ret = rng.normal(0, 0.015)
                price *= (1 + daily_ret)
                rows.append({"date": d, "asset_id": a, "factor": factor_val,
                             "forward_return": daily_ret, "price": price})
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        profile = fl.evaluate(df, "continuous", config=EventConfig())
        assert profile.event_ic is not None

    def test_event_ic_absent_for_discrete(self, strong_event):
        """When factor is {-1, 0, +1}, event_ic is skipped (None)."""
        profile = fl.evaluate(
            strong_event, "discrete",
            config=EventConfig(),
        )
        assert profile.event_ic is None


# ---------------------------------------------------------------------------
# Standalone metrics
# ---------------------------------------------------------------------------

class TestStandaloneMetrics:
    def test_corrado_rank_test(self, strong_event):
        result = corrado_rank_test(strong_event)
        assert result.name == "corrado_rank"
        assert result.metadata["n_events"] > 0

    def test_corrado_importable(self):
        from factorlib.metrics import corrado_rank_test as crt
        assert callable(crt)

    def test_compute_mfe_mae_importable(self):
        from factorlib.metrics import compute_mfe_mae as cmm
        assert callable(cmm)


# ---------------------------------------------------------------------------
# describe_profile reflects the typed dataclass fields
# ---------------------------------------------------------------------------

class TestDescribeProfile:
    def test_describe_event_signal(self, capsys):
        fl.describe_profile("event_signal")
        out = capsys.readouterr().out
        assert "EventProfile" in out
        assert "caar_p" in out
        assert "CANONICAL_P_FIELD" in out
