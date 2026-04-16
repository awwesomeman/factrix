"""Integration tests for event_signal full pipeline."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib._types import FactorType
from factorlib.config import EventConfig
from factorlib.evaluation.pipeline import evaluate, build_artifacts
from factorlib.evaluation.profile import compute_profile
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
    def test_evaluate_returns_result(self, strong_event):
        result = evaluate(
            strong_event, "test_event",
            config=EventConfig(),
        )
        assert result.factor_name == "test_event"
        assert result.status in ("PASS", "CAUTION", "FAILED", "VETOED")
        assert result.artifacts is not None

    def test_profile_has_event_metrics(self, strong_event):
        result = evaluate(
            strong_event, "test_event",
            config=EventConfig(), gates=[],
        )
        assert result.profile is not None
        assert result.profile.get("caar") is not None
        assert result.profile.get("bmp_sar") is not None
        assert result.profile.get("event_hit_rate") is not None
        assert result.profile.get("oos_decay") is not None
        assert result.profile.get("caar_trend") is not None
        assert result.profile.get("profit_factor") is not None
        assert result.profile.get("event_skewness") is not None

    def test_no_cs_metrics_in_profile(self, strong_event):
        result = evaluate(
            strong_event, "test_event",
            config=EventConfig(), gates=[],
        )
        assert result.profile.get("ic") is None
        assert result.profile.get("ic_ir") is None
        assert result.profile.get("quantile_spread") is None
        assert result.profile.get("monotonicity") is None

    def test_clustering_in_multi_asset_profile(self, strong_event):
        result = evaluate(
            strong_event, "test_event",
            config=EventConfig(), gates=[],
        )
        assert result.profile.get("clustering_hhi") is not None

    def test_repr_works(self, strong_event):
        result = evaluate(
            strong_event, "test_event",
            config=EventConfig(), gates=[],
        )
        text = repr(result)
        assert "Factor: test_event" in text
        assert "caar" in text

    def test_to_dataframe(self, strong_event):
        result = evaluate(
            strong_event, "test_event",
            config=EventConfig(), gates=[],
        )
        df = result.to_dataframe()
        assert len(df) > 0
        assert "caar" in df["metric"].to_list()

    def test_noise_fails_gate(self, noise_event):
        result = evaluate(
            noise_event, "noise_event",
            config=EventConfig(),
        )
        assert result.status == "FAILED"

    def test_artifacts_keys(self, strong_event):
        artifacts = build_artifacts(strong_event, EventConfig())
        assert "caar_series" in artifacts.intermediates
        assert "caar_values" in artifacts.intermediates


# ---------------------------------------------------------------------------
# Single asset (N=1)
# ---------------------------------------------------------------------------

class TestSingleAsset:
    def test_no_clustering_in_single_asset(self, single_asset_event):
        result = evaluate(
            single_asset_event, "single_asset",
            config=EventConfig(), gates=[],
        )
        assert result.profile.get("clustering_hhi") is None

    def test_core_metrics_still_present(self, single_asset_event):
        result = evaluate(
            single_asset_event, "single_asset",
            config=EventConfig(), gates=[],
        )
        assert result.profile.get("caar") is not None
        assert result.profile.get("event_hit_rate") is not None
        assert result.profile.get("profit_factor") is not None


# ---------------------------------------------------------------------------
# High clustering
# ---------------------------------------------------------------------------

class TestHighClustering:
    def test_clustering_diagnostic_high_hhi(self, high_clustering):
        result = clustering_diagnostic(high_clustering)
        assert result.value > 0.01
        assert result.metadata["hhi_normalized"] > 0

    def test_caution_on_high_clustering(self, high_clustering):
        result = evaluate(
            high_clustering, "cluster_test",
            config=EventConfig(), gates=[],
        )
        # Should trigger clustering caution if HHI_normalized > 0.3
        clust = result.profile.get("clustering_hhi")
        if clust is not None:
            hhi_norm = clust.metadata.get("hhi_normalized", 0)
            if hhi_norm > 0.3:
                assert any("clustering" in r.lower() for r in result.caution_reasons)


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
# quick_check integration
# ---------------------------------------------------------------------------

class TestQuickCheck:
    def test_quick_check_event_signal(self):
        """Test fl.quick_check with factor_type='event_signal'."""
        import factorlib as fl

        rng = np.random.default_rng(42)
        n = 500
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        rows = []
        price = 100.0
        for d in dates:
            is_event = rng.random() < 0.05
            direction = rng.choice([-1.0, 1.0]) if is_event else 0.0
            daily_ret = rng.normal(0, 0.015)
            if is_event:
                daily_ret += 0.03 * direction
            price *= (1 + daily_ret)
            rows.append({
                "date": d,
                "asset_id": "BTC",
                "factor": direction,
                "price": price,
            })

        df = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Datetime("ms")),
        )
        result = fl.quick_check(df, "BTC_Signal", factor_type="event_signal")
        assert result.status in ("PASS", "CAUTION", "FAILED", "VETOED")


# ---------------------------------------------------------------------------
# describe_profile
# ---------------------------------------------------------------------------

class TestDescribeProfile:
    def test_describe_event_signal(self, capsys):
        import factorlib as fl
        fl.describe_profile("event_signal")
        out = capsys.readouterr().out
        assert "caar" in out
        assert "bmp_sar" in out
        assert "event_hit_rate" in out
