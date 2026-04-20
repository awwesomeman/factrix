"""Tests for factorlib.datasets — synthetic panels with calibrated IC."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import factorlib as fl
from factorlib import datasets


class TestMakeCsPanelSchema:
    def test_canonical_columns_and_dtypes(self):
        df = datasets.make_cs_panel(n_assets=10, n_dates=60, seed=0)
        assert df.columns == ["date", "asset_id", "price", "factor"]
        assert df.schema["date"] == pl.Datetime("ms")
        assert df.schema["asset_id"] == pl.String
        assert df.schema["price"] == pl.Float64
        assert df.schema["factor"] == pl.Float64

    def test_row_count(self):
        df = datasets.make_cs_panel(n_assets=12, n_dates=40, seed=0)
        assert df.height == 12 * 40
        assert df["asset_id"].n_unique() == 12
        assert df["date"].n_unique() == 40

    def test_no_nan_or_inf(self):
        df = datasets.make_cs_panel(n_assets=10, n_dates=60, seed=0)
        for col in ("price", "factor"):
            assert df[col].is_nan().sum() == 0
            assert df[col].is_finite().all()

    def test_seed_is_deterministic(self):
        a = datasets.make_cs_panel(n_assets=8, n_dates=30, seed=123)
        b = datasets.make_cs_panel(n_assets=8, n_dates=30, seed=123)
        assert a.equals(b)

    def test_different_seeds_differ(self):
        a = datasets.make_cs_panel(n_assets=8, n_dates=30, seed=1)
        b = datasets.make_cs_panel(n_assets=8, n_dates=30, seed=2)
        assert not a["factor"].equals(b["factor"])

    def test_raises_on_short_panel(self):
        with pytest.raises(ValueError, match="n_dates must be >="):
            datasets.make_cs_panel(n_assets=5, n_dates=5, forward_periods=5)

    def test_raises_on_singleton_cross_section(self):
        with pytest.raises(ValueError, match="n_assets"):
            datasets.make_cs_panel(n_assets=1, n_dates=60)


class TestMakeCsPanelEndToEnd:
    def test_runs_through_preprocess_and_evaluate(self):
        cfg = fl.CrossSectionalConfig(forward_periods=5, n_groups=5)
        raw = datasets.make_cs_panel(
            n_assets=20, n_dates=80, ic_target=0.05, forward_periods=5, seed=7,
        )
        prepared = fl.preprocess(raw, config=cfg)
        profile = fl.evaluate(prepared, "synthetic", config=cfg)
        assert profile.factor_name == "synthetic"
        assert profile.ic_mean is not None

    def test_realized_ic_mean_near_target(self):
        """Realized ic_mean lands within ~3σ of target.

        Overlapping N-day forward returns give ≈ n_dates / forward_periods
        independent dates, so per-date IC s.e. ≈ 1/√n_assets and mean-IC
        s.e. ≈ 1/√((n_dates/forward_periods)·n_assets). At (100, 500, N=5):
        s.e. ≈ 1/√(100·100) = 0.01, so 3σ ≈ 0.03.
        """
        cfg = fl.CrossSectionalConfig(forward_periods=5)
        ic_target = 0.04
        raw = datasets.make_cs_panel(
            n_assets=100, n_dates=500,
            ic_target=ic_target, forward_periods=5, seed=2024,
        )
        prepared = fl.preprocess(raw, config=cfg)
        profile = fl.evaluate(prepared, "synthetic", config=cfg)
        assert abs(profile.ic_mean - ic_target) < 0.03, (
            f"realized ic_mean={profile.ic_mean:.4f} too far from "
            f"target={ic_target}"
        )

    def test_negative_ic_target_gives_negative_ic_mean(self):
        cfg = fl.CrossSectionalConfig(forward_periods=5)
        raw = datasets.make_cs_panel(
            n_assets=100, n_dates=500,
            ic_target=-0.05, forward_periods=5, seed=2025,
        )
        prepared = fl.preprocess(raw, config=cfg)
        profile = fl.evaluate(prepared, "synthetic_neg", config=cfg)
        assert profile.ic_mean < -0.02, (
            f"expected clearly negative ic_mean, got {profile.ic_mean:.4f}"
        )

    def test_zero_ic_target_gives_near_zero_ic_mean(self):
        cfg = fl.CrossSectionalConfig(forward_periods=5)
        raw = datasets.make_cs_panel(
            n_assets=80, n_dates=300, ic_target=0.0, forward_periods=5, seed=11,
        )
        prepared = fl.preprocess(raw, config=cfg)
        profile = fl.evaluate(prepared, "null", config=cfg)
        # At (80, 300, N=5): s.e. ≈ 1/√(60·80) ≈ 0.014, 3σ ≈ 0.045
        assert abs(profile.ic_mean) < 0.045


class TestMakeEventPanel:
    def test_canonical_columns_and_dtypes(self):
        df = datasets.make_event_panel(n_assets=10, n_dates=60, seed=0)
        assert df.columns == ["date", "asset_id", "price", "factor"]
        assert df.schema["date"] == pl.Datetime("ms")
        assert df.schema["factor"] == pl.Float64

    def test_factor_values_are_ternary(self):
        df = datasets.make_event_panel(n_assets=20, n_dates=120, seed=3)
        unique = set(df["factor"].unique().to_list())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_event_rate_is_approximately_honored(self):
        df = datasets.make_event_panel(
            n_assets=50, n_dates=200, event_rate=0.05, seed=17,
        )
        realized = (df["factor"] != 0.0).sum() / df.height
        assert 0.03 < realized < 0.07

    def test_runs_through_preprocess_and_evaluate(self):
        cfg = fl.EventConfig(forward_periods=5)
        raw = datasets.make_event_panel(
            n_assets=30, n_dates=120, event_rate=0.05,
            post_event_drift_bps=30.0, forward_periods=5, seed=5,
        )
        prepared = fl.preprocess(raw, config=cfg)
        profile = fl.evaluate(prepared, "events", config=cfg)
        assert profile.factor_name == "events"

    def test_post_event_drift_has_correct_sign(self):
        """Drift must land inside the forward-return measurement window.

        After preprocess, ``forward_return`` at event date t reflects
        (p[t+1+N]/p[t+1])/N − 1/N, which spans bars t+2..t+1+N. If the
        dataset injects drift on t+1..t+N instead (off-by-one), realized
        mean forward_return conditional on factor==+1 can be near zero.
        """
        cfg = fl.EventConfig(forward_periods=5)
        raw = datasets.make_event_panel(
            n_assets=50, n_dates=400, event_rate=0.05,
            post_event_drift_bps=100.0, forward_periods=5, seed=42,
        )
        prepared = fl.preprocess(raw, config=cfg)
        # Expected drift = 100 bps / 1e4 / 5 per bar, N bars measured →
        # mean forward_return per bar on +1 events ≈ 2e-4 (before noise).
        mean_fwd_pos = (
            prepared.filter(pl.col("factor") == 1.0)["forward_return"].mean()
        )
        mean_fwd_neg = (
            prepared.filter(pl.col("factor") == -1.0)["forward_return"].mean()
        )
        assert mean_fwd_pos > 0.5e-4, (
            f"expected ~2e-4 drift on +1 events, got {mean_fwd_pos:.2e}"
        )
        assert mean_fwd_neg < -0.5e-4, (
            f"expected ~-2e-4 drift on -1 events, got {mean_fwd_neg:.2e}"
        )

    def test_seed_is_deterministic(self):
        a = datasets.make_event_panel(n_assets=10, n_dates=60, seed=9)
        b = datasets.make_event_panel(n_assets=10, n_dates=60, seed=9)
        assert a.equals(b)
