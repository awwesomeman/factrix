"""Tests for factrix.datasets — synthetic panels with calibrated IC.

These tests own dataset schema and shape invariants. End-to-end
``preprocess`` -> ``evaluate`` behavior is covered by procedure-specific tests.
"""

from __future__ import annotations

import polars as pl
import pytest
from factrix import datasets
from factrix.metrics.event_quality import event_ic
from factrix.preprocess import compute_forward_return


class TestMakeCsPanelSchema:
    def test_canonical_columns_and_dtypes(self):
        df = datasets.make_cs_panel(n_assets=10, n_dates=60, rng=0)
        assert df.columns == ["date", "asset_id", "price", "factor"]
        assert df.schema["date"] == pl.Datetime("ms")
        assert df.schema["asset_id"] == pl.String
        assert df.schema["price"] == pl.Float64
        assert df.schema["factor"] == pl.Float64

    def test_row_count(self):
        df = datasets.make_cs_panel(n_assets=12, n_dates=40, rng=0)
        assert df.height == 12 * 40
        assert df["asset_id"].n_unique() == 12
        assert df["date"].n_unique() == 40

    def test_no_nan_or_inf(self):
        df = datasets.make_cs_panel(n_assets=10, n_dates=60, rng=0)
        for col in ("price", "factor"):
            assert df[col].is_nan().sum() == 0
            assert df[col].is_finite().all()

    def test_seed_is_deterministic(self):
        a = datasets.make_cs_panel(n_assets=8, n_dates=30, rng=123)
        b = datasets.make_cs_panel(n_assets=8, n_dates=30, rng=123)
        assert a.equals(b)

    def test_different_seeds_differ(self):
        a = datasets.make_cs_panel(n_assets=8, n_dates=30, rng=1)
        b = datasets.make_cs_panel(n_assets=8, n_dates=30, rng=2)
        assert not a["factor"].equals(b["factor"])

    def test_raises_on_short_panel(self):
        with pytest.raises(ValueError, match="n_dates"):
            datasets.make_cs_panel(n_assets=5, n_dates=5, signal_horizon=5)

    def test_raises_on_singleton_cross_section(self):
        with pytest.raises(ValueError, match="n_assets"):
            datasets.make_cs_panel(n_assets=1, n_dates=60)


class TestMakeEventPanelSchema:
    def test_canonical_columns_and_dtypes(self):
        df = datasets.make_event_panel(n_assets=10, n_dates=60, rng=0)
        assert df.columns == ["date", "asset_id", "price", "factor"]
        assert df.schema["factor"] == pl.Float64

    def test_row_count(self):
        df = datasets.make_event_panel(n_assets=12, n_dates=40, rng=0)
        assert df.height == 12 * 40

    def test_factor_values_ternary(self):
        df = datasets.make_event_panel(n_assets=20, n_dates=120, event_rate=0.05, rng=0)
        assert set(df["factor"].unique().to_list()) <= {-1.0, 0.0, 1.0}

    def test_event_magnitude_scales_factor(self):
        df = datasets.make_event_panel(
            n_assets=20, n_dates=120, event_rate=0.05, event_magnitude=2.0, rng=0
        )
        assert set(df["factor"].unique().to_list()) <= {-2.0, 0.0, 2.0}

    def test_event_magnitude_jitter_makes_event_ic_usable(self):
        raw = datasets.make_event_panel(
            n_assets=30,
            n_dates=252,
            event_rate=0.08,
            event_magnitude_jitter=0.5,
            post_event_drift_bps=250.0,
            rng=0,
        )
        panel = compute_forward_return(raw, forward_periods=5)
        result = event_ic(panel)
        # jitter gives |factor| variation (not just sign), and drift scales
        # with magnitude, so event_ic resolves to a detectable positive IC
        # instead of short-circuiting on a degenerate ternary signal.
        assert result.metadata.get("reason") is None
        assert result.value > 0

    def test_seed_is_deterministic(self):
        a = datasets.make_event_panel(n_assets=8, n_dates=30, rng=7)
        b = datasets.make_event_panel(n_assets=8, n_dates=30, rng=7)
        assert a.equals(b)

    def test_raises_on_short_panel(self):
        with pytest.raises(ValueError, match="n_dates"):
            datasets.make_event_panel(n_assets=5, n_dates=5, signal_horizon=5)


class TestMakeMultiFactorEventPanelSchema:
    def test_column_count_and_names(self):
        df = datasets.make_multi_factor_event_panel(
            n_factors=3, n_assets=10, n_dates=60, rng=0
        )
        assert df.columns == [
            "date",
            "asset_id",
            "price",
            "factor_0000",
            "factor_0001",
            "factor_0002",
        ]

    def test_row_count(self):
        df = datasets.make_multi_factor_event_panel(
            n_factors=4, n_assets=8, n_dates=50, rng=0
        )
        assert df.height == 8 * 50

    def test_factor_values_ternary(self):
        df = datasets.make_multi_factor_event_panel(
            n_factors=3, n_assets=20, n_dates=120, event_rate=0.05, rng=0
        )
        for col in ["factor_0000", "factor_0001", "factor_0002"]:
            assert set(df[col].unique().to_list()) <= {-1.0, 0.0, 1.0}

    def test_magnitude_jitter_adds_continuous_event_values(self):
        df = datasets.make_multi_factor_event_panel(
            n_factors=1,
            n_assets=20,
            n_dates=120,
            event_rate=0.10,
            event_magnitude_jitter=0.5,
            rng=0,
        )
        events = df.filter(pl.col("factor_0000") != 0)["factor_0000"].abs()
        assert events.n_unique() > 2

    def test_rejects_negative_magnitude_jitter(self):
        with pytest.raises(ValueError, match="event_magnitude_jitter"):
            datasets.make_multi_factor_event_panel(event_magnitude_jitter=-0.1)

    def test_seed_is_deterministic(self):
        a = datasets.make_multi_factor_event_panel(
            n_factors=3, n_assets=8, n_dates=30, rng=7
        )
        b = datasets.make_multi_factor_event_panel(
            n_factors=3, n_assets=8, n_dates=30, rng=7
        )
        assert a.equals(b)

    def test_raises_on_short_panel(self):
        with pytest.raises(ValueError, match="n_dates"):
            datasets.make_multi_factor_event_panel(
                n_factors=2, n_assets=5, n_dates=5, signal_horizon=5
            )


class TestCsPanelFactorPersistence:
    """``factor_persistence`` — the per-asset AR(phi) noise leg of the factor."""

    def test_zero_reproduces_the_iid_factor_panel_exactly(self):
        """The default path must be bit-for-bit unchanged off a given rng."""
        default = datasets.make_cs_panel(n_assets=20, n_dates=80, rng=7)
        explicit = datasets.make_cs_panel(
            n_assets=20, n_dates=80, rng=7, factor_persistence=0.0
        )
        assert default.equals(explicit)

    def test_positive_phi_makes_the_factor_persistent_per_asset(self):
        """A persistent factor, measured along the period grid one asset at a time."""
        import numpy as np
        from factrix._stats.diagnostics import _lag1_autocorr

        def _mean_asset_autocorr(phi: float) -> float:
            panel = datasets.make_cs_panel(
                n_assets=20,
                n_dates=400,
                ic_target=0.0,
                factor_persistence=phi,
                rng=11,
            ).sort(["asset_id", "date"])
            return float(
                np.mean(
                    [
                        _lag1_autocorr(g["factor"].to_numpy())
                        for g in panel.partition_by("asset_id")
                    ]
                )
            )

        assert abs(_mean_asset_autocorr(0.0)) < 0.1
        assert _mean_asset_autocorr(0.9) > 0.5

    def test_persistence_does_not_move_the_realized_ic(self):
        """The AR leg is independent of returns, so ``ic_target`` still governs.

        What it does move is the *precision* of the realized mean IC: a
        persistent factor makes the IC series persistent, so the same number
        of periods carries fewer independent ones and the mean scatters
        wider across seeds. Averaging over seeds is therefore required to
        read the level — a single seed at phi = 0.9 lands 2pp low often
        enough to be worth naming here rather than rediscovering.
        """
        import numpy as np
        from factrix.metrics._primitives import compute_ic

        def _mean_ics(phi: float) -> np.ndarray:
            out = []
            for seed in range(12):
                raw = datasets.make_cs_panel(
                    n_assets=50,
                    n_dates=400,
                    ic_target=0.08,
                    signal_horizon=5,
                    factor_persistence=phi,
                    rng=100 + seed,
                )
                panel = compute_forward_return(raw, forward_periods=5)
                out.append(float(compute_ic(panel)["factor"]["ic"].mean()))
            return np.array(out)

        iid, persistent = _mean_ics(0.0), _mean_ics(0.9)
        assert abs(persistent.mean() - iid.mean()) < 0.02
        assert persistent.std() > iid.std()

    def test_rejects_non_stationary_persistence(self):
        with pytest.raises(ValueError, match="factor_persistence"):
            datasets.make_cs_panel(factor_persistence=1.0)
        with pytest.raises(ValueError, match="factor_persistence"):
            datasets.make_cs_panel(factor_persistence=-0.1)
