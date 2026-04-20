"""Insufficient-data propagation: metric.metadata → profile.insufficient_metrics → diagnose rule."""

from __future__ import annotations

import math

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import factorlib as fl
from factorlib.metrics.caar import caar as caar_metric
from factorlib.metrics.ic import ic as ic_metric, compute_ic


def _cs_panel(n_dates: int, n_assets: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    prices = {f"a{i}": 100.0 for i in range(n_assets)}
    rows: list[dict] = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        r = 0.2 * f * 0.01 + 0.8 * 0.01 * rng.standard_normal(n_assets)
        for i in range(n_assets):
            prices[f"a{i}"] *= (1 + r[i])
            rows.append({
                "date": d, "asset_id": f"a{i}",
                "factor": float(f[i]), "price": float(prices[f"a{i}"]),
            })
    raw = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    return fl.preprocess(raw, config=fl.CrossSectionalConfig())


class TestMetricReasonContract:
    def test_ic_short_circuit_populates_reason(self):
        tiny = pl.DataFrame({
            "date": [datetime(2024, 1, i) for i in range(1, 6)],
            "ic": [0.01] * 5,
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        m = ic_metric(tiny, forward_periods=1)
        assert math.isnan(m.value)
        assert m.metadata["reason"] == "insufficient_ic_periods"
        assert m.metadata["n_observed"] == 5
        assert m.metadata["min_required"] >= 5

    def test_caar_short_circuit_populates_reason(self):
        tiny = pl.DataFrame({
            "date": [datetime(2024, 1, i) for i in range(1, 4)],
            "caar": [0.001, 0.002, 0.003],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        m = caar_metric(tiny, forward_periods=1)
        assert math.isnan(m.value)
        assert m.metadata["reason"] == "insufficient_event_dates"
        assert m.metadata["n_observed"] == 3

    def test_healthy_metric_has_no_reason(self):
        df = _cs_panel(n_dates=80, n_assets=30, seed=1)
        ic_series = compute_ic(df.with_columns(
            pl.col("price").pct_change().over("asset_id").shift(-1).alias("forward_return"),
        ))
        m = ic_metric(ic_series, forward_periods=5)
        assert "reason" not in m.metadata


class TestProfileInsufficientField:
    def test_healthy_profile_has_empty_tuple(self):
        df = _cs_panel(n_dates=100, n_assets=30, seed=42)
        profile = fl.evaluate(df, "healthy", factor_type="cross_sectional")
        assert profile.insufficient_metrics == ()

    def test_tiny_panel_profile_lists_insufficient_metrics(self):
        # 8 dates + 5 assets → IC can't run (MIN_IC_PERIODS cross-section);
        # even if forward returns exist they'll be stripped.
        df = _cs_panel(n_dates=8, n_assets=5, seed=7)
        profile = fl.evaluate(df, "tiny", factor_type="cross_sectional")
        # At least one metric should have flagged insufficient data.
        assert len(profile.insufficient_metrics) > 0
        assert all(isinstance(n, str) for n in profile.insufficient_metrics)


class TestDiagnoseInsufficientRule:
    def test_rule_fires_when_insufficient(self):
        df = _cs_panel(n_dates=8, n_assets=5, seed=99)
        profile = fl.evaluate(df, "tiny", factor_type="cross_sectional")
        codes = {d.code for d in profile.diagnose()}
        assert "data.insufficient" in codes

    def test_rule_silent_on_healthy_profile(self):
        df = _cs_panel(n_dates=100, n_assets=30, seed=11)
        profile = fl.evaluate(df, "healthy", factor_type="cross_sectional")
        codes = {d.code for d in profile.diagnose()}
        assert "data.insufficient" not in codes

    def test_rule_message_lists_affected_metrics(self):
        df = _cs_panel(n_dates=8, n_assets=5, seed=123)
        profile = fl.evaluate(df, "tiny", factor_type="cross_sectional")
        data_diag = next(
            d for d in profile.diagnose() if d.code == "data.insufficient"
        )
        # Message should mention at least one of the short-circuited fields.
        for name in profile.insufficient_metrics:
            if name in data_diag.message:
                break
        else:
            pytest.fail(
                f"data.insufficient message {data_diag.message!r} did not "
                f"mention any of {profile.insufficient_metrics!r}"
            )

    def test_rule_runs_before_type_specific_rules(self):
        df = _cs_panel(n_dates=8, n_assets=5, seed=4)
        profile = fl.evaluate(df, "tiny", factor_type="cross_sectional")
        codes = [d.code for d in profile.diagnose()]
        assert codes.index("data.insufficient") < len(codes)
        # Ensure the first emitted diagnostic is cross-type when it fires.
        assert codes[0] == "data.insufficient"


class TestIntentionalSkipDoesNotFire:
    def test_discrete_event_ic_is_not_treated_as_insufficient(self):
        # Build a discrete {-1, 0, +1} event fixture — event_ic should
        # skip on "not_applicable_discrete_signal", which is *not* an
        # insufficient-data reason and must not show up in
        # profile.insufficient_metrics.
        from factorlib.config import EventConfig
        rng = np.random.default_rng(0)
        n_dates = 300
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        assets = [f"a{i}" for i in range(20)]
        rows: list[dict] = []
        for a in assets:
            price = 100.0
            for d in dates:
                is_event = rng.random() < 0.05
                direction = rng.choice([-1.0, 1.0]) if is_event else 0.0
                ret = rng.normal(0, 0.01)
                if is_event:
                    ret += 0.01 * direction
                price *= (1 + ret)
                rows.append({
                    "date": d, "asset_id": a,
                    "factor": direction, "price": price,
                })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        df = fl.preprocess(df, config=EventConfig())
        profile = fl.evaluate(df, "discrete_signal", config=EventConfig())
        # event_ic is None but should not be listed as "insufficient"
        # (it's an intentional skip, not a data shortage).
        assert profile.event_ic is None
        assert "event_ic" not in profile.insufficient_metrics
