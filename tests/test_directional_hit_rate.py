"""Tests for factrix.metrics.directional_hit_rate (Pesaran-Timmermann)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.directional_hit_rate import (
    MIN_DIRECTIONAL_PERIODS,
    directional_hit_rate,
)
from factrix.metrics.hit_rate import hit_rate


def _ts_panel(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    """Single-asset panel from prediction ``x`` and realisation ``y``."""
    n = len(x)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {"date": dates, "asset_id": ["A"] * n, "factor": x, "forward_return": y}
    )


def _pt_reference(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Independent re-derivation of (P̂, S_n) for cross-checking."""
    xs, ys = x > 0, y > 0
    n = len(x)
    p_correct = float(np.mean(xs == ys))
    p_x, p_y = float(np.mean(xs)), float(np.mean(ys))
    p_star = p_x * p_y + (1 - p_x) * (1 - p_y)
    var_p_hat = p_star * (1 - p_star) / n
    var_p_star = (
        (2 * p_y - 1) ** 2 * p_x * (1 - p_x) / n
        + (2 * p_x - 1) ** 2 * p_y * (1 - p_y) / n
        + 4 * p_x * p_y * (1 - p_x) * (1 - p_y) / n**2
    )
    return p_correct, (p_correct - p_star) / math.sqrt(var_p_hat - var_p_star)


class TestStatisticCorrectness:
    def test_matches_independent_reference(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = 0.5 * x + rng.normal(size=200)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)

        ref_value, ref_stat = _pt_reference(x, y)
        assert result.value == pytest.approx(ref_value)
        assert result.stat == pytest.approx(ref_stat)
        assert result.metadata["method"] == "Pesaran-Timmermann (1992)"
        assert result.metadata["stat_type"] == "z"

    def test_positive_predictor_is_significant(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=300)
        y = x + 0.3 * rng.normal(size=300)  # strong sign agreement
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert result.value > 0.5
        assert result.stat > 0
        assert result.p_value < 0.01

    def test_independent_factor_not_significant(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=400)
        y = rng.normal(size=400)  # no relation
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert result.p_value > 0.05

    def test_negative_predictor_scores_poorly_one_sided(self):
        # PT directional accuracy is one-sided: a sign-inverted predictor has
        # poor directional accuracy (S_n < 0), so the one-sided p stays high.
        rng = np.random.default_rng(3)
        x = rng.normal(size=300)
        y = -x + 0.3 * rng.normal(size=300)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert result.value < 0.5
        assert result.stat < 0
        assert result.p_value > 0.95


class TestDivergenceFromHitRate:
    def test_discounts_persistent_market_drift(self):
        # Market is up ~85% of periods; the factor is sign-independent of it.
        # A naive hit-rate vs 0.5 on the sign-agreement series reads as
        # "skill" purely from the shared upward drift; PT conditions on the
        # marginal up-frequencies and correctly finds no directional skill.
        rng = np.random.default_rng(4)
        n = 400
        y = np.where(rng.random(n) < 0.85, 1.0, -1.0) * (rng.random(n) + 0.1)
        x = np.where(rng.random(n) < 0.85, 1.0, -1.0) * (rng.random(n) + 0.1)

        pt = directional_hit_rate(_ts_panel(x, y), forward_periods=1)

        agree = pl.DataFrame(
            {
                "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "value": (np.sign(x) == np.sign(y)).astype(float),
            }
        )
        naive = hit_rate(agree, forward_periods=1)

        # Naive binomial sees a high hit rate and calls it significant;
        # PT, conditioning on the imbalanced marginals, does not.
        assert naive.value > 0.5
        assert naive.p_value < 0.05
        assert pt.p_value > 0.10


class TestShortCircuits:
    def test_insufficient_samples(self):
        rng = np.random.default_rng(5)
        n = MIN_DIRECTIONAL_PERIODS - 1
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_directional_samples"

    def test_degenerate_one_signed_predictor(self):
        rng = np.random.default_rng(6)
        x = np.abs(rng.normal(size=100)) + 0.1  # all positive
        y = rng.normal(size=100)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "degenerate_directional_variance"

    def test_missing_return_column(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=50)
        df = _ts_panel(x, x).drop("forward_return")
        result = directional_hit_rate(df, forward_periods=1)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "no_return_column"

    def test_zero_signs_are_dropped(self):
        # Exact-zero predictions have undefined direction and must not count.
        x = np.array([0.0] * 30 + [1.0] * 30 + [-1.0] * 30)
        y = np.array([1.0] * 30 + [1.0] * 30 + [-1.0] * 30)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert result.n_obs == 60  # the 30 zero-factor rows dropped
        assert result.value == pytest.approx(1.0)


class TestDispatch:
    def test_runs_on_panel(self):
        import factrix as fx

        raw = fx.datasets.make_cs_panel(n_assets=25, n_dates=120)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        results = fx.evaluate(
            panel,
            metrics={"da": directional_hit_rate()},
            factor_cols=["factor"],
            forward_periods=5,
        )
        out = results["factor"].metrics["da"]
        assert out.name == "da"
        assert 0.0 <= out.value <= 1.0

    def test_runs_on_single_asset_timeseries(self):
        import factrix as fx

        raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=160)
        first = raw["asset_id"].unique()[0]
        ts = fx.preprocess.compute_forward_return(
            raw.filter(pl.col("asset_id") == first), forward_periods=5
        )
        result = directional_hit_rate(ts, forward_periods=5)
        assert math.isnan(result.value) or 0.0 <= result.value <= 1.0
