"""Tests for factrix.metrics.directional_hit_rate (Pesaran-Timmermann)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._types import MIN_DIRECTIONAL_PAIRS_HARD, MIN_DIRECTIONAL_PAIRS_WARN
from factrix.metrics.directional_hit_rate import directional_hit_rate
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
        n = MIN_DIRECTIONAL_PAIRS_HARD - 1
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_directional_samples"
        # Short-circuit is labelled on the pairs axis (pooled (date, asset)
        # directional trials), not periods.
        assert result.n_obs_axis == "pairs"

    def test_degenerate_one_signed_predictor(self):
        rng = np.random.default_rng(6)
        x = np.abs(rng.normal(size=100)) + 0.1  # all positive
        y = rng.normal(size=100)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "degenerate_directional_variance"
        assert result.n_obs_axis == "pairs"

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
        assert result.n_obs_axis == "pairs"
        assert result.value == pytest.approx(1.0)


class TestPairsAxisWarnTier:
    def test_thin_pooled_sample_warns_but_returns(self):
        # HARD <= n_pairs < WARN: the PT hit rate is still returned, but the
        # normal approximation is power-thin, so the result carries
        # FEW_DIRECTIONAL_PAIRS and a UserWarning fires.
        rng = np.random.default_rng(11)
        n = MIN_DIRECTIONAL_PAIRS_WARN - 5
        assert MIN_DIRECTIONAL_PAIRS_HARD <= n < MIN_DIRECTIONAL_PAIRS_WARN
        x = rng.normal(size=n)
        y = 0.6 * x + rng.normal(size=n)
        with pytest.warns(UserWarning, match="MIN_DIRECTIONAL_PAIRS_WARN"):
            result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert not math.isnan(result.value)
        assert result.n_obs == n
        assert result.n_obs_axis == "pairs"
        assert WarningCode.FEW_DIRECTIONAL_PAIRS.value in result.warning_codes

    def test_ample_pooled_sample_no_warn(self):
        # n_pairs >= WARN: clean tier, no FEW_DIRECTIONAL_PAIRS code.
        rng = np.random.default_rng(12)
        x = rng.normal(size=200)
        y = 0.5 * x + rng.normal(size=200)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert WarningCode.FEW_DIRECTIONAL_PAIRS.value not in result.warning_codes

    def test_wide_short_panel_clears_floor_on_pairs(self):
        # n_periods (5) < HARD but pooled pairs (5 * 40 = 200) >> WARN: the
        # metric computes a real result because the floor is on pairs, not
        # periods.
        rng = np.random.default_rng(13)
        n_dates, n_assets = 5, 40
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = {
            "date": [d for d in dates for _ in range(n_assets)],
            "asset_id": [f"A{a}" for _ in dates for a in range(n_assets)],
            "factor": rng.normal(size=n_dates * n_assets),
            "forward_return": rng.normal(size=n_dates * n_assets),
        }
        df = pl.DataFrame(rows)
        result = directional_hit_rate(df, forward_periods=1)
        assert not math.isnan(result.value)
        assert result.n_obs == n_dates * n_assets
        assert result.n_obs_axis == "pairs"


class TestCrossSectionalCorrection:
    @staticmethod
    def _panel(common_weight: float, seed: int) -> pl.DataFrame:
        # n_assets per date; ``common_weight`` injects a shared daily shock so
        # same-date hits are cross-sectionally correlated.
        rng = np.random.default_rng(seed)
        n_dates, n_assets = 60, 30
        rows = []
        for di in range(n_dates):
            d = date(2020, 1, 1) + timedelta(days=di)
            mkt = rng.normal()
            for a in range(n_assets):
                f = rng.normal()
                r = 0.3 * f + common_weight * mkt + 0.3 * rng.normal()
                rows.append(
                    {"date": d, "asset_id": f"A{a}", "factor": f, "forward_return": r}
                )
        return pl.DataFrame(rows)

    def test_single_asset_is_not_adjusted(self):
        # One trial per date → no within-date cross-section → exact PT.
        rng = np.random.default_rng(20)
        x = rng.normal(size=200)
        y = 0.5 * x + rng.normal(size=200)
        result = directional_hit_rate(_ts_panel(x, y), forward_periods=1)
        assert result.metadata["kolari_pynnonen_applied"] is False
        assert result.metadata["kolari_pynnonen_r"] is None
        assert "stat_uncorrected" not in result.metadata
        _, ref_stat = _pt_reference(x, y)
        assert result.stat == pytest.approx(ref_stat)

    def test_within_date_correlation_deflates_statistic(self):
        # A heavy shared daily shock correlates same-date hits; the K-P
        # deflation must shrink |S_n| and raise the one-sided p-value.
        result = self._panel(common_weight=0.9, seed=0)
        result = directional_hit_rate(result, forward_periods=1)
        m = result.metadata
        assert m["kolari_pynnonen_applied"] is True
        assert m["kolari_pynnonen_r"] > 0.0
        assert abs(result.stat) < abs(m["stat_uncorrected"])
        # Axis is unchanged: the estimand stays the pooled-pairs hit rate.
        assert result.n_obs_axis == "pairs"
        assert "Kolari-Pynnönen" in m["method"]

    def test_applies_documented_kp_scale(self):
        # The reported statistic is exactly the raw S_n times the
        # Kolari-Pynnönen factor √((1-r)/(1+(n_eff-1)r)) from metadata.
        result = directional_hit_rate(self._panel(common_weight=0.5, seed=3))
        m = result.metadata
        assert m["kolari_pynnonen_applied"] is True
        r, n_eff = m["kolari_pynnonen_r"], m["kolari_pynnonen_n_eff"]
        scale = math.sqrt((1.0 - r) / (1.0 + (n_eff - 1.0) * r))
        assert result.stat == pytest.approx(m["stat_uncorrected"] * scale)


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
