"""Correlated units must not be counted as independent draws.

Three pooled statistics treated their units as iid when the panel's defining
feature is that they are not: ``event_hit_rate`` pooled every event into one
binomial, ``event_ic`` pooled every event into one Spearman, and
``common_beta`` averaged per-asset betas that load on a common component. All
three now deflate by the Kolari-Pynnönen machinery ``bmp_z`` and
``directional_hit_rate`` already used.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import scipy.stats as sp_stats
from factrix._codes import WarningCode
from factrix.metrics._helpers import (
    _kp_cluster_scale,
    _kp_single_cross_section_scale,
)
from factrix.metrics._primitives import compute_common_betas
from factrix.metrics.common_beta import common_beta
from factrix.metrics.event_quality import event_hit_rate, event_ic

_H = 5


def _clustered_event_panel(
    seed: int,
    *,
    n_assets: int = 20,
    n_dates: int = 300,
    n_event_dates: int = 40,
    rho: float = 0.6,
    burn: int = 60,
) -> pl.DataFrame:
    """Every asset fires on the same dates, and returns share a common shock.

    The maximum-clustering design: 800 events resting on 40 independent
    periods, with a true null.
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    common = rng.normal(0, 1, n_dates)
    event_days = set(
        rng.choice(np.arange(burn, n_dates), size=n_event_dates, replace=False).tolist()
    )
    rows = []
    for a in range(n_assets):
        idio = rng.normal(0, 1, n_dates)
        rets = 0.01 * (np.sqrt(rho) * common + np.sqrt(1 - rho) * idio)
        for d in range(n_dates):
            rows.append(
                {
                    "date": dates[d],
                    "asset_id": f"A{a}",
                    "factor": (1.0 + rng.uniform(0, 1)) if d in event_days else 0.0,
                    "forward_return": float(rets[d]),
                }
            )
    return pl.DataFrame(rows)


def _correlated_beta_panel(
    seed: int, *, n_assets: int = 8, n_dates: int = 250, rho: float = 0.5
) -> pl.DataFrame:
    """True beta zero; returns share a common component of strength ``rho``."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    common = rng.normal(0, 1, n_dates)
    factor = rng.normal(0, 1, n_dates)
    rows = []
    for a in range(n_assets):
        idio = rng.normal(0, 1, n_dates)
        rets = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
        for d in range(n_dates):
            rows.append(
                {
                    "date": dates[d],
                    "asset_id": f"A{a}",
                    "factor": float(factor[d]),
                    "forward_return": float(rets[d]),
                }
            )
    return pl.DataFrame(rows)


class TestTheTwoDeflators:
    """The design effect and the full K-P factor are not interchangeable."""

    def test_design_effect_is_the_identity_without_clustering(self):
        assert _kp_cluster_scale(0.0, 20.0) == pytest.approx(1.0)

    def test_design_effect_matches_the_kish_formula(self):
        assert _kp_cluster_scale(0.4, 20.0) == pytest.approx(
            1.0 / np.sqrt(1.0 + 19 * 0.4)
        )

    def test_single_cross_section_factor_restores_the_shrunk_denominator(self):
        # sqrt((1 - r) / (1 + (n - 1) r)): strictly smaller than the design
        # effect, because a single cross-section's dispersion is itself
        # deflated by the correlation.
        assert _kp_single_cross_section_scale(0.5, 8) == pytest.approx(
            np.sqrt(0.5 / (1.0 + 7 * 0.5))
        )
        assert _kp_single_cross_section_scale(0.5, 8) < _kp_cluster_scale(0.5, 8.0)

    def test_single_cross_section_factor_is_the_identity_without_correlation(self):
        assert _kp_single_cross_section_scale(0.0, 8) == pytest.approx(1.0)


class TestEventHitRateUnderClustering:
    def test_clustered_events_switch_to_the_deflated_normal(self):
        panel = _clustered_event_panel(0)
        with pytest.warns(UserWarning, match="events share periods"):
            result = event_hit_rate(panel, forward_periods=_H)
        assert WarningCode.EVENT_CLUSTERING_ADJUSTED.value in result.warning_codes
        assert result.metadata["stat_type"] == "z"
        assert result.metadata["kolari_pynnonen_applied"] is True
        # The deflation only ever widens the p-value.
        assert abs(result.stat) < abs(result.metadata["stat_uncorrected"])

    def test_one_event_per_period_keeps_the_exact_binomial(self):
        rng = np.random.default_rng(1)
        n = 400
        rets = rng.normal(0, 0.01, n)
        panel = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "factor": [1.0 if i >= 60 and i % 7 == 0 else 0.0 for i in range(n)],
                "forward_return": rets,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = event_hit_rate(panel, forward_periods=_H)
        assert WarningCode.EVENT_CLUSTERING_ADJUSTED.value not in result.warning_codes
        assert result.metadata["stat_type"] == "binomial_hits"
        assert result.metadata["kolari_pynnonen_applied"] is False

    def test_rejection_rate_collapses_on_a_clustered_null(self):
        reps = 15
        rejected = before = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                r = event_hit_rate(_clustered_event_panel(seed), forward_periods=_H)
                if r.p_value is not None and r.p_value < 0.05:
                    rejected += 1
                z0 = r.metadata["stat_uncorrected"]
                if 2 * sp_stats.norm.sf(abs(z0)) < 0.05:
                    before += 1
        # Measured at 40 reps: 50.0% uncorrected, 0.0% corrected. The deflator
        # is conservative in this extreme (800 events on 40 periods) — it trades
        # power for size rather than splitting the difference.
        assert before >= 5
        assert rejected <= 2


class TestEventIcInference:
    def test_fisher_z_uses_the_spearman_standard_error(self):
        rng = np.random.default_rng(2)
        n = 400
        rets = rng.normal(0, 0.01, n)
        magnitudes = rng.uniform(0.5, 2.0, n)
        panel = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "factor": [
                    float(magnitudes[i]) if i >= 60 and i % 7 == 0 else 0.0
                    for i in range(n)
                ],
                "forward_return": rets,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = event_ic(panel, forward_periods=_H)
        rho, n_obs = result.value, result.n_obs
        assert result.stat == pytest.approx(np.arctanh(rho) * np.sqrt(n_obs - 3) / 1.06)
        # One approximation behind stat and p, not two.
        assert result.p_value == pytest.approx(
            float(2 * sp_stats.norm.sf(abs(result.stat)))
        )


class TestCommonBetaUnderCrossAssetCorrelation:
    def test_correlated_betas_are_deflated_and_disclosed(self):
        betas = compute_common_betas(_correlated_beta_panel(0, rho=0.9))["factor"]
        with pytest.warns(UserWarning, match="mean pairwise correlation"):
            result = common_beta(betas)
        assert WarningCode.EVENT_CLUSTERING_ADJUSTED.value in result.warning_codes
        assert result.metadata["cross_asset_correlation_applied"] is True
        assert result.metadata["residual_mean_pairwise_corr"] > 0.5
        assert abs(result.stat) < abs(result.metadata["stat_uncorrected"])
        # The point estimate is untouched.
        assert result.value == pytest.approx(float(betas["beta"].mean()))

    def test_hand_built_frame_without_the_estimate_says_so(self):
        frame = pl.DataFrame(
            {
                "asset_id": ["A", "B", "C", "D"],
                "beta": [1.0, 1.1, 0.9, 1.05],
                "alpha": [0.0] * 4,
                "t_stat": [3.0] * 4,
                "r_squared": [0.5] * 4,
                "n_obs": [50] * 4,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = common_beta(frame)
        assert result.metadata["cross_asset_correlation_applied"] is False
        assert (
            result.metadata["cross_asset_correlation_source"]
            == "unavailable_hand_built_frame"
        )

    @pytest.mark.parametrize("rho", [0.0, 0.5])
    def test_size_is_near_nominal_at_every_correlation(self, rho):
        reps = 40
        rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                betas = compute_common_betas(_correlated_beta_panel(seed, rho=rho))[
                    "factor"
                ]
                p = common_beta(betas).p_value
                if p is not None and p < 0.05:
                    rejected += 1
        # Measured at 200 draws: 3.5% (rho=0), 3.5% (0.5), 5.0% (0.9), against
        # 3.5% / 48.5% / 81.5% uncorrected.
        assert rejected / reps <= 0.20, (rho, rejected)

    def test_uncorrected_statistic_reproduces_the_old_failure(self):
        # Guard on the guard: the size test above must not pass vacuously.
        reps = 20
        rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                betas = compute_common_betas(_correlated_beta_panel(seed, rho=0.9))[
                    "factor"
                ]
                r = common_beta(betas)
                t0 = r.metadata["stat_uncorrected"]
                if 2 * sp_stats.t.sf(abs(t0), r.n_obs - 1) < 0.05:
                    rejected += 1
        assert rejected / reps >= 0.5


def _skewed_panel(seed: int, *, n: int = 2000, event_every: int = 9) -> pl.DataFrame:
    """One asset, no signal, but a right-skewed return: positive well under
    half the time even though its mean is zero."""
    rng = np.random.default_rng(seed)
    # Lognormal-minus-mean: E[x] = 0, median < 0, so P(x > 0) < 0.5.
    raw = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    rets = 0.01 * (raw - raw.mean())
    return pl.DataFrame(
        {
            "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
            "asset_id": ["A"] * n,
            "factor": [
                1.0 if i >= 60 and i % event_every == 0 else 0.0 for i in range(n)
            ],
            "forward_return": rets,
        }
    )


class TestGeneralisedSignNull:
    """event_hit_rate tests against the base rate, not a coin flip."""

    def test_null_is_the_non_event_positive_share(self):
        panel = _skewed_panel(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = event_hit_rate(panel, forward_periods=_H)
        p0 = result.metadata["sign_base_rate"]
        assert result.metadata["sign_base_rate_source"] == "non_event_rows"
        # The skew is real: this series is positive well under half the time.
        assert p0 < 0.45
        assert result.metadata["h0"] == f"p={p0:.4f}"
        # And the reported p is the exact binomial against that null.
        assert result.p_value == pytest.approx(
            float(
                sp_stats.binomtest(result.metadata["n_hits"], result.n_obs, p0).pvalue
            )
        )

    def test_skew_alone_is_not_read_as_skill(self):
        # Against a 0.5 null this panel looks strongly "anti-predictive"; the
        # generalised null says there is nothing there.
        reps = 12
        rejected_vs_half = rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                r = event_hit_rate(_skewed_panel(seed), forward_periods=_H)
                if r.p_value is not None and r.p_value < 0.05:
                    rejected += 1
                naive = float(
                    sp_stats.binomtest(r.metadata["n_hits"], r.n_obs, 0.5).pvalue
                )
                if naive < 0.05:
                    rejected_vs_half += 1
        assert rejected_vs_half >= reps // 2
        assert rejected <= 2

    def test_too_few_non_event_rows_fall_back_to_symmetry(self):
        # Every row is an event: nothing to estimate the base rate from.
        n = 120
        rng = np.random.default_rng(3)
        panel = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "factor": [1.0] * n,
                "forward_return": rng.normal(0, 0.01, n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = event_hit_rate(panel, forward_periods=_H)
        assert result.metadata["sign_base_rate"] == 0.5
        assert result.metadata["sign_base_rate_source"] == "assumed_symmetric"
