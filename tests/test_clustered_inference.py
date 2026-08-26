"""Correlated units must not be counted as independent draws.

Three pooled statistics treated their units as iid when the panel's defining
feature is that they are not: ``event_hit_rate`` pooled every event into one
binomial, ``event_ic`` pooled every event into one Spearman, and
``common_beta`` averaged per-asset betas that load on a common component. The
two event statistics deflate by the Kolari-Pynnönen machinery ``bmp_z`` and
``directional_hit_rate`` already used; ``common_beta`` reads its SE off the
calendar-time equal-weight portfolio.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import scipy.stats as sp_stats
from factrix._codes import WarningCode
from factrix.metrics._helpers import KP_MATERIAL_SCALE, _kp_cluster_scale
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
    effect: float = 0.0,
) -> pl.DataFrame:
    """Every asset fires on the same dates, and returns share a common shock.

    The maximum-clustering design: 800 events resting on 40 independent
    periods, with a true null. ``effect`` plants ``effect * factor`` on the
    event rows' returns (in units of the 1% return sd), so both the sign and
    the magnitude link are real.
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
            factor = (1.0 + rng.uniform(0, 1)) if d in event_days else 0.0
            rows.append(
                {
                    "date": dates[d],
                    "asset_id": f"A{a}",
                    "factor": factor,
                    "forward_return": float(rets[d] + 0.01 * effect * factor),
                }
            )
    return pl.DataFrame(rows)


def _correlated_beta_panel(
    seed: int,
    *,
    n_assets: int = 8,
    n_dates: int = 250,
    rho: float = 0.5,
    beta_mean: float = 0.0,
    beta_sd: float = 0.0,
    hetero_vol: bool = False,
) -> pl.DataFrame:
    """Per-asset betas drawn around ``beta_mean`` with spread ``beta_sd``;
    residuals share a common component of strength ``rho``."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    common = rng.normal(0, 1, n_dates)
    factor = rng.normal(0, 1, n_dates)
    betas = beta_mean + beta_sd * rng.normal(0, 1, n_assets)
    scales = np.linspace(0.5, 2.0, n_assets) if hetero_vol else np.ones(n_assets)
    rows = []
    for a in range(n_assets):
        idio = rng.normal(0, 1, n_dates)
        eps = scales[a] * (np.sqrt(rho) * common + np.sqrt(1 - rho) * idio)
        rets = betas[a] * factor + eps
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


class TestTheDeflator:
    def test_design_effect_is_the_identity_without_clustering(self):
        assert _kp_cluster_scale(0.0, 20.0) == pytest.approx(1.0)

    def test_design_effect_matches_the_kish_formula(self):
        assert _kp_cluster_scale(0.4, 20.0) == pytest.approx(
            1.0 / np.sqrt(1.0 + 19 * 0.4)
        )

    def test_immaterial_deflation_is_disclosed_but_not_applied(self):
        # Independent triggers on 20 names: the ICC is sampling noise around 0
        # and the design effect sits within 5% of the identity. Applying it
        # fired the code on every multi-asset run and moved event_hit_rate off
        # the exact binomial for nothing.
        panel = _clustered_event_panel(0, rho=0.0, n_event_dates=200)
        rng = np.random.default_rng(0)
        thin = panel.with_columns(
            pl.when(pl.Series(rng.random(panel.height) < 0.05))
            .then(pl.col("factor"))
            .otherwise(0.0)
            .alias("factor")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = event_hit_rate(thin, forward_periods=1)
        assert result.metadata["kolari_pynnonen_applied"] is False
        assert result.metadata["kolari_pynnonen_scaling"] >= KP_MATERIAL_SCALE
        assert result.metadata["stat_type"] == "binomial_hits"
        assert WarningCode.EVENT_CLUSTERING_ADJUSTED.value not in result.warning_codes


class TestEventHitRateUnderClustering:
    def test_clustered_events_switch_to_the_deflated_normal(self):
        panel = _clustered_event_panel(0)
        with pytest.warns(UserWarning, match="events share periods"):
            result = event_hit_rate(panel, forward_periods=_H)
        assert WarningCode.EVENT_CLUSTERING_ADJUSTED.value in result.warning_codes
        assert result.metadata["stat_type"] == "z"
        assert result.metadata["kolari_pynnonen_applied"] is True
        assert result.metadata["kolari_pynnonen_scaling"] < KP_MATERIAL_SCALE
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


class TestDeflatedTestsKeepPower:
    """A future 'fix' must not pass the size tests by rejecting nothing."""

    def test_event_hit_rate_rejects_a_planted_effect(self):
        reps = 12
        rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                r = event_hit_rate(
                    _clustered_event_panel(seed, effect=0.6), forward_periods=_H
                )
                if r.p_value is not None and r.p_value < 0.05:
                    rejected += 1
        assert rejected >= reps // 2

    def test_event_ic_rejects_a_planted_magnitude_link(self):
        reps = 12
        rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                r = event_ic(
                    _clustered_event_panel(seed, effect=0.6), forward_periods=_H
                )
                if r.p_value is not None and r.p_value < 0.05:
                    rejected += 1
        assert rejected >= reps // 2


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


class TestCommonBetaCalendarTimeSe:
    def test_correlated_betas_widen_the_se_and_disclose_it(self):
        betas = compute_common_betas(
            _correlated_beta_panel(0, rho=0.9), forward_periods=1
        )["factor"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = common_beta(betas)
        assert result.metadata["calendar_time_se_applied"] is True
        assert result.metadata["ew_portfolio_beta_se"] > 0
        assert result.metadata["ew_portfolio_periods"] == 250
        assert result.metadata["beta_dispersion_excess"] >= 0.0
        # A shared component makes the iid SE far too small: |t| shrinks.
        assert abs(result.stat) < abs(result.metadata["stat_uncorrected"])
        # The point estimate is untouched, and on a rectangular panel it is
        # exactly the equal-weight portfolio's slope.
        assert result.value == pytest.approx(float(betas["beta"].mean()))
        assert result.metadata["ew_portfolio_beta"] == pytest.approx(result.value)

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
        assert result.metadata["calendar_time_se_applied"] is False
        assert (
            result.metadata["calendar_time_se_source"] == "unavailable_hand_built_frame"
        )
        assert "stat_uncorrected" not in result.metadata

    @pytest.mark.parametrize(
        ("rho", "beta_sd", "hetero_vol"),
        [(0.0, 0.0, False), (0.5, 0.0, False), (0.5, 0.0, True), (0.5, 1.0, False)],
    )
    def test_size_is_near_nominal_across_regimes(self, rho, beta_sd, hetero_vol):
        reps = 40
        rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                betas = compute_common_betas(
                    _correlated_beta_panel(
                        seed, rho=rho, beta_sd=beta_sd, hetero_vol=hetero_vol
                    ),
                    forward_periods=1,
                )["factor"]
                p = common_beta(betas).p_value
                if p is not None and p < 0.05:
                    rejected += 1
        # Measured at 300 draws (N=20, T=300): 4.0% / 5.0% at rho 0 / 0.5,
        # 5.3% on the heteroskedastic null, 7.0% with beta sd 1 around 0 at
        # rho 0.5. The Kolari-Pynnonen factor this replaced gave 0.7% on the
        # heteroskedastic null and 0.0% power with dispersed betas.
        assert rejected / reps <= 0.20, (rho, beta_sd, hetero_vol, rejected)

    def test_uncorrected_statistic_reproduces_the_old_failure(self):
        # Guard on the guard: the size test above must not pass vacuously.
        reps = 20
        rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                betas = compute_common_betas(
                    _correlated_beta_panel(seed, rho=0.9), forward_periods=1
                )["factor"]
                r = common_beta(betas)
                t0 = r.metadata["stat_uncorrected"]
                if 2 * sp_stats.t.sf(abs(t0), r.n_obs - 1) < 0.05:
                    rejected += 1
        assert rejected / reps >= 0.5

    def test_dispersed_betas_keep_their_power(self):
        # The regime the Kolari-Pynnonen factor destroyed: true betas spread
        # around a non-zero mean with a shared residual component. Measured
        # at 300 draws (N=20, T=300, mean 0.4, sd 0.5, rho 0.5): 0.0% under
        # the old deflator, 80.7% with the calendar-time SE.
        reps = 20
        rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                betas = compute_common_betas(
                    _correlated_beta_panel(
                        seed, n_assets=20, rho=0.5, beta_mean=0.4, beta_sd=0.5
                    ),
                    forward_periods=1,
                )["factor"]
                p = common_beta(betas).p_value
                if p is not None and p < 0.05:
                    rejected += 1
        assert rejected / reps >= 0.5


def _skewed_panel(
    seed: int,
    *,
    n: int = 2000,
    event_every: int = 9,
    sign: float = 1.0,
    sign_mix: float = 0.0,
) -> pl.DataFrame:
    """One asset, no signal, but a right-skewed return: positive well under
    half the time even though its mean is zero.

    ``sign`` is the factor value on every event; ``sign_mix`` flips that share
    of the events to the opposite side, so the hit rate is tested on a
    mixed-sign trigger.
    """
    rng = np.random.default_rng(seed)
    # Lognormal-minus-mean: E[x] = 0, median < 0, so P(x > 0) < 0.5.
    raw = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    rets = 0.01 * (raw - raw.mean())
    factor = np.array(
        [sign if i >= 60 and i % event_every == 0 else 0.0 for i in range(n)]
    )
    if sign_mix > 0:
        flip = (factor != 0) & (rng.random(n) < sign_mix)
        factor[flip] = -factor[flip]
    return pl.DataFrame(
        {
            "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
            "asset_id": ["A"] * n,
            "factor": factor,
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
        # Every event is long, so the tested null is the unsigned rate itself.
        assert p0 == pytest.approx(result.metadata["sign_base_rate_up"])
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

    def test_short_events_are_tested_against_the_complement(self):
        # A hit on a short event is AR < 0, whose null frequency is 1 - p_up.
        # Testing it against p_up read the skew as near-certain skill.
        panel = _skewed_panel(0, sign=-1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = event_hit_rate(panel, forward_periods=_H)
        p_up = result.metadata["sign_base_rate_up"]
        assert p_up < 0.45
        assert result.metadata["sign_base_rate"] == pytest.approx(1.0 - p_up)
        assert result.metadata["h0"] == f"p={1.0 - p_up:.4f}"
        assert result.p_value == pytest.approx(
            float(
                sp_stats.binomtest(
                    result.metadata["n_hits"], result.n_obs, 1.0 - p_up
                ).pvalue
            )
        )

    @pytest.mark.parametrize(("sign", "sign_mix"), [(-1.0, 0.0), (1.0, 0.5)])
    def test_skew_is_not_read_as_skill_on_short_or_mixed_events(self, sign, sign_mix):
        # Measured at 300 draws on a 20-asset skewed panel: 100% rejection
        # with every event short and 93% with a 50/50 mix under the unsigned
        # null; 4.7% / 6.0% under the mixture.
        reps = 12
        rejected = rejected_unsigned = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                r = event_hit_rate(
                    _skewed_panel(seed, sign=sign, sign_mix=sign_mix),
                    forward_periods=_H,
                )
                if r.p_value is not None and r.p_value < 0.05:
                    rejected += 1
                unsigned = float(
                    sp_stats.binomtest(
                        r.metadata["n_hits"], r.n_obs, r.metadata["sign_base_rate_up"]
                    ).pvalue
                )
                if unsigned < 0.05:
                    rejected_unsigned += 1
        assert rejected_unsigned >= reps // 2
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
        assert result.metadata["sign_base_rate_up"] == 0.5
        assert result.metadata["sign_base_rate"] == 0.5
        assert result.metadata["sign_base_rate_source"] == "assumed_symmetric"
