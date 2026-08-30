"""Why ``event_skewness`` publishes no p-value.

``event_skewness`` used to report D'Agostino's skew-test ``z`` and its
two-sided ``p``. The test has no calibrated pooled form, for two
independent reasons, and the same-period clustering deflation
``event_hit_rate`` / ``event_ic`` apply repairs neither. Full table (300
replications per cell, base seed 20260830) in
``reference/statistical-methods`` section 6.

1. **Non-normal signed CARs, no clustering needed.** On the null event
   panel — ``make_event_panel`` with ``post_event_drift_bps=0``,
   ``event_rate=0.02``, ``signal_horizon=5``, ``forward_periods=5``, 1.55
   events per event period — the sampled signed CARs carry excess
   kurtosis (+0.63 at 252 periods, +0.95 at 504), which inflates the
   sampling variance of the skewness above the ``6/n`` D'Agostino's test
   assumes. Size 19.0% and 23.3%; deflation moves it to 17.7% and 22.7%,
   because the cubed-deviation score has near-zero within-period
   correlation when ``sign(factor)`` enters the shared shock with an
   asset-random sign.
2. **Same-period shocks with sign-aligned events.** 40 assets firing
   together on 20 shared periods, one common shock and one common factor
   sign per period: size 30.3% with excess kurtosis −0.09 — pure
   dependence, no non-normality. Here the deflator grips (ICC 0.30) but
   over-corrects to 0.0%.

This module re-runs one cell of each at a cut replication count so the
suite stays fast — the bounds characterise the result rather than pin it
(the Monte-Carlo standard error at 60 replications is ~5pp) — and pins the
contract that no p-value is published.
"""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Callable

import factrix as fx
import numpy as np
import polars as pl
from factrix.metrics._helpers import _deflate_for_within_date_clustering
from factrix.metrics.event_quality import (
    _finite_events,
    _signed_car,
    _spaced_events,
    event_skewness,
)
from factrix.preprocess import compute_forward_return
from scipy import stats as sp_stats

N_ASSETS = 50
N_DATES = 252
N_REPS = 60
OVERLAP = 5
BASE_SEED = 20260830
NOMINAL = 0.05
EXPECTED = ("event_spacing_enforced", "event_window_overlap", "few_events")


def _null_panel(rep: int) -> pl.DataFrame:
    raw = fx.datasets.make_event_panel(
        n_assets=N_ASSETS,
        n_dates=N_DATES,
        post_event_drift_bps=0,
        event_rate=0.02,
        signal_horizon=OVERLAP,
        rng=BASE_SEED + rep,
    )
    return compute_forward_return(raw, forward_periods=OVERLAP)


def _sign_aligned_panel(rep: int) -> pl.DataFrame:
    """40 assets firing together on 20 shared periods.

    Returns are a common per-period shock plus idiosyncratic noise; every
    event on a period carries one common factor sign, so signing the
    abnormal return does not cancel the shared shock the way the
    independently-signed panel null does.
    """
    rng = np.random.default_rng(BASE_SEED + rep)
    n_assets, n_dates = 40, 400
    shock = rng.normal(0, 0.01, n_dates)
    idio = rng.normal(0, 0.01, (n_dates, n_assets))
    price = 100 * np.exp(np.cumsum(shock[:, None] + idio, axis=0))
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_dates)
    ]
    factor = np.zeros((n_dates, n_assets))
    for d in rng.choice(np.arange(80, n_dates - OVERLAP - 2), size=20, replace=False):
        factor[d, :] = rng.choice([-1.0, 1.0])
    raw = pl.DataFrame(
        {
            "asset_id": np.repeat([f"A{j}" for j in range(n_assets)], n_dates),
            "date": pl.Series(list(dates) * n_assets, dtype=pl.Date),
            "price": price.T.reshape(-1),
            "factor": factor.T.reshape(-1),
        }
    )
    return compute_forward_return(raw, forward_periods=OVERLAP)


def _withdrawn_test_rates(
    make_panel: Callable[[int], pl.DataFrame],
) -> tuple[float, float, float, float]:
    """(raw rejection, deflated rejection, z SD, mean excess kurtosis)."""
    raw_rejected = 0
    deflated_rejected = 0
    zs: list[float] = []
    kurtoses: list[float] = []
    for rep in range(N_REPS):
        panel = make_panel(rep)
        events, _, _ = _finite_events(
            panel,
            "factor",
            "forward_return",
            estimation_window=60,
            overlap_periods=OVERLAP,
        )
        events = _spaced_events(
            panel,
            events,
            "event_skewness",
            OVERLAP,
            {},
            [],
            expected_warnings=EXPECTED,
        )
        signed = _signed_car(events, "factor", "_abnormal_return")
        if len(signed) < 20:
            continue
        z_raw = float(sp_stats.skewtest(signed)[0])
        zs.append(z_raw)
        kurtoses.append(float(sp_stats.kurtosis(signed)))
        raw_rejected += abs(z_raw) > sp_stats.norm.isf(NOMINAL / 2)
        centred = signed - signed.mean()
        score = (centred / np.sqrt(np.mean(centred**2))) ** 3
        z_deflated = _deflate_for_within_date_clustering(
            events.select("date", pl.Series("_score", score)),
            "_score",
            z_raw,
            "event_skewness",
            {},
            [],
            expected_warnings=EXPECTED,
        )
        deflated_rejected += abs(z_deflated) > sp_stats.norm.isf(NOMINAL / 2)
    n = len(zs)
    return (
        raw_rejected / n,
        deflated_rejected / n,
        float(np.std(zs)),
        float(np.mean(kurtoses)),
    )


class TestWithdrawnSkewTest:
    def test_non_normal_cars_over_reject_and_deflation_does_not_save_it(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw, deflated, z_sd, kurtosis = _withdrawn_test_rates(_null_panel)
        # Documented 19.0% at 300 replications; the floor is well clear of
        # nominal at the Monte-Carlo error of this replication count.
        assert raw > 0.12
        # The clustering deflation the sibling event tests apply moves it
        # barely (19.0% -> 17.7% at 300 replications): this failure is not
        # same-period clustering.
        assert deflated > 0.12
        # The mechanism: the null z is over-dispersed because the signed CARs
        # are leptokurtic, not normal as D'Agostino's test assumes. Only 1.55
        # events share an event period here, so there is nothing to deflate.
        assert z_sd > 1.15
        assert kurtosis > 0.3

    def test_sign_aligned_clustering_over_rejects_and_deflation_over_corrects(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw, deflated, z_sd, kurtosis = _withdrawn_test_rates(_sign_aligned_panel)
        # Documented 30.3% at 300 replications.
        assert raw > 0.20
        # The deflator does grip on this null (documented ICC 0.30) but does
        # not restore size — it drives the rejection rate to 0.0%, a test
        # with no power rather than a calibrated one. A deflator strong
        # enough to touch this cell annihilates it; one weak enough to spare
        # it does nothing for the cell above.
        assert deflated < 0.02
        assert z_sd > 1.6
        # Pure dependence: none of the non-normality that breaks the panel
        # null is present here.
        assert kurtosis < 0.2


class TestNoPublishedTest:
    def test_no_p_value_or_stat_on_a_null_panel(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = event_skewness(_null_panel(0), overlap_periods=OVERLAP)
        assert result.p_value is None
        assert result.stat is None
        assert result.alternative is None
        assert np.isfinite(result.value)
