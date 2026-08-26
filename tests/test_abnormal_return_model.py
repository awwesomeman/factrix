"""The event family's abnormal-return model.

Every event statistic is defined on an abnormal return $AR = R - E[R]$. Until
:func:`~factrix.metrics._helpers._attach_abnormal_return` existed, each of them
used the raw forward return, so any unconditional drift was read as event
alpha — on a drifting panel whose event dates carry zero information, ``bmp_z``
rejected half of all null draws.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics._helpers import _attach_abnormal_return
from factrix.metrics.caar import bmp_z, caar, compute_caar
from factrix.metrics.corrado_rank import corrado_rank
from factrix.metrics.event_quality import event_hit_rate

_H = 5


def _drift_panel(
    seed: int,
    *,
    n_assets: int = 8,
    n_dates: int = 400,
    mu: float = 0.0008,
    sigma: float = 0.01,
    n_events: int = 25,
    burn: int = 80,
) -> pl.DataFrame:
    """Pure drift, and event dates drawn uniformly at random — zero information.

    Every rejection this panel produces is a false positive by construction.
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for a in range(n_assets):
        rets = rng.normal(mu, sigma, n_dates)
        prices = 100.0 * np.cumprod(1.0 + rets)
        events = set(
            rng.choice(np.arange(burn, n_dates), size=n_events, replace=False).tolist()
        )
        for d in range(n_dates):
            rows.append(
                {
                    "date": dates[d],
                    "asset_id": f"A{a}",
                    "factor": 1.0 if d in events else 0.0,
                    "forward_return": float(rets[d]),
                    "price": float(prices[d]),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestAttachAbnormalReturn:
    def test_matches_a_hand_computed_estimation_window_mean(self):
        rng = np.random.default_rng(0)
        n = 200
        rets = rng.normal(0.001, 0.01, n)
        panel = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "forward_return": rets,
            }
        )
        out, diagnostics = _attach_abnormal_return(
            panel, estimation_window=60, forward_periods=_H
        )
        got = out["_abnormal_return"].to_numpy()

        # AR_t = R_t - mean(R over the 60 rows ending _H rows before t), with
        # the estimate withheld until 20 of those rows exist.
        for t in (24, 90, 199):
            end = t - _H
            start = max(0, end - 60 + 1)
            assert got[t] == pytest.approx(rets[t] - rets[start : end + 1].mean())
        # Nothing before the estimate exists.
        assert np.isnan(got[:24]).all()
        assert diagnostics["abnormal_return_model"] == "mean_adjusted"
        assert diagnostics["estimation_window_lag"] == _H

    def test_estimation_window_ends_before_the_event_window_opens(self):
        # A single huge return must not enter its own estimation window: with a
        # lag of h, the h rows immediately before t are excluded too.
        n = 120
        rets = np.zeros(n)
        rets[100] = 1.0
        panel = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "forward_return": rets,
            }
        )
        out, _ = _attach_abnormal_return(
            panel, estimation_window=60, forward_periods=_H
        )
        got = out["_abnormal_return"].to_numpy()
        assert got[100] == pytest.approx(1.0)  # its own mean is still zero
        # It only starts contaminating estimates h rows later.
        assert got[104] == pytest.approx(0.0)
        assert got[105] == pytest.approx(-1.0 / 60)

    def test_a_supplied_abnormal_return_column_is_honoured(self):
        panel = _drift_panel(0).with_columns(
            (pl.col("forward_return") - 0.5).alias("abnormal_return")
        )
        out, diagnostics = _attach_abnormal_return(panel, forward_periods=_H)
        assert diagnostics["abnormal_return_model"] == "market_adjusted_supplied"
        assert out["_abnormal_return"].to_numpy() == pytest.approx(
            out["abnormal_return"].to_numpy()
        )

    def test_one_nan_does_not_blank_the_whole_window(self):
        # polars propagates float NaN through a rolling aggregate; masking it to
        # null first keeps the next estimation_window events computable.
        n = 200
        rets = np.full(n, 0.01)
        rets[50] = np.nan
        panel = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "forward_return": rets,
            }
        )
        out, _ = _attach_abnormal_return(
            panel, estimation_window=60, forward_periods=_H
        )
        got = out["_abnormal_return"].to_numpy()
        assert np.isnan(got[50])
        assert not np.isnan(got[60])
        assert got[60] == pytest.approx(0.0)


class TestDriftIsNotEventAlpha:
    """The size claim the model exists for."""

    @staticmethod
    def _p_values(panel: pl.DataFrame) -> dict[str, float | None]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {
                "caar": caar(
                    compute_caar(panel, forward_periods=_H), forward_periods=_H
                ).p_value,
                "bmp_z": bmp_z(panel, forward_periods=_H).p_value,
                "corrado_rank": corrado_rank(panel, forward_periods=_H).p_value,
                "event_hit_rate": event_hit_rate(panel, forward_periods=_H).p_value,
            }

    def test_rejection_rate_is_near_nominal_on_a_pure_drift_null(self):
        reps = 40
        rejections = {k: 0 for k in ("caar", "bmp_z", "corrado_rank", "event_hit_rate")}
        for seed in range(reps):
            for name, p in self._p_values(_drift_panel(seed)).items():
                if p is not None and p < 0.05:
                    rejections[name] += 1
        # Measured at 60 reps on a 20-asset panel: 6.7 / 3.3 / 3.3 / 5.0%.
        # Without the model the same panels gave 13.3 / 50.0 / 5.0 / 31.7%.
        for name, count in rejections.items():
            assert count / reps <= 0.20, (name, count, reps)

    def test_supplying_the_raw_return_as_abnormal_reproduces_the_old_failure(self):
        # The guard on the guard: if this panel did NOT over-reject without a
        # model, the test above would pass vacuously. Declaring the raw return
        # as the abnormal return is exactly the pre-fix computation.
        reps = 12
        rejected = 0
        for seed in range(reps):
            # 20 assets: the drift signal scales with sqrt(N), and this is the
            # panel width the audit measured 50% rejection on.
            panel = _drift_panel(seed, n_assets=20).with_columns(
                pl.col("forward_return").alias("abnormal_return")
            )
            p = self._p_values(panel)["bmp_z"]
            if p is not None and p < 0.05:
                rejected += 1
        assert rejected / reps >= 0.33
