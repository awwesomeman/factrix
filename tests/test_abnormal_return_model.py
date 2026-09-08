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
from factrix._errors import UserInputError
from factrix.metrics._helpers import _attach_abnormal_return
from factrix.metrics.caar import bmp_z, caar, compute_caar
from factrix.metrics.corrado_rank import corrado_rank
from factrix.metrics.event_quality import (
    event_hit_rate,
    event_ic,
    event_skewness,
    profit_factor,
)

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


def _supplied_abnormal_panel() -> pl.DataFrame:
    """Panel whose only usable event-return source is ``abnormal_return``."""
    rows = []
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(120)]
    for asset_index, asset_id in enumerate(("A", "B")):
        for date_index, date in enumerate(dates):
            is_event = date_index >= 20 and date_index % 3 == 0
            magnitude = 1.0 + (date_index % 5) / 4.0
            direction = -1.0 if (date_index + asset_index) % 4 == 0 else 1.0
            abnormal_return = (
                0.003 * np.sin(date_index / 3.0 + asset_index)
                + 0.001 * ((date_index % 7) - 3)
            )
            rows.append(
                {
                    "date": date,
                    "asset_id": asset_id,
                    "factor": direction * magnitude if is_event else 0.0,
                    "abnormal_return": float(abnormal_return),
                }
            )
    return pl.DataFrame(rows)


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
            panel, estimation_window=60, overlap_periods=_H
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
            panel, estimation_window=60, overlap_periods=_H
        )
        got = out["_abnormal_return"].to_numpy()
        assert got[100] == pytest.approx(1.0)  # its own mean is still zero
        # It only starts contaminating estimates h rows later.
        assert got[104] == pytest.approx(0.0)
        assert got[105] == pytest.approx(-1.0 / 60)

    def test_price_path_uses_the_bar_window_ending_at_the_event(self):
        rng = np.random.default_rng(1)
        n = 200
        bars = rng.normal(0.001, 0.01, n)
        prices = 100.0 * np.cumprod(1.0 + bars)
        fwd = rng.normal(0.0, 0.01, n)
        panel = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "price": prices,
                "forward_return": fwd,
            }
        )
        out, diagnostics = _attach_abnormal_return(
            panel, estimation_window=60, overlap_periods=_H
        )
        assert diagnostics["estimation_window_source"] == "price"
        assert diagnostics["estimation_window_lag"] == 0
        got = out["_abnormal_return"].to_numpy()
        bar_ret = prices[1:] / prices[:-1] - 1.0  # bar_ret[i] is the bar (i, i+1]
        t = 150
        # The 60 one-bar returns ending at t: bars (t-60, t-59] .. (t-1, t].
        expected = fwd[t] - bar_ret[t - 60 : t].mean()
        assert got[t] == pytest.approx(expected)
        # Without a price the mean comes from lagged forward-return rows.
        out_rows, diag_rows = _attach_abnormal_return(
            panel.drop("price"), estimation_window=60, overlap_periods=_H
        )
        assert diag_rows["estimation_window_source"] == "forward_return"
        assert diag_rows["estimation_window_lag"] == _H
        assert out_rows["_abnormal_return"][t] == pytest.approx(
            fwd[t] - fwd[t - _H - 59 : t - _H + 1].mean()
        )

    def test_a_supplied_abnormal_return_column_is_honoured(self):
        panel = _drift_panel(0).with_columns(
            (pl.col("forward_return") - 0.5).alias("abnormal_return")
        )
        out, diagnostics = _attach_abnormal_return(panel, overlap_periods=_H)
        assert diagnostics["abnormal_return_model"] == "market_adjusted_supplied"
        assert out["_abnormal_return"].to_numpy() == pytest.approx(
            out["abnormal_return"].to_numpy()
        )

    def test_missing_return_sources_raise_the_project_input_error(self):
        panel = _supplied_abnormal_panel().drop("abnormal_return")
        with pytest.raises(UserInputError, match="return_col") as excinfo:
            _attach_abnormal_return(panel, func_name="event_hit_rate")
        assert excinfo.value.func_name == "event_hit_rate"
        assert excinfo.value.docs_url.endswith("/api/data-schema")

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
            panel, estimation_window=60, overlap_periods=_H
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
                    compute_caar(panel, overlap_periods=_H), overlap_periods=_H
                ).p_value,
                "bmp_z": bmp_z(panel, overlap_periods=_H).p_value,
                "corrado_rank": corrado_rank(panel, overlap_periods=_H).p_value,
                "event_hit_rate": event_hit_rate(panel, overlap_periods=_H).p_value,
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


class TestSuppliedAbnormalReturnContract:
    """Every event consumer must honour the source selected by the model."""

    @staticmethod
    def _quality_results(panel: pl.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {
                "corrado_rank": corrado_rank(panel, overlap_periods=1),
                "event_hit_rate": event_hit_rate(panel, overlap_periods=1),
                "event_ic": event_ic(panel, overlap_periods=1),
                "profit_factor": profit_factor(panel, overlap_periods=1),
                "event_skewness": event_skewness(panel, overlap_periods=1),
            }

    def test_abnormal_only_panel_runs_without_naming_a_raw_return(self):
        panel = _supplied_abnormal_panel()
        series = compute_caar(panel, overlap_periods=1)
        assert series.height > 0
        assert set(series["abnormal_return_model"]) == {"market_adjusted_supplied"}

        for result in self._quality_results(panel).values():
            assert result.n_obs > 0
            assert result.metadata["abnormal_return_model"] == (
                "market_adjusted_supplied"
            )

    def test_non_finite_unused_raw_return_does_not_discard_valid_events(self):
        panel = _supplied_abnormal_panel()
        dirty = panel.with_columns(
            pl.when(pl.col("factor") != 0)
            .then(float("inf"))
            .otherwise(float("nan"))
            .alias("forward_return")
        )

        clean_series = compute_caar(panel, overlap_periods=1)
        dirty_series = compute_caar(dirty, overlap_periods=1)
        assert dirty_series["caar"].to_list() == pytest.approx(
            clean_series["caar"].to_list()
        )
        assert set(dirty_series["n_events_dropped_non_finite"]) == {0}

        clean_results = self._quality_results(panel)
        dirty_results = self._quality_results(dirty)
        for name, clean in clean_results.items():
            dirty_result = dirty_results[name]
            assert dirty_result.n_obs == clean.n_obs
            assert dirty_result.value == pytest.approx(clean.value)
            assert dirty_result.metadata["n_events_dropped_non_finite"] == 0

    @pytest.mark.parametrize(
        "metric",
        [
            compute_caar,
            corrado_rank,
            event_hit_rate,
            event_ic,
            profit_factor,
            event_skewness,
        ],
    )
    def test_public_consumers_raise_user_input_error_without_a_return_source(
        self, metric
    ):
        panel = _supplied_abnormal_panel().drop("abnormal_return")
        with pytest.raises(UserInputError, match="return_col") as excinfo:
            metric(panel, overlap_periods=1)
        assert excinfo.value.func_name == metric.__name__
