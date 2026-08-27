"""Event-axis non-overlap discipline shared by the event significance tests.

Every test in the event battery whose p-value assumes independent per-event
draws (``caar``, ``bmp_z``, ``corrado_rank``, ``event_hit_rate``, ``event_ic``,
``event_skewness``) strides its own event axis at ``overlap_periods``, per
asset, before it tests anything. Without it a trigger that fires in bursts on
one name hands each test the same shock several times over: with a single
asset there is no cross-section for the Kolari-Pynnönen adjustment or the
per-period collapse to work on, so time is the only clustering axis and the
stride is the whole discipline.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics._helpers import _sample_events_non_overlapping
from factrix.metrics.caar import bmp_z, caar, compute_caar
from factrix.metrics.corrado_rank import corrado_rank
from factrix.metrics.event_quality import event_hit_rate, event_ic, event_skewness

_H = 5


def _burst_panel(
    seed: int = 0,
    n_dates: int = 900,
    n_assets: int = 1,
    burst: int = 4,
    every: int = 40,
    drift: float = 0.0,
) -> pl.DataFrame:
    """Panel whose trigger fires ``burst`` periods in a row, every ``every``.

    Consecutive events inside a burst sit one period apart, well inside the
    ``_H``-period forward-return window, so the spacing pass must keep exactly
    one event per burst on each asset.
    """
    rng = np.random.default_rng(seed)
    starts = set(range(60, n_dates - 60, every))
    rows = []
    for a in range(n_assets):
        rets = rng.normal(drift, 0.02, size=n_dates)
        # A real price path: bmp_z needs a non-degenerate estimation-window
        # volatility, which a constant price cannot give it.
        prices = 100.0 * np.cumprod(1.0 + rets)
        for d in range(n_dates):
            in_burst = any(s <= d < s + burst for s in starts)
            rows.append(
                {
                    "date": datetime(2020, 1, 1) + timedelta(days=d),
                    "asset_id": f"A{a}",
                    "factor": 1.0 if in_burst else 0.0,
                    "forward_return": float(rets[d]),
                    "price": float(prices[d]),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestSampleEventsNonOverlapping:
    """The shared helper: per asset, gap >= overlap_periods, first kept."""

    @staticmethod
    def _events(ordinals: dict[str, list[int]], n_cal: int = 60) -> pl.DataFrame:
        base = datetime(2020, 1, 1)
        rows = [
            {"date": base + timedelta(days=o), "asset_id": a, "factor": 1.0}
            for a, ords in ordinals.items()
            for o in ords
        ]
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    @staticmethod
    def _calendar(n_cal: int = 60) -> pl.Series:
        base = datetime(2020, 1, 1)
        return pl.Series(
            "date",
            [base + timedelta(days=i) for i in range(n_cal)],
            dtype=pl.Datetime("ms"),
        )

    def test_keeps_first_then_every_gap_of_at_least_h(self):
        events = self._events({"A": [0, 1, 2, 5, 6, 12]})
        kept = _sample_events_non_overlapping(events, 5, grid_dates=self._calendar())
        got = [(d - datetime(2020, 1, 1)).days for d in kept["date"].to_list()]
        assert got == [0, 5, 12]

    def test_spacing_is_per_asset(self):
        # B's events are unrelated to A's: one asset's burst must not consume
        # another asset's slot.
        events = self._events({"A": [0, 1, 10], "B": [0, 1, 10]})
        kept = _sample_events_non_overlapping(events, 5, grid_dates=self._calendar())
        assert kept.group_by("asset_id").len().sort("asset_id")["len"].to_list() == [
            2,
            2,
        ]

    def test_horizon_one_is_a_no_op(self):
        events = self._events({"A": [0, 1, 2]})
        kept = _sample_events_non_overlapping(events, 1, grid_dates=self._calendar())
        assert kept.height == events.height

    def test_gap_is_measured_on_the_full_calendar_not_the_event_index(self):
        # Three events 20 calendar steps apart are all independent even though
        # they are adjacent *rows* of the event frame.
        events = self._events({"A": [0, 20, 40]})
        kept = _sample_events_non_overlapping(events, 5, grid_dates=self._calendar())
        assert kept.height == 3


class TestBurstsAreThinnedByEveryEventTest:
    """Each test's reported sample counts non-overlapping events only."""

    @pytest.mark.parametrize(
        "metric",
        [bmp_z, corrado_rank, event_hit_rate, event_skewness],
    )
    def test_sample_drops_to_one_event_per_burst(self, metric):
        panel = _burst_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = metric(panel, overlap_periods=_H)
        n_bursts = len(range(60, 900 - 60, 40))
        assert result.metadata["n_events_sampled"] == n_bursts
        assert result.metadata["n_events_overlapping"] == n_bursts * 3
        assert result.n_obs == n_bursts

    def test_caar_reports_the_same_removal(self):
        panel = _burst_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = caar(compute_caar(panel), overlap_periods=_H)
        n_bursts = len(range(60, 900 - 60, 40))
        assert result.metadata["n_events_sampled"] == n_bursts
        assert result.n_obs == n_bursts

    def test_event_ic_thins_a_continuous_magnitude_signal(self):
        panel = _burst_panel().with_columns(
            pl.when(pl.col("factor") != 0)
            .then(pl.col("forward_return").abs() + 0.5)
            .otherwise(0.0)
            .alias("factor")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = event_ic(panel, overlap_periods=_H)
        assert result.metadata["n_events_sampled"] == result.n_obs
        assert result.metadata["n_events_overlapping"] > 0


class TestEventWindowOverlapCode:
    """``EVENT_WINDOW_OVERLAP`` has emitters, one per metric, and stays quiet
    when the events are genuinely spaced."""

    @pytest.mark.parametrize(
        "metric",
        [bmp_z, corrado_rank, event_hit_rate, event_skewness],
    )
    def test_fires_once_when_the_spacing_pass_removed_events(self, metric):
        panel = _burst_panel()
        with pytest.warns(UserWarning, match="forward-return windows overlapped"):
            result = metric(panel, overlap_periods=_H)
        assert result.warning_codes.count(WarningCode.EVENT_WINDOW_OVERLAP.value) == 1

    @pytest.mark.parametrize(
        "metric",
        [bmp_z, corrado_rank, event_hit_rate, event_skewness],
    )
    def test_silent_when_every_event_clears_the_horizon(self, metric):
        # burst=1 with a 40-period cadence: no two events on one asset sit
        # inside a 5-period window, so nothing is removed.
        panel = _burst_panel(burst=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = metric(panel, overlap_periods=_H)
        assert WarningCode.EVENT_WINDOW_OVERLAP.value not in result.warning_codes
        assert result.metadata["n_events_overlapping"] == 0

    def test_declaring_the_code_stops_the_echo_but_keeps_the_record(self):
        panel = _burst_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = event_hit_rate(
                panel,
                overlap_periods=_H,
                expected_warnings=("event_window_overlap", "few_events"),
            )
        assert WarningCode.EVENT_WINDOW_OVERLAP.value in result.warning_codes


class TestNullSizeUnderClusteredEvents:
    """The point of the stride: a bursty trigger on a true null must not read
    as significant. Before the spacing pass these three rejected 18-35% of
    null draws at a nominal 5% (single asset, T=2500, 5% event rate).
    """

    @pytest.mark.parametrize(
        "metric", [bmp_z, corrado_rank, event_hit_rate], ids=lambda m: m.__name__
    )
    def test_rejection_rate_is_near_nominal(self, metric):
        reps = 60
        rej = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                r = metric(_burst_panel(seed=seed), overlap_periods=_H)
                if r.p_value is not None and r.p_value < 0.05:
                    rej += 1
        # 60 reps at a true 5% has SE ~2.8pp; the band catches a return to the
        # pre-fix regime (18-35%), not a few points of drift.
        assert rej / reps <= 0.15


class TestEventBatteryReportsOneAxisToken:
    """One quantity, one token. ``caar`` and ``corrado_rank`` used to report
    ``"periods"`` while ``bmp_z`` / ``event_hit_rate`` / ``mfe_mae`` reported
    ``"events"`` — on single-asset data these count the same thing, so
    stacking ``to_frame()`` across the battery produced two labels for one
    column. Every member now reports the event axis; the period-level counts
    that differ between them stay in metadata under their own names.
    """

    def test_every_event_metric_reports_the_event_axis(self):
        from factrix.metrics.event_horizon import event_around_return
        from factrix.metrics.event_quality import (
            profit_factor,
            signal_density,
        )
        from factrix.metrics.mfe_mae import compute_mfe_mae, mfe_mae

        panel = _burst_panel(burst=1, n_assets=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = {
                "caar": caar(compute_caar(panel), overlap_periods=_H),
                "bmp_z": bmp_z(panel, overlap_periods=_H),
                "corrado_rank": corrado_rank(panel, overlap_periods=_H),
                "event_hit_rate": event_hit_rate(panel, overlap_periods=_H),
                "event_skewness": event_skewness(panel, overlap_periods=_H),
                "profit_factor": profit_factor(panel),
                "signal_density": signal_density(panel),
                "mfe_mae": mfe_mae(compute_mfe_mae(panel, window=10)),
                "event_around_return": event_around_return(panel),
            }
        axes = {name: r.n_obs_axis for name, r in results.items()}
        assert set(axes.values()) == {"events"}, axes
        assert all(r.n_obs is not None for r in results.values()), axes
