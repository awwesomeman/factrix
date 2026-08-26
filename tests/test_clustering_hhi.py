"""Tests for factrix.metrics.clustering_hhi."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from factrix.metrics._helpers import _estimate_within_date_icc
from factrix.metrics.clustering_hhi import clustering_hhi


def _panel(rows) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _event_rows(n_dates: int, events_per_date: int, filler: float | None = 0.0):
    """One row per (date, asset); the first ``events_per_date`` are events."""
    rows = []
    for d in range(n_dates):
        day = datetime(2024, 1, 1) + timedelta(days=d)
        for a in range(10):
            rows.append(
                {
                    "date": day,
                    "asset_id": f"A{a}",
                    "factor": 1.0 if a < events_per_date else filler,
                }
            )
    return rows


class TestEventSelection:
    def test_uniform_clustering_is_one_over_d(self):
        result = clustering_hhi(_panel(_event_rows(20, 2)))
        assert result.metadata["n_events"] == 40
        assert result.metadata["n_event_periods"] == 20
        assert result.value == pytest.approx(1 / 20)

    def test_nan_factor_is_not_an_event(self):
        """``factor != 0`` is True for NaN, so NaN rows used to be counted.

        Every non-event row here is NaN: the old filter kept all 200 rows as
        "events", flattening the histogram to a uniform 10-per-date and
        understating the concentration.
        """
        nan_panel = _panel(_event_rows(20, 2, filler=float("nan")))
        zero_panel = _panel(_event_rows(20, 2, filler=0.0))

        nan_result = clustering_hhi(nan_panel)
        zero_result = clustering_hhi(zero_panel)

        assert nan_result.metadata["n_events"] == 40
        assert nan_result.n_obs == 40
        assert nan_result.value == pytest.approx(zero_result.value)

    def test_null_factor_is_not_an_event(self):
        null_panel = _panel(_event_rows(20, 2, filler=None)).with_columns(
            pl.col("factor").cast(pl.Float64)
        )
        assert clustering_hhi(null_panel).metadata["n_events"] == 40

    def test_nan_rows_do_not_create_phantom_event_dates(self):
        """Dates carrying only NaN factors must not enter the histogram."""
        rows = _event_rows(20, 2)
        for d in range(20, 40):
            day = datetime(2024, 1, 1) + timedelta(days=d)
            for a in range(10):
                rows.append({"date": day, "asset_id": f"A{a}", "factor": float("nan")})
        result = clustering_hhi(_panel(rows))
        assert result.metadata["n_event_periods"] == 20
        assert result.value == pytest.approx(1 / 20)


class TestTheAxesHhiCannotSee:
    """HHI is invariant to cross-sectional clustering and to temporal bursting.
    Both are now measured beside it, because the documented "HHI >= 0.3 means
    clustered" rule was unreachable and pointed users away from the regime the
    Kolari-Pynnonen adjustment exists for.
    """

    @staticmethod
    def _panel(n_assets: int, event_days: list[int], n_dates: int = 200):
        rows = []
        for d in range(n_dates):
            for a in range(n_assets):
                rows.append(
                    {
                        "date": datetime(2020, 1, 1) + timedelta(days=d),
                        "asset_id": f"A{a}",
                        "factor": 1.0 if d in event_days else 0.0,
                        "forward_return": 0.01,
                    }
                )
        return pl.DataFrame(rows)

    def test_hhi_is_blind_to_cross_sectional_clustering(self):
        days = list(range(10, 90, 2))
        one = clustering_hhi(self._panel(1, days))
        many = clustering_hhi(self._panel(20, days))
        # Same HHI, same normalized HHI — 20x the same-period clustering.
        assert one.value == pytest.approx(many.value)
        assert one.metadata["hhi_normalized"] == pytest.approx(
            many.metadata["hhi_normalized"]
        )
        assert many.metadata["hhi_normalized"] == pytest.approx(0.0, abs=1e-12)
        # The companion measure is not blind.
        assert one.metadata["events_per_period_mean"] == pytest.approx(1.0)
        assert many.metadata["events_per_period_mean"] == pytest.approx(20.0)
        assert many.metadata["max_events_per_period"] == 20

    def test_events_per_period_mean_feeds_the_kp_adjustment(self):
        # It is the same n_eff the deflator consumes, so the diagnostic and the
        # correction read one number.
        panel = self._panel(20, list(range(10, 90, 2)))
        events = panel.filter(pl.col("factor") != 0).select(
            "date", pl.col("factor").alias("_v")
        )
        _, n_eff, _ = _estimate_within_date_icc(events, "_v")
        assert clustering_hhi(panel).metadata[
            "events_per_period_mean"
        ] == pytest.approx(n_eff)

    def test_temporal_bursting_is_measured(self):
        # 30 events on consecutive periods vs 30 spread far apart: identical
        # HHI, opposite burst shares.
        burst = clustering_hhi(self._panel(1, list(range(20, 50))), cluster_window=3)
        spread = clustering_hhi(
            self._panel(1, list(range(20, 200, 6))), cluster_window=3
        )
        assert burst.value == pytest.approx(spread.value, rel=0.2)
        assert burst.metadata["share_events_in_bursts"] > 0.9
        assert spread.metadata["share_events_in_bursts"] == pytest.approx(0.0)

    def test_cluster_window_is_no_longer_inert(self):
        panel = self._panel(1, list(range(20, 200, 5)))
        narrow = clustering_hhi(panel, cluster_window=3)
        wide = clustering_hhi(panel, cluster_window=10)
        assert narrow.metadata["share_events_in_bursts"] == pytest.approx(0.0)
        assert wide.metadata["share_events_in_bursts"] > 0.9
