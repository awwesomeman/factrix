"""Tests for factrix.metrics.clustering_hhi."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
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
