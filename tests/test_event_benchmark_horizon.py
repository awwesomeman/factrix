"""The excess benchmark matches the horizon it is subtracted from (#1052)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from factrix.metrics.event_horizon import event_around_return

OFFSETS = [-3, -1, 1, 6, 12, 24]


def _drift_only_panel(
    *,
    n_assets: int = 8,
    n_dates: int = 200,
    growth: float = 1.004,
    event_every: int = 17,
    first_event: int = 40,
) -> pl.DataFrame:
    """Every asset compounds at exactly ``growth`` per period, forever.

    No event carries information here: the price path is the same whether
    or not the factor fired. Every excess return is therefore zero by
    construction, at every offset, and any residue is the benchmark
    failing to match the horizon it was subtracted from.
    """
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(n_dates)]
    rows: list[dict[str, object]] = []
    for asset_index in range(n_assets):
        for index, current_date in enumerate(dates):
            is_event = index >= first_event and (index - asset_index) % event_every == 0
            rows.append(
                {
                    "date": current_date,
                    "asset_id": f"A{asset_index}",
                    "factor": 1.0 if is_event else 0.0,
                    "price": 100.0 * growth**index,
                }
            )
    return pl.DataFrame(rows)


def test_drift_only_panel_has_no_excess_at_any_offset() -> None:
    """A pure-drift panel is the null: every offset must price to zero.

    Offset ``k > 0`` is a ``k``-period return from ``t+1`` to ``t+1+k``,
    so subtracting one single-period baseline leaves roughly ``k - 1``
    periods of drift in the answer and reads as event alpha.
    """
    result = event_around_return(_drift_only_panel(), offsets=OFFSETS)

    per_offset = result.metadata["per_offset"]
    residues = {
        offset: per_offset[offset]["mean"]
        for offset in OFFSETS
        if per_offset[offset].get("mean") is not None
    }
    assert set(residues) == set(OFFSETS), residues
    for offset, mean in residues.items():
        assert mean == pytest.approx(0.0, abs=1e-12), (
            f"offset {offset} keeps {mean:.6g} of pure drift"
        )


def test_baseline_weights_assets_equally_on_a_ragged_panel() -> None:
    """The docstring's per-asset baseline is what the code computes.

    Pooling every bar observation lets an asset with a longer history
    outvote a short one. Two assets with different drifts and very
    different history lengths pin the difference: the equal-weighted
    baseline is the mean of the two per-asset drifts, while the pooled
    one sits near the long asset's.
    """
    long_growth, short_growth = 1.006, 1.001
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(120)]
    rows: list[dict[str, object]] = []
    for index, current_date in enumerate(dates):
        rows.append(
            {
                "date": current_date,
                "asset_id": "LONG",
                "factor": 1.0 if index in (60, 80, 100) else 0.0,
                "price": 100.0 * long_growth**index,
            }
        )
        if index >= 100:
            rows.append(
                {
                    "date": current_date,
                    "asset_id": "SHORT",
                    "factor": 1.0 if index == 110 else 0.0,
                    "price": 50.0 * short_growth ** (index - 100),
                }
            )
    result = event_around_return(pl.DataFrame(rows), offsets=[-1, 1])

    equal_weighted = (long_growth - 1.0 + short_growth - 1.0) / 2.0
    assert result.metadata["baseline_bar_return"] == pytest.approx(
        equal_weighted, rel=1e-9
    )
    assert result.metadata["n_assets_in_baseline"] == 2


def test_per_offset_reports_the_benchmark_it_subtracted() -> None:
    """Each offset publishes the benchmark actually applied to it."""
    result = event_around_return(_drift_only_panel(), offsets=[-1, 6])

    baseline = result.metadata["baseline_bar_return"]
    per_offset = result.metadata["per_offset"]
    assert per_offset[-1]["benchmark"] == pytest.approx(baseline, rel=1e-12)
    assert per_offset[6]["benchmark"] == pytest.approx(
        (1.0 + baseline) ** 6 - 1.0, rel=1e-12
    )
