"""Event windows, lags and offsets are counted on the panel's period grid.

A polars ``rolling_*`` window or ``shift`` counts rows within a frame; the
event family needs a count of periods on the panel's own distinct-date grid
(CLAUDE.md, "Period grid, not calendar"). The two agree only on a dense panel.
On a ragged one — an asset missing periods the other names have — the row form
let a 30-period estimation window reach 50 grid periods back and let a
``k``-offset return step over the hole as though it were one period.

Two properties pin the fix: on a dense panel nothing moves at all (the golden
tests below reproduce the old per-asset row arithmetic exactly), and on the
ragged hand case the window spans exactly the requested number of grid periods
with the missing periods counting as missing observations inside it.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics._helpers import _attach_abnormal_return, _densify_on_period_grid
from factrix.metrics._primitives import compute_event_returns

_D0 = date(2020, 1, 1)


def _panel(
    *,
    n_dates: int = 120,
    gap: tuple[int, int] | None = None,
    gap_asset: str = "B",
    event_at: int = 100,
    seed: int = 0,
) -> pl.DataFrame:
    """Two assets on a ``n_dates``-period grid, one event, optional hole.

    ``gap`` drops ``range(*gap)`` from ``gap_asset`` only, so the panel grid
    keeps every period and exactly one asset is ragged.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for aid in ("A", "B"):
        prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n_dates))
        for t in range(n_dates):
            if gap is not None and aid == gap_asset and gap[0] <= t < gap[1]:
                continue
            rows.append(
                {
                    "date": _D0 + timedelta(days=t),
                    "asset_id": aid,
                    "price": float(prices[t]),
                    "factor": 1.0 if t == event_at else 0.0,
                    "forward_return": float(rng.normal(0.0, 0.01)),
                }
            )
    return pl.DataFrame(rows)


def _dense_event_panel() -> pl.DataFrame:
    """A dense synthetic event panel carrying the forward return."""
    return fx.preprocess.compute_forward_return(
        fx.datasets.make_event_panel(n_assets=10, n_dates=200, event_rate=0.05, rng=7),
        forward_periods=5,
    )


class TestDensifyHelper:
    def test_dense_panel_is_a_no_op(self) -> None:
        panel = _panel()
        dense, densified = _densify_on_period_grid(panel)
        assert not densified
        assert dense.equals(panel.sort(["asset_id", "date"]))

    def test_ragged_panel_gains_the_missing_periods_as_nulls(self) -> None:
        panel = _panel(gap=(60, 80))
        dense, densified = _densify_on_period_grid(panel)
        assert densified
        assert dense.height == 2 * 120
        holes = dense.filter(pl.col("asset_id") == "B").filter(
            pl.col("price").is_null()
        )
        assert holes.height == 20
        # The panel grid itself is untouched — no date was invented.
        assert dense["date"].n_unique() == 120


class TestGoldenDensePanel:
    """A dense panel must be bit-identical to the pre-grid row arithmetic."""

    def test_abnormal_return_matches_per_asset_rolling(self) -> None:
        panel = _dense_event_panel()
        out, _ = _attach_abnormal_return(panel, estimation_window=30, overlap_periods=5)
        # The old row form: rolling over each asset's own rows.
        expected = panel.sort(["asset_id", "date"]).with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1).alias(
                "_bar"
            )
        )
        expected = expected.with_columns(
            (
                pl.col("forward_return")
                - pl.when(pl.col("_bar").is_finite())
                .then(pl.col("_bar"))
                .rolling_mean(window_size=30, min_samples=20)
                .over("asset_id")
            ).alias("_golden")
        )
        joined = out.join(
            expected.select(["asset_id", "date", "_golden"]),
            on=["asset_id", "date"],
            how="left",
        )
        assert joined.select(
            (pl.col("_abnormal_return") - pl.col("_golden")).abs().max()
        ).item() == pytest.approx(0.0, abs=0.0)

    def test_event_returns_match_per_asset_row_offsets(self) -> None:
        panel = _dense_event_panel()
        offsets = [-6, -1, 1, 6]
        got = compute_event_returns(panel, offsets=offsets)
        # The old row form, recomputed here as the golden reference.
        rows: list[dict] = []
        sorted_df = panel.sort(["asset_id", "date"])
        for aid, adf in sorted_df.group_by("asset_id", maintain_order=True):
            prices = adf["price"].to_numpy()
            idx_of = {d: i for i, d in enumerate(adf["date"].to_list())}
            for row in adf.filter(pl.col("factor") != 0).iter_rows(named=True):
                i = idx_of[row["date"]]
                s = float(np.sign(row["factor"]))
                for k in offsets:
                    if k > 0:
                        if i + 1 + k >= len(prices):
                            continue
                        r = prices[i + 1 + k] / prices[i + 1] - 1
                    else:
                        if i + k - 1 < 0 or i + k >= len(prices):
                            continue
                        r = prices[i + k] / prices[i + k - 1] - 1
                    rows.append(
                        {
                            "offset": k,
                            "date": row["date"],
                            "asset_id": aid[0],
                            "signed_return": float(s * r),
                        }
                    )
        golden = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(panel.schema["date"])
        )
        assert got.height == golden.height
        merged = got.join(
            golden, on=["offset", "date", "asset_id"], suffix="_golden", how="inner"
        )
        assert merged.height == got.height
        assert merged.select(
            (pl.col("signed_return") - pl.col("signed_return_golden")).abs().max()
        ).item() == pytest.approx(0.0, abs=0.0)


class TestRaggedHandCase:
    """2 assets, a 120-period grid, B missing periods 60-79, window 30, h = 5."""

    def test_estimation_window_spans_thirty_grid_periods(self) -> None:
        panel = _panel(gap=(60, 80))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, _ = _attach_abnormal_return(
                panel, estimation_window=30, overlap_periods=5
            )
        event = out.filter((pl.col("asset_id") == "B") & (pl.col("factor") != 0))
        assert event.height == 1
        got = float(event["_abnormal_return"][0])

        b = panel.filter(pl.col("asset_id") == "B").sort("date")
        prices = dict(zip(b["date"].to_list(), b["price"].to_list(), strict=True))

        def bar(t: int) -> float | None:
            here = prices.get(_D0 + timedelta(days=t))
            prev = prices.get(_D0 + timedelta(days=t - 1))
            return None if here is None or prev is None else here / prev - 1.0

        # The window is grid periods 71..100 — exactly 30 grid periods ending
        # at the event. Periods 71..79 fall in B's hole and count as missing,
        # so 20 bar returns contribute (period 80 has no predecessor on the
        # grid); the window does NOT reach further back to find 30 of them.
        grid_window = [bar(t) for t in range(71, 101)]
        assert len(grid_window) == 30
        contributing = [v for v in grid_window if v is not None]
        assert len(contributing) == 20
        expected = float(event["forward_return"][0]) - float(np.mean(contributing))
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15)

        # The row form took B's 30 preceding *rows*: grid periods 51..59 and
        # 80..100 — a "30-period" window spanning 50 grid periods, the hand
        # case from the audit. It gives a different number.
        row_periods = list(range(51, 60)) + list(range(80, 101))
        assert len(row_periods) == 30
        assert row_periods[-1] - row_periods[0] + 1 == 50
        row_window = [v for v in (bar(t) for t in row_periods) if v is not None]
        row_form = float(event["forward_return"][0]) - float(np.mean(row_window))
        assert got != pytest.approx(row_form, rel=1e-9)

    def test_window_mostly_inside_the_hole_falls_below_min_samples(self) -> None:
        # Event at period 85: the 30-period window 56..85 mostly covers the
        # hole, leaving 9 contributing periods — below min_samples (20), so the
        # abnormal return is null and the event is dropped downstream. The row
        # form would have quietly filled the window from grid period 36.
        panel = _panel(gap=(60, 80), event_at=85)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, _ = _attach_abnormal_return(
                panel, estimation_window=30, overlap_periods=5
            )
        event = out.filter((pl.col("asset_id") == "B") & (pl.col("factor") != 0))
        assert event.height == 1
        assert event["_abnormal_return"][0] is None

    def test_ragged_grid_warning_fires(self) -> None:
        panel = _panel(gap=(60, 80))
        with pytest.warns(UserWarning, match=WarningCode.RAGGED_PERIOD_GRID.value):
            _attach_abnormal_return(panel, estimation_window=30, overlap_periods=5)

    def test_dense_panel_does_not_warn(self) -> None:
        panel = _panel()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _attach_abnormal_return(panel, estimation_window=30, overlap_periods=5)


class TestRaggedOffsets:
    def test_offset_steps_grid_periods_not_rows(self) -> None:
        # B's event sits at period 55; a +6 offset exits at period 62, inside
        # the hole, so it has no return at all. The row form would have paired
        # it with period 82 — 27 grid periods out, reported as 6.
        panel = _panel(gap=(60, 80), event_at=55)
        out = compute_event_returns(panel, offsets=[6])
        assert out.filter(pl.col("asset_id") == "B").height == 0
        assert out.filter(pl.col("asset_id") == "A").height == 1

    def test_offset_clear_of_the_hole_is_unaffected(self) -> None:
        panel = _panel(gap=(60, 80), event_at=100)
        out = compute_event_returns(panel, offsets=[6])
        b = out.filter(pl.col("asset_id") == "B")
        assert b.height == 1
        prices = dict(
            zip(
                panel.filter(pl.col("asset_id") == "B")["date"].to_list(),
                panel.filter(pl.col("asset_id") == "B")["price"].to_list(),
                strict=True,
            )
        )
        expected = (
            prices[_D0 + timedelta(days=107)] / prices[_D0 + timedelta(days=101)] - 1.0
        )
        assert float(b["signed_return"][0]) == pytest.approx(expected, rel=1e-12)
