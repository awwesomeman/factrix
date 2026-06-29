"""Aggregate drop-rate warning — Phase 2 (SERIES→SCALAR null-drops).

Period-axis SERIES→SCALAR consumers drop null / non-finite observations from
their value series before collapsing to a scalar, shrinking the effective
sample. Each now records the same canonical five-key schema (via
``_surface_null_drop``) into ``MetricResult.metadata`` and emits one aggregate
``UserWarning`` + ``WarningCode.EXCESSIVE_PERIOD_DROPS`` when the null-drop rate
clears ``DROP_RATE_WARN_THRESHOLD``.

Scope: the period-axis null-droppers — ``positive_rate``, ``oos_decay``,
``ic_trend`` (IC series) and ``quantile_spread`` / ``quantile_spread_vw`` /
``k_spread`` (spread series). Asset-axis (``common_beta``) and event-axis
(``caar`` / ``mfe_mae``) consumers are out of the ``n_periods_*`` schema.
"""

from __future__ import annotations

import datetime as dt
import warnings

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics._helpers import DROP_RATE_WARN_THRESHOLD, DROP_STAT_KEYS
from factrix.metrics.k_spread import k_spread
from factrix.metrics.oos_decay import oos_decay
from factrix.metrics.positive_rate import positive_rate
from factrix.metrics.quantile import quantile_spread, quantile_spread_vw
from factrix.metrics.trend import ic_trend

EXCESSIVE = WarningCode.EXCESSIVE_PERIOD_DROPS.value


def _ic_series(*, n: int = 150, null_every: int | None = 3, seed: int = 0):
    """A ``(date, value)`` IC-like series; every ``null_every``-th value null."""
    rng = np.random.default_rng(seed)
    base = dt.date(2020, 1, 1)
    rows = []
    for d in range(n):
        v: float | None = float(rng.normal())
        if null_every is not None and d % null_every == 0:
            v = None
        rows.append({"date": base + dt.timedelta(days=d), "value": v})
    return pl.DataFrame(rows)


def _spread_panel(*, n_dates: int = 200, thin: int = 3, full: int = 50, seed: int = 0):
    """Panel where even dates carry ``thin`` assets (< n_groups → null spread)."""
    rng = np.random.default_rng(seed)
    base = dt.date(2020, 1, 1)
    rows = []
    for d in range(n_dates):
        n = thin if d % 2 == 0 else full
        for a in range(n):
            rows.append(
                {
                    "date": base + dt.timedelta(days=d),
                    "asset_id": f"A{a:03d}",
                    "factor": float(rng.normal()),
                    "forward_return": float(rng.normal()),
                    "market_cap": float(abs(rng.normal()) + 1.0),
                }
            )
    return pl.DataFrame(rows)


class TestICSeriesTools:
    @pytest.mark.parametrize(
        ("fn", "name"),
        [
            (positive_rate, "positive_rate"),
            (oos_decay, "oos_decay"),
            (ic_trend, "ic_trend"),
        ],
    )
    def test_high_null_drop_warns(self, fn, name):
        series = _ic_series(null_every=3)  # ~1/3 null
        with pytest.warns(UserWarning, match="of periods dropped"):
            result = (
                fn(series) if name != "positive_rate" else fn(series, forward_periods=1)
            )
        assert EXCESSIVE in result.warning_codes
        assert result.metadata["drop_rate"] > DROP_RATE_WARN_THRESHOLD
        assert set(DROP_STAT_KEYS) <= set(result.metadata)

    @pytest.mark.parametrize(
        ("fn", "name"),
        [
            (positive_rate, "positive_rate"),
            (oos_decay, "oos_decay"),
            (ic_trend, "ic_trend"),
        ],
    )
    def test_no_null_drop_no_warn_but_keys_present(self, fn, name):
        series = _ic_series(null_every=None)  # no nulls
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = (
                fn(series) if name != "positive_rate" else fn(series, forward_periods=1)
            )
        assert EXCESSIVE not in result.warning_codes
        assert result.metadata["drop_rate"] == 0.0
        assert result.metadata["dropped_periods"] == 0
        assert set(DROP_STAT_KEYS) <= set(result.metadata)

    def test_short_circuit_defers_no_drop_warn(self):
        # Mostly-null series → too few survivors → short-circuit; no drop warn.
        rng = np.random.default_rng(0)
        base = dt.date(2020, 1, 1)
        rows = [
            {
                "date": base + dt.timedelta(days=d),
                "value": None if d >= 4 else float(rng.normal()),
            }
            for d in range(150)
        ]
        series = pl.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = oos_decay(series)
        assert "reason" in result.metadata
        assert EXCESSIVE not in result.warning_codes
        assert "drop_rate" not in result.metadata


class TestSpreadTools:
    def test_quantile_spread_surfaces_null_drop(self):
        with pytest.warns(UserWarning, match="of periods dropped"):
            result = quantile_spread(_spread_panel(), forward_periods=1)["factor"]
        assert EXCESSIVE in result.warning_codes
        assert result.metadata["drop_rate"] == pytest.approx(0.5, abs=0.02)
        assert "spread" in result.metadata["drop_reason"]

    def test_quantile_spread_vw_records_schema(self):
        # vw bucketing fills top/bottom even on thin dates here, so no null
        # spread to drop — the five-key schema is still recorded on success.
        result = quantile_spread_vw(_spread_panel(), forward_periods=1)
        assert set(DROP_STAT_KEYS) <= set(result.metadata)
        # Nothing dropped → drop_reason is null (not the static criterion label).
        assert result.metadata["drop_rate"] == 0.0
        assert result.metadata["drop_reason"] is None

    def test_k_spread_records_schema(self):
        # k_spread's top-k/bottom-k yields a spread even on thin dates, so the
        # drop rate is low here — the schema is still recorded. Single-factor
        # path returns a MetricResult directly.
        result = k_spread(_spread_panel(), forward_periods=1)
        assert set(DROP_STAT_KEYS) <= set(result.metadata)
        assert result.metadata["drop_rate"] >= 0.0


class TestEvaluateBoundary:
    def test_evaluate_records_drop_warning(self):
        panel = _spread_panel()
        with pytest.warns(UserWarning, match="of periods dropped"):
            results = fx.evaluate(
                panel,
                metrics={"qs": quantile_spread()},
                factor_cols=["factor"],
                forward_periods=1,
                strict=False,
            )
        er = results["factor"]
        codes = [w.code for w in er.warnings]
        assert WarningCode.EXCESSIVE_PERIOD_DROPS in codes
        drop_warn = next(
            w for w in er.warnings if w.code == WarningCode.EXCESSIVE_PERIOD_DROPS
        )
        assert drop_warn.source == "qs"
