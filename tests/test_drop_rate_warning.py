"""Aggregate drop-rate warning — Phase 1 (PANEL→SERIES).

PANEL→SERIES primitives (``compute_ic``, ``compute_fm_betas``) silently drop
dates whose cross-section is too thin. Each records a canonical five-key
drop-stat schema at its ``.filter(...)`` step; each consumer copies the schema
into ``MetricResult.metadata`` and emits one aggregate ``UserWarning`` (plus a
``WarningCode.EXCESSIVE_PERIOD_DROPS``) when ``drop_rate`` clears the threshold.
"""

from __future__ import annotations

import warnings

import factrix as fx
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics._helpers import (
    DROP_RATE_WARN_THRESHOLD,
    DROP_STAT_KEYS,
    _read_drop_stats,
)
from factrix.metrics._primitives import compute_fm_betas
from factrix.metrics.ic import compute_ic, ic, ic_ir


def _thinned_panel(
    *, n_dates: int = 300, full: int = 40, thin: int = 1, seed: int = 0
) -> pl.DataFrame:
    """Panel where every other date is thinned to ``thin`` assets.

    ``thin`` < ``MIN_IC_ASSETS_HARD`` (2), so the thinned dates are dropped
    by ``compute_ic`` — yielding a ~50% period drop with enough survivors to
    clear the consumers' own sample floors.
    """
    raw = fx.datasets.make_cs_panel(n_assets=full, n_dates=n_dates, seed=seed)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
    dates = panel["date"].unique().sort()
    thin_dates = dates.gather(range(0, len(dates), 2))
    keep_assets = panel["asset_id"].unique().sort().gather(range(thin))
    return panel.filter(
        ~pl.col("date").is_in(thin_dates.implode())
        | pl.col("asset_id").is_in(keep_assets.implode())
    )


def _full_panel(*, n_dates: int = 300, n_assets: int = 40, seed: int = 0):
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, seed=seed)
    return fx.preprocess.compute_forward_return(raw, forward_periods=5)


class TestSchema:
    def test_keys_are_canonical(self):
        assert DROP_STAT_KEYS == (
            "n_periods_in",
            "n_periods_out",
            "dropped_periods",
            "drop_rate",
            "drop_reason",
        )

    def test_compute_ic_attaches_drop_stats(self):
        stats = _read_drop_stats(compute_ic(_thinned_panel())["factor"])
        assert stats is not None
        assert set(stats) == set(DROP_STAT_KEYS)
        # Half the dates are thinned below the IC per-date floor.
        assert (
            stats["n_periods_out"] == stats["n_periods_in"] - stats["dropped_periods"]
        )
        assert stats["drop_rate"] == pytest.approx(0.5, abs=0.02)
        assert "MIN_IC_ASSETS_HARD" in stats["drop_reason"]

    def test_compute_fm_betas_attaches_drop_stats(self):
        # Thin below MIN_FM_ASSETS (3): thinned dates carry a single asset.
        stats = _read_drop_stats(compute_fm_betas(_thinned_panel(thin=1))["factor"])
        assert stats is not None
        assert set(stats) == set(DROP_STAT_KEYS)
        assert stats["drop_rate"] > DROP_RATE_WARN_THRESHOLD
        assert "MIN_FM_ASSETS" in stats["drop_reason"]

    def test_full_panel_reports_zero_drop(self):
        stats = _read_drop_stats(compute_ic(_full_panel())["factor"])
        assert stats is not None
        assert stats["dropped_periods"] == 0
        assert stats["drop_rate"] == 0.0
        # Nothing dropped → reason is null, not the static criterion label.
        assert stats["drop_reason"] is None

    def test_partial_drop_reports_reason(self):
        stats = _read_drop_stats(compute_ic(_thinned_panel())["factor"])
        assert stats is not None
        assert stats["dropped_periods"] > 0
        assert "MIN_IC_ASSETS_HARD" in stats["drop_reason"]

    def test_hand_built_series_has_no_stats(self):
        # A series without the primitive's diagnostic column reads as None.
        bare = compute_ic(_full_panel())["factor"].select("date", "ic")
        assert _read_drop_stats(bare) is None


class TestConsumerWarning:
    def test_high_drop_rate_warns_and_codes(self):
        ic_df = compute_ic(_thinned_panel())["factor"]
        with pytest.warns(UserWarning, match="of periods dropped"):
            result = ic(ic_df, forward_periods=5)
        assert WarningCode.EXCESSIVE_PERIOD_DROPS.value in result.warning_codes
        assert result.metadata["drop_rate"] > DROP_RATE_WARN_THRESHOLD
        # Full five-key schema lands in metadata for programmatic inspection.
        assert set(DROP_STAT_KEYS) <= set(result.metadata)

    def test_low_drop_rate_no_warn_but_keys_present(self):
        ic_df = compute_ic(_full_panel())["factor"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = ic(ic_df, forward_periods=5)
        assert WarningCode.EXCESSIVE_PERIOD_DROPS.value not in result.warning_codes
        # Schema is still recorded even when below threshold.
        assert result.metadata["drop_rate"] == 0.0
        assert set(DROP_STAT_KEYS) <= set(result.metadata)

    def test_ic_ir_surfaces_drop_stats(self):
        ic_df = compute_ic(_thinned_panel())["factor"]
        with pytest.warns(UserWarning, match="ic_ir: .* of periods dropped"):
            result = ic_ir(ic_df)
        assert WarningCode.EXCESSIVE_PERIOD_DROPS.value in result.warning_codes
        assert set(DROP_STAT_KEYS) <= set(result.metadata)

    def test_full_drop_defers_to_short_circuit(self):
        # Every date below the IC floor → empty IC series → consumer
        # short-circuits and must NOT emit a drop-rate warning (no double-warn).
        raw = fx.datasets.make_cs_panel(n_assets=2, n_dates=300, seed=0)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        keep_asset = panel["asset_id"].unique().sort()[0]
        panel = panel.with_columns(
            pl.when(pl.col("asset_id") == keep_asset)
            .then(pl.col("factor"))
            .otherwise(None)
            .alias("factor")
        )
        ic_df = compute_ic(panel)["factor"]
        assert ic_df.height == 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = ic(ic_df, forward_periods=5)
        # Short-circuit output: NaN value, its own reason, no drop code.
        assert result.value != result.value  # NaN
        assert "reason" in result.metadata
        assert WarningCode.EXCESSIVE_PERIOD_DROPS.value not in result.warning_codes


class TestEvaluateBoundary:
    def test_evaluate_records_drop_warning(self):
        panel = _thinned_panel()
        with pytest.warns(UserWarning, match="of periods dropped"):
            results = fx.evaluate(
                panel, metrics={"m": ic()}, factor_cols=["factor"], forward_periods=5
            )
        er = results["factor"]
        codes = [w.code for w in er.warnings]
        assert WarningCode.EXCESSIVE_PERIOD_DROPS in codes
        # The structured Warning is sourced to the metric label.
        drop_warn = next(
            w for w in er.warnings if w.code == WarningCode.EXCESSIVE_PERIOD_DROPS
        )
        assert drop_warn.source == "m"

    def test_direct_and_evaluate_metadata_parity(self):
        panel = _thinned_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = ic(compute_ic(panel)["factor"], forward_periods=5)
            results = fx.evaluate(
                panel, metrics={"m": ic()}, factor_cols=["factor"], forward_periods=5
            )
        via_eval = results["factor"].metrics["m"]
        for key in DROP_STAT_KEYS:
            assert direct.metadata[key] == via_eval.metadata[key]
