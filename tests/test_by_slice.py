"""Tests for factrix.by_slice — partition a raw panel and evaluate per slice."""

from datetime import datetime, timedelta

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix import by_slice
from factrix._codes import WarningCode
from factrix._results import EvaluationResult
from factrix.metrics import caar, event_hit_rate, ic, positive_rate
from factrix.preprocess import compute_forward_return
from factrix.slicing._primitive import _slice_by
from factrix.slicing.dispatcher import _warn_date_axis_truncation


def _label_series(
    n: int = 40, label_col: str = "regime", seed: int = 42
) -> pl.DataFrame:
    """Date-keyed frame with a label column — for _slice_by tests."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    half = n // 2
    return pl.DataFrame(
        {
            "date": dates,
            "ic": rng.normal(0.05, 0.02, n),
            label_col: ["bull"] * half + ["bear"] * (n - half),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _sector_panel(
    n_assets: int = 60, n_dates: int = 200, seed: int = 0
) -> pl.DataFrame:
    """Raw cross-sectional panel with an asset-level ``sector`` column
    (cross-sectional axis: constant within each asset) and a date-derived
    ``year`` column (date axis: varies within an asset over time)."""
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, seed=seed)
    panel = compute_forward_return(raw, forward_periods=5)
    assets = panel["asset_id"].unique().sort().to_list()
    sector = {a: ("tech" if i % 2 else "fin") for i, a in enumerate(assets)}
    return panel.with_columns(
        pl.col("asset_id").replace_strict(sector).alias("sector"),
        pl.col("date").dt.year().alias("year"),
    )


class TestSliceByLabel:
    def test_partitions_on_existing_column(self):
        df = _label_series(20)
        out = _slice_by(df, "regime")
        assert set(out) == {"bull", "bear"}
        assert all(isinstance(v, pl.DataFrame) for v in out.values())
        assert sum(len(v) for v in out.values()) == 20

    def test_drops_label_column_from_partitions(self):
        df = _label_series(20)
        out = _slice_by(df, "regime")
        for sub in out.values():
            assert "regime" not in sub.columns

    def test_missing_label_raises(self):
        df = _label_series(10)
        with pytest.raises(ValueError, match="not found in data"):
            _slice_by(df, "sector")

    def test_empty_df_raises(self):
        df = _label_series(0)
        with pytest.raises(ValueError, match="empty"):
            _slice_by(df, "regime")

    def test_null_label_values_raise(self):
        df = _label_series(10).with_columns(
            pl.when(pl.int_range(0, 10) < 3)
            .then(None)
            .otherwise(pl.col("regime"))
            .alias("regime")
        )
        with pytest.raises(ValueError, match="contains nulls"):
            _slice_by(df, "regime")

    def test_numeric_label_stringified(self):
        df = _label_series(20).with_columns(
            pl.Series("decile", [1, 2] * 10, dtype=pl.Int64)
        )
        out = _slice_by(df, "decile")
        assert set(out) == {"1", "2"}

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="polars DataFrame"):
            _slice_by([1, 2, 3], "regime")  # type: ignore[arg-type]


class TestBySlice:
    def test_returns_dict_of_evaluation_results(self):
        panel = _sector_panel()
        out = by_slice(panel, ic(), by="sector", factor_col="factor")
        assert isinstance(out, dict)
        assert set(out) == {"tech", "fin"}
        for v in out.values():
            assert isinstance(v, EvaluationResult)
            assert "metric" in v.metrics

    def test_per_slice_sample_sizes_differ_from_full(self):
        panel = _sector_panel(n_assets=60)
        out = by_slice(panel, ic(), by="sector", factor_col="factor")
        # Each sector is an independent universe with ~half the assets.
        for v in out.values():
            assert v.n_assets <= 30

    def test_forward_periods_forwarded(self):
        panel = _sector_panel()
        out = by_slice(panel, ic(), by="sector", factor_col="factor", forward_periods=5)
        for v in out.values():
            assert (
                v.metrics["metric"]
                .metadata.get("method", "")
                .startswith("non-overlapping")
            )

    def test_comparison_frame_idiom(self):
        """The documented cross-slice table: stack to_frame + tag slice key."""
        panel = _sector_panel()
        out = by_slice(panel, ic(), by="sector", factor_col="factor")
        frame = pl.concat(
            [
                r.to_frame().with_columns(pl.lit(k).alias("slice"))
                for k, r in out.items()
            ]
        )
        assert set(frame["slice"].to_list()) == {"tech", "fin"}
        assert "value" in frame.columns

    def test_missing_by_raises(self):
        panel = _sector_panel()
        with pytest.raises(ValueError, match="not found in data"):
            by_slice(panel, ic(), by="industry", factor_col="factor")

    def test_bare_class_rejected(self):
        """metric must be an instance, consistent with evaluate."""
        panel = _sector_panel()
        with pytest.raises(Exception):  # noqa: B017 — evaluate's UserInputError
            by_slice(panel, ic, by="sector", factor_col="factor")  # type: ignore[arg-type]


class TestDateAxisTruncationWarning:
    """A cross-date metric sliced on a date axis warns; cross-sectional and
    per-date cases stay silent."""

    def test_cross_date_metric_on_date_axis_warns(self):
        panel = _sector_panel(n_dates=600)  # spans >1 year so `year` varies
        with pytest.warns(
            UserWarning, match=WarningCode.SLICE_BOUNDARY_TRUNCATION.value
        ):
            _warn_date_axis_truncation(panel, caar(), "year")

    def test_cross_date_metric_on_cross_sectional_axis_silent(self):
        panel = _sector_panel(n_dates=600)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_date_axis_truncation(panel, caar(), "sector")  # no raise == no warn

    def test_per_date_metric_on_date_axis_silent(self):
        panel = _sector_panel(n_dates=600)  # year genuinely varies
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_date_axis_truncation(panel, ic(), "year")  # CS_THEN_TS: no warn

    def test_sampling_phase_metric_on_date_axis_warns(self):
        panel = _sector_panel(n_dates=600)
        with pytest.warns(
            UserWarning, match=WarningCode.SLICE_BOUNDARY_TRUNCATION.value
        ):
            _warn_date_axis_truncation(panel, positive_rate(), "year")

    def test_boundary_insensitive_event_metric_on_date_axis_silent(self):
        panel = _sector_panel(n_dates=600)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_date_axis_truncation(panel, event_hit_rate(), "year")
