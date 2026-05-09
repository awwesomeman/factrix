"""Tests for factrix.metrics.regime — Layer A dispatcher."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._types import MetricOutput
from factrix.metrics import by_regime, ic, ic_ir
from factrix.metrics.regime import _slice_by_regime

pytestmark = pytest.mark.filterwarnings(
    "ignore:factrix.metrics.by_regime is deprecated:DeprecationWarning"
)


def _ic_series(n: int = 30, mean: float = 0.05, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": dates, "ic": rng.normal(mean, 0.02, n)}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


def _labels(dates: list, partition: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, "regime": partition}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestSliceByRegime:
    def test_join_with_labels(self):
        ic_df = _ic_series(10)
        labels = _labels(ic_df["date"].to_list(), ["a"] * 5 + ["b"] * 5)
        out = _slice_by_regime(ic_df, labels)
        assert "regime" in out.columns
        assert set(out["regime"].unique().to_list()) == {"a", "b"}
        assert len(out) == 10

    def test_unlabelled_dates_dropped(self):
        ic_df = _ic_series(10)
        labels = _labels(ic_df["date"].to_list()[:6], ["a"] * 6)
        out = _slice_by_regime(ic_df, labels)
        assert len(out) == 6

    def test_time_bisection_fallback(self):
        ic_df = _ic_series(11)
        out = _slice_by_regime(ic_df, regime_labels=None)
        assert set(out["regime"].unique().to_list()) == {"first_half", "second_half"}
        counts = out.group_by("regime").len().sort("regime")
        assert counts.filter(pl.col("regime") == "first_half")["len"][0] == 5
        assert counts.filter(pl.col("regime") == "second_half")["len"][0] == 6


class TestByRegimeDispatcher:
    def test_returns_dict_per_regime(self):
        ic_df = _ic_series(40)
        labels = _labels(ic_df["date"].to_list(), ["bull"] * 20 + ["bear"] * 20)
        out = by_regime(ic, ic_df, regime_labels=labels)
        assert isinstance(out, dict)
        assert set(out) == {"bull", "bear"}
        for v in out.values():
            assert isinstance(v, MetricOutput)
            assert v.name == "ic"

    def test_no_cross_regime_aggregation(self):
        """Layer A returns per-regime outputs only — no top-level summary."""
        ic_df = _ic_series(40)
        out = by_regime(ic, ic_df)
        for v in out.values():
            assert "per_regime" not in v.metadata
            assert "p_value_bhy_adjusted" not in v.metadata

    def test_fallback_when_labels_omitted(self):
        ic_df = _ic_series(40)
        out = by_regime(ic, ic_df)
        assert set(out) == {"first_half", "second_half"}

    def test_kwargs_forwarded(self):
        """Non-slice kwargs reach the underlying metric on every call."""
        ic_df = _ic_series(60)
        out = by_regime(ic, ic_df, forward_periods=3)
        for v in out.values():
            assert v.metadata.get("method", "").startswith("non-overlapping")

    def test_works_with_any_metric_callable(self):
        """No registry — any metric with a date-keyed first arg works."""
        ic_df = _ic_series(40)
        out = by_regime(ic_ir, ic_df)
        assert set(out) == {"first_half", "second_half"}
        for v in out.values():
            assert v.name == "ic_ir"

    def test_accepts_arbitrary_callable_matching_contract(self):
        """The contract is `(df, **kwargs) -> MetricOutput` — nothing factrix-specific.

        This pins the dispatcher's coupling: it must not rely on
        anything beyond the input being a date-keyed DataFrame and
        the callable returning a ``MetricOutput``.
        """

        def fake_metric(df: pl.DataFrame, *, multiplier: float = 1.0) -> MetricOutput:
            return MetricOutput(
                name="fake",
                value=float(df["ic"].mean()) * multiplier,
            )

        ic_df = _ic_series(40)
        labels = _labels(ic_df["date"].to_list(), ["a"] * 20 + ["b"] * 20)
        out = by_regime(fake_metric, ic_df, regime_labels=labels, multiplier=2.0)
        assert set(out) == {"a", "b"}
        assert all(o.name == "fake" for o in out.values())
        # kwarg forwarded to every per-regime call
        first_half_mean = ic_df.head(20)["ic"].mean()
        assert out["a"].value == pytest.approx(first_half_mean * 2.0)

    def test_missing_date_column_raises(self):
        df = pl.DataFrame({"asset_id": [1, 2], "ic": [0.1, 0.2]})
        with pytest.raises(ValueError, match="must have a 'date' column"):
            by_regime(ic, df)

    def test_scalar_input_raises_typeerror(self):
        from factrix.metrics import net_spread

        with pytest.raises(TypeError, match="polars DataFrame"):
            by_regime(net_spread, 0.001)

    def test_empty_overlap_raises(self):
        ic_df = _ic_series(10)
        future_dates = [datetime(2030, 1, 1) + timedelta(days=i) for i in range(5)]
        labels = _labels(future_dates, ["x"] * 5)
        with pytest.raises(ValueError, match="no rows survived"):
            by_regime(ic, ic_df, regime_labels=labels)

    def test_missing_regime_column_raises(self):
        ic_df = _ic_series(10)
        bad_labels = pl.DataFrame(
            {"date": ic_df["date"].to_list(), "label": ["a"] * 10}
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        with pytest.raises(ValueError, match="missing required column 'regime'"):
            by_regime(ic, ic_df, regime_labels=bad_labels)

    def test_fallback_emits_warning(self):
        ic_df = _ic_series(40)
        with pytest.warns(UserWarning, match="time-bisection"):
            by_regime(ic, ic_df)


class TestByRegimeDeprecation:
    @pytest.mark.filterwarnings("default")
    def test_emits_deprecation_warning(self):
        ic_df = _ic_series(40)
        labels = _labels(ic_df["date"].to_list(), ["bull"] * 20 + ["bear"] * 20)
        with pytest.warns(DeprecationWarning, match="by_slice"):
            by_regime(ic, ic_df, regime_labels=labels)
