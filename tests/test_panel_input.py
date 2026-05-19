"""Tests for `_coerce_panel`, the strict polars-native API gateway.

`fx.evaluate` and `fx.run_metrics` accept `pl.DataFrame` (passes
through) and `pl.LazyFrame` (collected at the boundary). `pd.DataFrame`
is rejected with a guiding `TypeError` that points to
`factrix.adapt(df, ...)` or `pl.from_pandas(df)` as the documented
conversion paths.
"""

from __future__ import annotations

import factrix as fx
import polars as pl
import pytest
from factrix._metric_index import spec_by_name
from factrix._panel_input import _coerce_panel
from factrix.preprocess import compute_forward_return


@pytest.fixture(scope="module")
def panel_pl() -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=50, n_dates=120, seed=7)
    return compute_forward_return(raw, forward_periods=5)


def test_coerce_polars_dataframe_passthrough(panel_pl: pl.DataFrame) -> None:
    out = _coerce_panel(panel_pl)
    assert out is panel_pl


def test_coerce_lazyframe_collects(panel_pl: pl.DataFrame) -> None:
    out = _coerce_panel(panel_pl.lazy())
    assert isinstance(out, pl.DataFrame)
    assert out.equals(panel_pl)


def test_coerce_pandas_dataframe_is_rejected_with_guidance() -> None:
    pd = pytest.importorskip("pandas")
    pdf = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError, match=r"factrix\.adapt|pl\.from_pandas"):
        _coerce_panel(pdf)


def test_coerce_unsupported_type_raises() -> None:
    with pytest.raises(TypeError, match=r"pl\.DataFrame or pl\.LazyFrame"):
        _coerce_panel([{"date": 1}])


def test_evaluate_accepts_lazyframe_end_to_end(panel_pl: pl.DataFrame) -> None:
    ic = spec_by_name()["ic"]
    eager = fx.evaluate(
        panel_pl, metrics=[ic], factor_cols=["factor"], forward_periods=5
    )
    lazy = fx.evaluate(
        panel_pl.lazy(), metrics=[ic], factor_cols=["factor"], forward_periods=5
    )
    assert eager[0].metrics["ic"].value == lazy[0].metrics["ic"].value


def test_evaluate_rejects_pandas_with_guidance(panel_pl: pl.DataFrame) -> None:
    pytest.importorskip("pandas")
    pdf = panel_pl.to_pandas()
    ic = spec_by_name()["ic"]
    with pytest.raises(TypeError, match=r"factrix\.adapt|pl\.from_pandas"):
        fx.evaluate(pdf, metrics=[ic], factor_cols=["factor"], forward_periods=5)


def test_run_metrics_rejects_pandas_with_guidance(panel_pl: pl.DataFrame) -> None:
    pytest.importorskip("pandas")
    pdf = panel_pl.to_pandas()
    cfg = fx.AnalysisConfig.individual_continuous(forward_periods=5)
    with pytest.raises(TypeError, match=r"factrix\.adapt|pl\.from_pandas"):
        fx.run_metrics(pdf, cfg, metrics=["ic"])


def test_evaluate_rejects_unsupported_type() -> None:
    ic = spec_by_name()["ic"]
    with pytest.raises(TypeError, match=r"pl\.DataFrame or pl\.LazyFrame"):
        fx.evaluate(object(), metrics=[ic], factor_cols=["factor"], forward_periods=5)
