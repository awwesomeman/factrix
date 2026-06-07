"""Tests for `_coerce_data`, the strict polars-native API gateway.

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
from factrix._data_input import _coerce_data
from factrix.preprocess import compute_forward_return


@pytest.fixture(scope="module")
def data_pl() -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=50, n_dates=120, seed=7)
    return compute_forward_return(raw, forward_periods=5)


def test_coerce_polars_dataframe_passthrough(data_pl: pl.DataFrame) -> None:
    out = _coerce_data(data_pl)
    assert out is data_pl


def test_coerce_lazyframe_collects(data_pl: pl.DataFrame) -> None:
    out = _coerce_data(data_pl.lazy())
    assert isinstance(out, pl.DataFrame)
    assert out.equals(data_pl)


def test_coerce_pandas_dataframe_is_rejected_with_guidance() -> None:
    pd = pytest.importorskip("pandas")
    pdf = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError, match=r"factrix\.adapt|pl\.from_pandas"):
        _coerce_data(pdf)


def test_coerce_unsupported_type_raises() -> None:
    with pytest.raises(TypeError, match=r"pl\.DataFrame or pl\.LazyFrame"):
        _coerce_data([{"date": 1}])


def test_evaluate_accepts_lazyframe_end_to_end(data_pl: pl.DataFrame) -> None:
    from factrix.metrics import ic

    eager = fx.evaluate(
        data_pl, metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
    )
    lazy = fx.evaluate(
        data_pl.lazy(), metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
    )
    assert eager["factor"].metrics["ic"].value == lazy["factor"].metrics["ic"].value


def test_evaluate_rejects_pandas_with_guidance(data_pl: pl.DataFrame) -> None:
    pytest.importorskip("pandas")
    pdf = data_pl.to_pandas()
    from factrix.metrics import ic

    with pytest.raises(TypeError, match=r"factrix\.adapt|pl\.from_pandas"):
        fx.evaluate(
            pdf, metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
        )


def test_evaluate_rejects_unsupported_type() -> None:
    from factrix.metrics import ic

    with pytest.raises(TypeError, match=r"pl\.DataFrame or pl\.LazyFrame"):
        fx.evaluate(
            object(), metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
        )
