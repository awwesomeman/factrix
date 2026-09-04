"""Reserved horizon stamps are complete-panel facts, not first-row hints."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import factrix as fx
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix.metrics import ic, quantile_spread

_STAMP_COLUMNS = ("_forward_periods", "_overlap_periods")


@pytest.fixture
def stamped_panel() -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=40, n_dates=80, rng=1050)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=1)
    assets = panel.get_column("asset_id").unique().sort().to_list()
    sectors = {asset: ("a" if i % 2 else "b") for i, asset in enumerate(assets)}
    return panel.with_columns(
        pl.col("asset_id").replace_strict(sectors).alias("sector")
    )


def _replace_stamp(
    panel: pl.DataFrame,
    column: str,
    values: Sequence[object],
    *,
    dtype: pl.DataType | type[pl.DataType],
) -> pl.DataFrame:
    return panel.with_columns(pl.Series(column, values, dtype=dtype))


def _mixed_stamp(panel: pl.DataFrame, column: str) -> pl.DataFrame:
    split = panel.height // 2
    values = [1] * split + [5] * (panel.height - split)
    return _replace_stamp(panel, column, values, dtype=pl.Int32)


@pytest.mark.parametrize("column", _STAMP_COLUMNS)
@pytest.mark.parametrize("reverse", [False, True])
def test_evaluate_rejects_mixed_stamps_regardless_of_row_order(
    stamped_panel: pl.DataFrame,
    column: str,
    reverse: bool,
) -> None:
    panel = _mixed_stamp(stamped_panel, column)
    if reverse:
        panel = panel.reverse()

    with pytest.raises(UserInputError) as excinfo:
        fx.evaluate(
            panel,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            strict=False,
        )

    assert excinfo.value.func_name == "evaluate"
    assert excinfo.value.field == column
    assert "constant" in str(excinfo.value)


@pytest.mark.parametrize("column", _STAMP_COLUMNS)
@pytest.mark.parametrize(
    ("values_factory", "dtype"),
    [
        (lambda n: [1] * (n - 1) + [None], pl.Int32),
        (lambda n: [True] * n, pl.Boolean),
        (lambda n: [1.0] * n, pl.Float64),
        (lambda n: [0] * n, pl.Int32),
        (lambda n: [-1] * n, pl.Int32),
    ],
    ids=["null", "bool", "float", "zero", "negative"],
)
def test_evaluate_rejects_invalid_stamp_columns(
    stamped_panel: pl.DataFrame,
    column: str,
    values_factory: Callable[[int], list[object]],
    dtype: pl.DataType | type[pl.DataType],
) -> None:
    values = values_factory(stamped_panel.height)
    panel = _replace_stamp(stamped_panel, column, values, dtype=dtype)

    with pytest.raises(UserInputError) as excinfo:
        fx.evaluate(panel, metrics={"ic": ic()}, factor_cols=["factor"])

    assert excinfo.value.func_name == "evaluate"
    assert excinfo.value.field == column
    assert "positive integer" in str(excinfo.value)


@pytest.mark.parametrize("column", _STAMP_COLUMNS)
def test_standalone_metric_validates_both_stamp_columns(
    stamped_panel: pl.DataFrame,
    column: str,
) -> None:
    panel = _mixed_stamp(stamped_panel, column)

    with pytest.raises(UserInputError) as excinfo:
        quantile_spread(panel, n_groups=3)

    assert excinfo.value.field == column


@pytest.mark.parametrize("column", _STAMP_COLUMNS)
def test_by_slice_rejects_mixed_stamps_before_partitioning(
    stamped_panel: pl.DataFrame,
    column: str,
) -> None:
    panel = _mixed_stamp(stamped_panel, column)
    split = panel.height // 2
    panel = panel.with_columns(
        pl.Series(
            "stamp_slice",
            ["one"] * split + ["five"] * (panel.height - split),
        )
    )

    with pytest.raises(UserInputError) as excinfo:
        fx.by_slice(
            panel,
            ic(),
            by="stamp_slice",
            factor_col="factor",
            strict=False,
        )

    assert excinfo.value.func_name == "by_slice"
    assert excinfo.value.field == column


@pytest.mark.parametrize("column", _STAMP_COLUMNS)
def test_slice_inference_validates_both_stamp_columns(
    stamped_panel: pl.DataFrame,
    column: str,
) -> None:
    panel = _mixed_stamp(stamped_panel, column)

    with pytest.raises(UserInputError) as excinfo:
        fx.slice_pairwise_test(
            panel,
            ic(),
            by="sector",
            factor_col="factor",
        )

    assert excinfo.value.func_name == "slice_pairwise_test"
    assert excinfo.value.field == column


def test_valid_constant_stamps_preserve_existing_results(
    stamped_panel: pl.DataFrame,
) -> None:
    standalone = quantile_spread(stamped_panel, n_groups=3)["factor"]
    evaluated = fx.evaluate(
        stamped_panel,
        metrics={"spread": quantile_spread(n_groups=3)},
        factor_cols=["factor"],
    )["factor"].metrics["spread"]

    assert standalone.value == evaluated.value
    assert standalone.n_obs == evaluated.n_obs
