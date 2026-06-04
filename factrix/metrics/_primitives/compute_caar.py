from __future__ import annotations

import warnings

import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    InputShape,
    OutputShape,
    SEMethod,
    SpecRole,
    TestMethod,
)
from factrix._metric_index import cell
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _is_sparse_magnitude_weighted


@metric(
    cell=cell(
        None, FactorDensity.SPARSE, DataStructure.PANEL, raw="(*, SPARSE, PANEL)"
    ),
    aggregation=Aggregation.EVENT_TIME,
    test_method=TestMethod.DESCRIPTIVE,
    se_method=SEMethod.NONE,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
)
def compute_caar(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    r"""Per-event-date weighted abnormal return series.

    Magnitude is preserved — no ``.sign()`` coercion.
    """
    if _is_sparse_magnitude_weighted(df, factor_col):
        warnings.warn(
            "compute_caar: factor column is mixed-sign and not a clean ±1 "
            "ternary. The result is the Sefcik-Thompson (1986) "
            "magnitude-weighted CAAR, not the textbook MacKinlay (1997) "
            "signed CAAR; apply .sign() to the column before calling for "
            "sign-flip semantics.",
            UserWarning,
            stacklevel=2,
        )
    return (
        df.filter(pl.col(factor_col) != 0)
        .with_columns((pl.col(return_col) * pl.col(factor_col)).alias("_signed_car"))
        .group_by("date")
        .agg(pl.col("_signed_car").mean().alias("caar"))
        .sort("date")
    )
