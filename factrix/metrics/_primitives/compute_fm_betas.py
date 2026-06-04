from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
    OutputShape,
    SEMethod,
    SpecRole,
    TestMethod,
)
from factrix._metric_index import cell
from factrix.metrics import metric


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
)
def compute_fm_betas(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    r"""Per-date cross-sectional ordinary least squares (OLS): $R_i = \alpha + \beta \cdot \text{Signal}_i + \varepsilon$."""
    dates = df["date"].unique().sort()
    rows: list[dict] = []

    for dt in dates:
        chunk = df.filter(pl.col("date") == dt)
        y = chunk[return_col].to_numpy().astype(np.float64)
        x = chunk[factor_col].to_numpy().astype(np.float64)

        if len(y) < 3:
            continue

        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        rows.append({"date": dt, "beta": float(beta[1])})

    if not rows:
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Datetime("ms")),
                "beta": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)
