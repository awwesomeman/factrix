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
from factrix._types import EPSILON
from factrix.metrics import metric

MIN_TS_OBS: int = 20


@metric(
    cell=cell(FactorScope.COMMON, FactorDensity.DENSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.TS_THEN_CS,
    test_method=TestMethod.T,
    se_method=SEMethod.OLS,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
)
def compute_ts_betas(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    r"""Per-asset time-series ordinary least squares (OLS): R_{i,t} = α_i + β_i · F_t + ε."""
    assets = df["asset_id"].unique().sort()
    rows: list[dict] = []

    for asset in assets:
        chunk = df.filter(pl.col("asset_id") == asset).sort("date")
        y = chunk[return_col].drop_nulls().to_numpy().astype(np.float64)
        x = chunk[factor_col].drop_nulls().to_numpy().astype(np.float64)

        n = min(len(y), len(x))
        if n < MIN_TS_OBS:
            continue
        y, x = y[:n], x[:n]

        X = np.column_stack([np.ones(n), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alpha_val = float(beta[0])
        beta_val = float(beta[1])

        resid = y - X @ beta
        ss_res = float(np.dot(resid, resid))
        centered = y - np.mean(y)
        ss_tot = float(np.dot(centered, centered))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0

        dof = n - 2
        if dof > 0 and ss_res / dof > EPSILON:
            sigma2 = ss_res / dof
            try:
                xtx_inv = np.linalg.inv(X.T @ X)
                se_beta = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
                t_stat = beta_val / se_beta if se_beta > EPSILON else 0.0
            except np.linalg.LinAlgError:
                t_stat = 0.0
        else:
            t_stat = 0.0

        rows.append(
            {
                "asset_id": asset,
                "beta": beta_val,
                "alpha": alpha_val,
                "t_stat": t_stat,
                "r_squared": r_sq,
                "n_obs": n,
            }
        )

    if not rows:
        return pl.DataFrame(
            {
                "asset_id": pl.Series([], dtype=pl.String),
                "beta": pl.Series([], dtype=pl.Float64),
                "alpha": pl.Series([], dtype=pl.Float64),
                "t_stat": pl.Series([], dtype=pl.Float64),
                "r_squared": pl.Series([], dtype=pl.Float64),
                "n_obs": pl.Series([], dtype=pl.Int64),
            }
        )

    return pl.DataFrame(rows)
