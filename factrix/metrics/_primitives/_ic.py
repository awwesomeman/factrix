from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._metric_index import cell
from factrix._types import MIN_ASSETS_PER_DATE_IC
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _attach_drop_stats


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
    batchable=True,
)
def compute_ic(
    df: pl.DataFrame,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
) -> dict[str, pl.DataFrame]:
    r"""Per-date Spearman Rank information coefficient (IC).

    Args:
        df: Panel with ``date``, ``asset_id``, every name in
            ``factor_cols``, and ``return_col``.
        factor_cols: Factor column names to score. All factors run in a
            single polars query (one ``with_columns`` + one
            ``group_by("date").agg(...)`` + one ``collect``) regardless
            of N. The N=1 case is just the general path specialised —
            no fast/slow path divergence.
        return_col: Forward-return column shared across factors.

    Returns:
        Dict mapping each factor name to a DataFrame with columns
        ``date, ic, tie_ratio`` sorted by date, plus an internal
        ``_drop_stats`` diagnostic struct column. Dates with fewer than
        ``MIN_ASSETS_PER_DATE_IC`` assets are dropped; ``_drop_stats``
        records how many were dropped (the aggregate drop-rate schema).
        ``tie_ratio`` is the per-date factor tie density
        $1 - n_{\mathrm{unique}} / n$ in $[0, 1]$.
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")

    rank_exprs: list[pl.Expr] = [
        pl.col(return_col).rank(method="average").over("date").alias("_rank_return"),
        *[
            pl.col(f).rank(method="average").over("date").alias(f"_rank__{f}")
            for f in cols
        ],
    ]
    agg_exprs: list[pl.Expr] = [pl.len().alias("n")]
    for f in cols:
        agg_exprs.append(pl.corr(f"_rank__{f}", "_rank_return").alias(f"_ic__{f}"))
        agg_exprs.append((1.0 - pl.col(f).n_unique() / pl.len()).alias(f"_tie__{f}"))

    # Collect the per-date frame *before* the cross-section filter so the
    # pre-drop date count is observable; the filter is shared across factors
    # (``n`` is per-date, not per-factor), so the drop stats are identical
    # for every factor.
    grouped = (
        df.lazy().with_columns(rank_exprs).group_by("date").agg(agg_exprs).sort("date")
    ).collect()
    n_periods_in = grouped.height
    wide = grouped.filter(pl.col("n") >= MIN_ASSETS_PER_DATE_IC)
    drop_reason = f"n_assets below MIN_ASSETS_PER_DATE_IC ({MIN_ASSETS_PER_DATE_IC})"

    return {
        f: _attach_drop_stats(
            wide.select(
                pl.col("date"),
                pl.col(f"_ic__{f}").alias("ic"),
                pl.col(f"_tie__{f}").alias("tie_ratio"),
            ),
            n_in=n_periods_in,
            drop_reason=drop_reason,
        )
        for f in cols
    }
