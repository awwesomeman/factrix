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
from factrix._types import MIN_IC_ASSETS_HARD
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
    data: pl.DataFrame,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
) -> dict[str, pl.DataFrame]:
    r"""Per-date Spearman Rank information coefficient (IC).

    Args:
        data: Panel with ``date``, ``asset_id``, every name in
            ``factor_cols``, and ``return_col``.
        factor_cols: Factor column names to score. All factors run in a
            single polars query (one ``with_columns`` + one
            ``group_by("date").agg(...)`` + one ``collect``) regardless
            of N. The N=1 case is just the general path specialised —
            no fast/slow path divergence.
        return_col: Forward-return column shared across factors.

    Returns:
        Dict mapping each factor name to a DataFrame with columns
        ``date, ic, tie_ratio, n_assets`` sorted by date, plus an internal
        ``_drop_stats`` diagnostic struct column. Dates with fewer than
        ``MIN_IC_ASSETS_HARD`` assets are dropped; dates below
        ``MIN_IC_ASSETS_WARN`` survive but are surfaced by downstream IC
        consumers as thin-cross-section warnings. ``_drop_stats``
        records how many were dropped (the aggregate drop-rate schema).
        ``tie_ratio`` is the per-date factor tie density
        $1 - n_{\mathrm{unique}} / n$ in $[0, 1]$.
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")

    # Spearman ρ must rank each factor and the return over the *pairwise-complete*
    # (factor, return) set per date: a null in either column would otherwise shift
    # the surviving assets' ranks in the other column (polars' ``pl.corr`` drops
    # the null-paired rows only *after* ranking, so ranking the raw column first
    # distorts the ρ). The return rank is therefore masked per factor — each
    # factor has its own complete set — and collapses to the shared single rank
    # when the panel is dense (the common DENSE case).
    rank_exprs: list[pl.Expr] = []
    for f in cols:
        valid = pl.col(f).is_not_null() & pl.col(return_col).is_not_null()
        rank_exprs.append(
            pl.when(valid)
            .then(pl.col(return_col))
            .rank(method="average")
            .over("date")
            .alias(f"_rank_return__{f}")
        )
        rank_exprs.append(
            pl.when(valid)
            .then(pl.col(f))
            .rank(method="average")
            .over("date")
            .alias(f"_rank__{f}")
        )
    # The effective cross-section is the per-date *valid-pair* count: the IC ρ is
    # estimated on the complete (factor, return) names only, so that same count —
    # not the raw row count — is what gates the date (``MIN_IC_ASSETS_HARD``) and
    # denominates the tie ratio. Counting null-factor names would let a thin
    # cross-section (e.g. a factor defined for 8 of 200 names) clear the floor and
    # leak a high-variance IC into the series. The count is therefore per factor
    # (each factor nulls out a different set); it collapses to the row count on a
    # dense panel.
    agg_exprs: list[pl.Expr] = []
    for f in cols:
        valid = pl.col(f).is_not_null() & pl.col(return_col).is_not_null()
        agg_exprs.append(valid.sum().alias(f"_n_assets__{f}"))
        agg_exprs.append(
            pl.corr(f"_rank__{f}", f"_rank_return__{f}").alias(f"_ic__{f}")
        )
        agg_exprs.append(
            (1.0 - pl.col(f).filter(valid).n_unique() / valid.sum()).alias(f"_tie__{f}")
        )

    # Collect the per-date frame *before* the cross-section filter so the
    # pre-drop date count is observable. The floor is applied per factor on its
    # own valid-pair count, so the drop stats are per factor, not shared.
    grouped = (
        data.lazy()
        .with_columns(rank_exprs)
        .group_by("date")
        .agg(agg_exprs)
        .sort("date")
    ).collect()
    n_periods_in = grouped.height
    drop_reason = f"n_assets below MIN_IC_ASSETS_HARD ({MIN_IC_ASSETS_HARD})"

    return {
        f: _attach_drop_stats(
            grouped.filter(pl.col(f"_n_assets__{f}") >= MIN_IC_ASSETS_HARD).select(
                pl.col("date"),
                pl.col(f"_ic__{f}").alias("ic"),
                pl.col(f"_tie__{f}").alias("tie_ratio"),
                pl.col(f"_n_assets__{f}").alias("n_assets"),
            ),
            n_in=n_periods_in,
            drop_reason=drop_reason,
        )
        for f in cols
    }
