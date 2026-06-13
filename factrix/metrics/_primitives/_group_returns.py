from __future__ import annotations

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
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _assign_quantile_groups,
    _sample_non_overlapping,
)


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
)
def compute_group_returns(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    tie_policy: str = "ordinal",
) -> pl.DataFrame:
    """Mean forward return per quantile bucket (for monotonicity charts).

    Formula:
        1. Sample dates every ``forward_periods`` rows (non-overlapping).
        2. Per sampled date, assign each asset to a quantile group
           0..n_groups-1 by ``factor`` (see ``_assign_quantile_groups``
           for tie_policy semantics).
        3. For each group g:
              mean_return[g] = mean across (date, asset) where _group=g
                                of ``return_col``
        (Equal-weighted across all obs in the bucket, not per-date then
         averaged — use ``compute_spread_series`` if you want the latter.)

    Returns:
        DataFrame with ``group, mean_return`` sorted ascending by group.
        Group 0 = lowest factor rank, n_groups-1 = highest.

    Notes:
        ``mean_return[g] = mean over (date, asset) where _group=g of
        return_col`` — equal-weighted across all observations in the
        bucket pooled across dates. Use ``compute_spread_series`` if you
        want per-date bucket means averaged afterwards (the IC/IR-style
        aggregation order); the two differ when bucket cardinality moves
        across dates.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import compute_group_returns
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> groups = compute_group_returns(panel, forward_periods=5, n_groups=5)
        >>> set(groups.columns) >= {"group", "mean_return"}
        True
    """
    sampled = _sample_non_overlapping(df, forward_periods)
    grouped = _assign_quantile_groups(
        sampled,
        factor_col,
        n_groups,
        tie_policy=tie_policy,
    )

    return (
        grouped.group_by("_group")
        .agg(pl.col(return_col).mean().alias("mean_return"))
        .sort("_group")
        .rename({"_group": "group"})
    )
