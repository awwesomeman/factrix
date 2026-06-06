from __future__ import annotations

import warnings
from collections.abc import Sequence

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
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _assign_quantile_groups_batch,
    _median_universe_size,
    _sample_non_overlapping,
)


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.OLS,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
    batchable=True,
)
def compute_spread_series(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
    tie_policy: str = "ordinal",
) -> dict[str, pl.DataFrame]:
    """Per-date long-short spread series (non-overlapping).

    Top bucket = highest factor rank; bottom bucket = lowest. Labels use
    ``top_return`` / ``bottom_return`` rather than ``q1_return`` /
    ``q5_return`` because the bucket width depends on ``n_groups`` — at
    ``n_groups=10`` the bottom is Q10, not Q5.

    Args:
        df: Panel with ``date, asset_id, factor, forward_return``.
        forward_periods: Number of periods forward.
        n_groups: Number of quantile groups.
        factor_cols: Factor column names to score. All factors run in a
            single polars query (one ``with_columns`` + one
            ``group_by("date").agg(...)`` + one ``collect``) regardless
            of N. The N=1 case is just the general path specialised —
            no fast/slow path divergence.
        return_col: Forward-return column shared across factors.
        tie_policy: See ``_assign_quantile_groups``. ``"ordinal"`` (default)
            keeps balanced bucket sizes; ``"average"`` keeps tied assets
            in the same bucket — prefer for low-cardinality factors.

    Returns:
        DataFrame with ``date, spread, top_return, bottom_return, universe_return``.

    Notes:
        Per non-overlapping date ``t``::

            top_return[t]    = mean_{i in Q_top} return[i, t]
            bottom_return[t] = mean_{i in Q_bot} return[i, t]
            spread[t]        = top_return[t] - bottom_return[t]

        factrix uses non-overlap sub-sampling (stride ``forward_periods``)
        before bucketing, not overlapping panel re-balancing — keeps the
        spread series free of MA(h-1) autocorrelation so downstream
        non-overlap t-tests are valid without heteroskedasticity-and-autocorrelation-consistent (HAC).

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import compute_spread_series
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> spreads = compute_spread_series(panel, forward_periods=5, n_groups=5)
        >>> spread_df = spreads["factor"]
        >>> set(spread_df.columns) >= {"date", "spread", "top_return", "bottom_return"}
        True
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")

    sampled = _sample_non_overlapping(df, forward_periods)

    median_n = _median_universe_size(sampled)
    per_group = median_n // n_groups if n_groups > 0 else 0
    if per_group < 5:
        warnings.warn(
            f"Median {per_group} assets per group (N={median_n}, "
            f"n_groups={n_groups}). Spread may be dominated by "
            f"individual assets. Consider reducing n_groups.",
            UserWarning,
            stacklevel=2,
        )

    grouped = _assign_quantile_groups_batch(sampled, cols, n_groups, tie_policy)

    top_group = n_groups - 1
    bottom_group = 0

    # Per-factor top / bottom means + a single shared universe mean.
    agg_exprs: list[pl.Expr] = [
        pl.col(return_col).mean().alias("universe_return"),
    ]
    for f in cols:
        agg_exprs.append(
            pl.col(return_col)
            .filter(pl.col(f"_group__{f}") == top_group)
            .mean()
            .alias(f"_top__{f}")
        )
        agg_exprs.append(
            pl.col(return_col)
            .filter(pl.col(f"_group__{f}") == bottom_group)
            .mean()
            .alias(f"_bot__{f}")
        )

    wide = grouped.group_by("date").agg(agg_exprs).sort("date")

    return {
        f: wide.select(
            pl.col("date"),
            pl.col(f"_top__{f}").alias("top_return"),
            pl.col(f"_bot__{f}").alias("bottom_return"),
            pl.col("universe_return"),
            (pl.col(f"_top__{f}") - pl.col(f"_bot__{f}")).alias("spread"),
        )
        for f in cols
    }
