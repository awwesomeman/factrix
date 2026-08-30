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
from factrix._types import TiePolicy
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _assign_quantile_groups_batch,
    _finite_expr,
    _sample_non_overlapping,
    _validate_n_groups,
    _warn_thin_quantile_groups,
)


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
def compute_spread_series(
    data: pl.DataFrame,
    overlap_periods: int = 5,
    n_groups: int = 5,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
    tie_policy: TiePolicy = "ordinal",
    expected_warnings: tuple[str, ...] = (),
) -> dict[str, pl.DataFrame]:
    """Per-period long-short spread series (non-overlapping).

    Top bucket = highest factor rank; bottom bucket = lowest. Labels use
    ``top_return`` / ``bottom_return`` rather than ``q1_return`` /
    ``q5_return`` because the bucket width depends on ``n_groups`` — at
    ``n_groups=10`` the bottom is Q10, not Q5.

    Args:
        data: Panel with ``date, asset_id, factor, forward_return``.
        overlap_periods: Number of periods forward.
        n_groups: Number of quantile groups.
        factor_cols: Factor column names to score. All factors run in a
            single polars query (one ``with_columns`` + one
            ``group_by("date").agg(...)`` + one ``collect``) regardless
            of ``n_assets``. The ``n_assets == 1`` case is just the general path specialised —
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

        factrix uses non-overlap sub-sampling (stride ``overlap_periods``)
        before bucketing, not overlapping panel re-balancing — keeps the
        spread series free of MA(h-1) autocorrelation so downstream
        non-overlap t-tests are valid without heteroskedasticity-and-autocorrelation-consistent (HAC).

        **Non-finite handling.** A NaN ``return_col`` value is treated as
        missing (polars ``mean`` propagates NaN, so one bad print would
        otherwise NaN out the bucket mean and the spread); a null or NaN
        factor lands in no bucket and is excluded from ``_n_assets`` /
        ``_n_unique`` **and** ``universe_return``, so both the diagnostics
        and the benchmark the long / short legs are measured against
        describe the cross-section that was actually ranked.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import compute_spread_series
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> spreads = compute_spread_series(panel, overlap_periods=5, n_groups=5)
        >>> spread_df = spreads["factor"]
        >>> set(spread_df.columns) >= {"date", "spread", "top_return", "bottom_return"}
        True
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")
    _validate_n_groups(
        n_groups, func_name="compute_spread_series", docs_path="api/metrics/quantile"
    )

    sampled = _sample_non_overlapping(data, overlap_periods)

    _warn_thin_quantile_groups(sampled, n_groups, expected_warnings=expected_warnings)

    # Neutralise non-finite returns at the producer boundary: polars ``mean``
    # propagates float NaN, so one NaN return would turn a whole bucket mean —
    # and every spread built from it — into NaN. Nulls are what the aggregation
    # skips, so NaN is mapped onto null here rather than downstream.
    sampled = sampled.with_columns(
        pl.when(_finite_expr(return_col)).then(pl.col(return_col)).alias(return_col)
    )

    grouped = _assign_quantile_groups_batch(sampled, cols, n_groups, tie_policy)

    top_group = n_groups - 1
    bottom_group = 0

    # Per-factor top / bottom / universe means. The universe is per factor
    # and restricted to rows whose factor is finite — the same cross-section
    # the bucketing ranks. An earlier version shared one unfiltered mean
    # across factors, which benchmarked the long / short legs against names
    # the factor never ranked: with half the panel unranked and carrying a
    # different return level, a leg with zero true excess reported the gap
    # to the unranked names as alpha (the headline spread was unaffected —
    # the universe cancels in ``long_alpha + short_alpha``).
    agg_exprs: list[pl.Expr] = []
    for f in cols:
        # Counts and the universe are over *finite* factor values: ``count``
        # counts a NaN (it is not null) and ``n_unique`` would score NaN as
        # a distinct value, so a NaN-poisoned date would look wider and more
        # varied than the cross-section that actually gets bucketed.
        finite_f = _finite_expr(f)
        agg_exprs.append(
            pl.col(return_col).filter(finite_f).mean().alias(f"_universe__{f}")
        )
        agg_exprs.append(pl.col(f).filter(finite_f).n_unique().alias(f"_n_unique__{f}"))
        agg_exprs.append(pl.col(f).filter(finite_f).len().alias(f"_n_assets__{f}"))
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
            pl.when((pl.col(f"_n_assets__{f}") > 0) & (pl.col(f"_n_unique__{f}") <= 1))
            .then(pl.col(f"_universe__{f}"))
            .otherwise(pl.col(f"_top__{f}"))
            .alias("top_return"),
            pl.when((pl.col(f"_n_assets__{f}") > 0) & (pl.col(f"_n_unique__{f}") <= 1))
            .then(pl.col(f"_universe__{f}"))
            .otherwise(pl.col(f"_bot__{f}"))
            .alias("bottom_return"),
            pl.col(f"_universe__{f}").alias("universe_return"),
            pl.when((pl.col(f"_n_assets__{f}") > 0) & (pl.col(f"_n_unique__{f}") <= 1))
            .then(pl.lit(0.0))
            .otherwise(pl.col(f"_top__{f}") - pl.col(f"_bot__{f}"))
            .alias("spread"),
            ((pl.col(f"_n_assets__{f}") > 0) & (pl.col(f"_n_unique__{f}") <= 1)).alias(
                "_zero_variance_factor"
            ),
            pl.col(f"_n_assets__{f}").alias("_n_assets"),
        )
        for f in cols
    }
