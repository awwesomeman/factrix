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
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _attach_drop_stats

# Minimum complete (factor, return) pairs per date to estimate a slope.
# Two parameters (intercept + slope) leave one residual degree of freedom
# at three observations.
MIN_FM_ASSETS_HARD: int = 3
MIN_FM_ASSETS_WARN: int = 10


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
def compute_fm_betas(
    data: pl.DataFrame,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
) -> dict[str, pl.DataFrame]:
    r"""Per-date cross-sectional ordinary least squares (OLS) slope.

    Fits $R_i = \alpha + \beta \cdot \text{Signal}_i + \varepsilon$ per date
    and returns the time series of slopes $\beta_t$. The single-regressor
    OLS slope has the closed form

    $$\beta_t = \frac{\operatorname{Cov}_t(x, y)}{\operatorname{Var}_t(x)},$$

    so the whole panel is scored in one polars query (one
    ``group_by("date").agg(...)`` + one ``collect``) across all factors,
    with no per-date Python loop — the ``n_assets == 1`` case is just the general path
    specialised.

    Args:
        data: Panel with ``date``, ``asset_id``, every name in
            ``factor_cols``, and ``return_col``.
        factor_cols: Factor column names to score. All factors run in a
            single query regardless of N.
        return_col: Forward-return column shared across factors.

    Returns:
        Dict mapping each factor name to a DataFrame with columns
        ``date, beta, n_assets`` sorted by date, plus an internal
        ``_drop_stats`` diagnostic struct column. A date is emitted only
        when it has at least ``MIN_FM_ASSETS_HARD`` complete
        ``(factor, return)`` pairs and a non-degenerate cross-sectional
        spread; dates with zero factor variance (no identifiable slope)
        are dropped. Dates below ``MIN_FM_ASSETS_WARN`` survive but are
        surfaced by downstream FM consumers as thin-cross-section warnings.
        ``_drop_stats`` records the per-factor aggregate drop count.
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")

    agg_exprs: list[pl.Expr] = []
    for f in cols:
        # Restrict every moment to the pairwise-complete (factor, return) set
        # so the slope numerator and denominator share one sample — polars'
        # ``cov`` already pairwise-drops, but ``var`` would otherwise keep
        # factor-present / return-null rows and bias the ratio.
        both = pl.col(f).is_not_null() & pl.col(return_col).is_not_null()
        xf = pl.col(f).filter(both)
        yf = pl.col(return_col).filter(both)
        var_f = xf.var()
        agg_exprs.append(both.sum().alias(f"_cnt__{f}"))
        # ``var_f > 0`` (not an absolute epsilon) is the scale-free degeneracy
        # test: variance is exactly 0 only when the date has no cross-sectional
        # spread, which is the single case with no identifiable slope. A small
        # but real spread is a legitimate (if noisy) estimate and is kept.
        agg_exprs.append(
            pl.when(var_f > 0)
            .then(pl.cov(xf, yf) / var_f)
            .otherwise(None)
            .alias(f"_beta__{f}")
        )

    wide = data.lazy().group_by("date").agg(agg_exprs).sort("date").collect()
    # ``wide`` holds every date before the per-factor thinness / degeneracy
    # filter; its height is the shared pre-drop date count. ``n_periods_out``
    # differs per factor (each factor has its own ``_cnt`` and null betas).
    n_periods_in = wide.height
    drop_reason = (
        f"n_assets below MIN_FM_ASSETS_HARD ({MIN_FM_ASSETS_HARD}) or "
        f"degenerate cross-sectional variance"
    )

    return {
        f: _attach_drop_stats(
            wide.select(
                pl.col("date"),
                pl.col(f"_cnt__{f}").alias("_cnt"),
                pl.col(f"_beta__{f}").alias("beta"),
            )
            .filter(pl.col("_cnt") >= MIN_FM_ASSETS_HARD)
            .drop_nulls("beta")
            .select("date", "beta", pl.col("_cnt").alias("n_assets")),
            n_in=n_periods_in,
            drop_reason=drop_reason,
        )
        for f in cols
    }
