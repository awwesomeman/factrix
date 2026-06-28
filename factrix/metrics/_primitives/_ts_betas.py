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
from factrix._types import EPSILON
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _attach_drop_stats

# Minimum complete (factor, return) observations per asset to fit a
# time-series slope. Mirrors the historical per-asset floor.
MIN_TS_PERIODS_HARD: int = 20

# One carrier label covering the three silent asset-axis reductions in
# ``_ts_betas_one``: assets with no complete (factor, return) pairs vanish at
# the valid-mask group-by; assets with fewer than MIN_TS_PERIODS_HARD complete pairs
# are filtered; assets with zero factor time-variation yield a null slope that
# is dropped. The cross-asset consumers aggregate over the survivors, so the
# drop rate is measured against the raw universe.
_TS_BETA_DROP_REASON = (
    f"per-asset history below MIN_TS_PERIODS_HARD ({MIN_TS_PERIODS_HARD}), zero factor "
    f"variation, or no complete (factor, return) pairs"
)


@metric(
    cell=cell(FactorScope.COMMON, FactorDensity.DENSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.TS_THEN_CS,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
    batchable=True,
)
def compute_ts_betas(
    data: pl.DataFrame,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
) -> dict[str, pl.DataFrame]:
    r"""Per-asset time-series ordinary least squares (OLS).

    Fits $R_{i,t} = \alpha_i + \beta_i \cdot F_t + \varepsilon$ per asset and
    returns one row per asset. The single-regressor OLS estimates have
    closed forms (all moments over the asset's pairwise-complete
    ``(factor, return)`` sample, with $S_{xx} = (n-1)\operatorname{Var}(x)$,
    etc.):

    $$\beta_i = \frac{\operatorname{Cov}(F, R_i)}{\operatorname{Var}(F)},
    \quad \alpha_i = \bar{R}_i - \beta_i \bar{F},
    \quad \text{SSR} = S_{yy} - \beta_i S_{xy},$$

    $$R^2 = 1 - \frac{\text{SSR}}{S_{yy}}, \quad
    \operatorname{SE}(\beta_i) = \sqrt{\frac{\text{SSR} / (n - 2)}{S_{xx}}},
    \quad t_i = \frac{\beta_i}{\operatorname{SE}(\beta_i)}.$$

    The whole panel is scored in one ``group_by("asset_id").agg(...)`` +
    one ``collect`` across all factors — no per-asset Python loop.

    Args:
        data: Panel with ``date``, ``asset_id``, every name in
            ``factor_cols``, and ``return_col``.
        factor_cols: Factor column names to score. All factors run in a
            single query regardless of N.
        return_col: Forward-return column shared across factors.

    Returns:
        Dict mapping each factor name to a DataFrame with columns
        ``asset_id, beta, alpha, t_stat, r_squared, n_obs`` sorted by
        ``asset_id``, plus a broadcast ``_drop_stats`` carrier column on the
        assets axis (see :func:`_attach_drop_stats`) so cross-asset consumers
        can surface how much of the universe was silently dropped. An asset is
        emitted only with at least ``MIN_TS_PERIODS_HARD`` complete pairs and
        non-zero factor time-variation (zero-variance assets have no
        identifiable slope and are dropped).
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")

    return {f: _ts_betas_one(data, f, return_col) for f in cols}


def _ts_betas_one(data: pl.DataFrame, factor_col: str, return_col: str) -> pl.DataFrame:
    # In-line asset count vs the raw universe, captured before the valid-mask
    # filter so the carried drop rate reflects the silent reduction the
    # cross-asset consumers see — including assets dropped for having no
    # complete (factor, return) pairs at all.
    n_assets_in = data["asset_id"].n_unique()

    # Restrict every moment to the pairwise-complete (factor, return) set so
    # cov and var share one sample (polars cov pairwise-drops; bare var would
    # not), matching a per-asset OLS on the complete observations.
    valid_mask = pl.col(factor_col).is_not_null() & pl.col(return_col).is_not_null()

    moments = (
        data.lazy()
        .filter(valid_mask)
        .group_by("asset_id")
        .agg(
            pl.len().alias("n_obs"),
            pl.col(factor_col).mean().alias("_xbar"),
            pl.col(return_col).mean().alias("_ybar"),
            pl.col(factor_col).var().alias("_var_x"),
            pl.col(return_col).var().alias("_var_y"),
            pl.cov(factor_col, return_col).alias("_cov"),
        )
        .filter(pl.col("n_obs") >= MIN_TS_PERIODS_HARD)
    )

    n = pl.col("n_obs")
    s_xx = (n - 1) * pl.col("_var_x")
    s_yy = (n - 1) * pl.col("_var_y")
    s_xy = (n - 1) * pl.col("_cov")
    # ``_var_x > 0`` (scale-free) is the degeneracy test: a factor with no
    # time-variation for an asset has no identifiable slope. Producing a null
    # (not 0/0 = NaN) lets the null be dropped instead of poisoning downstream
    # cross-asset aggregates.
    beta = (
        pl.when(pl.col("_var_x") > EPSILON)
        .then(pl.col("_cov") / pl.col("_var_x"))
        .otherwise(None)
    )
    # ss_res theoretically >= 0, but max_horizontal prevents float errors from producing negative values (e.g. when R^2 ≈ 1)
    ss_res = pl.max_horizontal(s_yy - beta * s_xy, 0.0)
    dof = n - 2
    se_beta = (ss_res / dof / s_xx).sqrt()

    result = (
        moments.with_columns(beta.alias("beta"))
        .with_columns(
            (pl.col("_ybar") - pl.col("beta") * pl.col("_xbar")).alias("alpha"),
            pl.when(s_yy > EPSILON)
            .then(1.0 - ss_res / s_yy)
            .otherwise(0.0)
            .alias("r_squared"),
            pl.when((dof > 0) & (ss_res / dof > EPSILON) & (se_beta > EPSILON))
            .then(pl.col("beta") / se_beta)
            .otherwise(0.0)
            .alias("t_stat"),
        )
        .drop_nulls("beta")
        .select(
            "asset_id",
            "beta",
            "alpha",
            "t_stat",
            "r_squared",
            pl.col("n_obs").cast(pl.Int64),
        )
        .sort("asset_id")
        .collect()
    )
    return _attach_drop_stats(
        result,
        axis="assets",
        n_in=n_assets_in,
        drop_reason=_TS_BETA_DROP_REASON,
    )
