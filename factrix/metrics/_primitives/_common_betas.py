from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
from factrix.metrics._helpers import _attach_drop_stats, _finite_expr

# Minimum complete (factor, return) observations per asset to fit a
# time-series slope. Mirrors the historical per-asset floor.
MIN_COMMON_BETA_PERIODS_HARD: int = 20

# One carrier label covering the three silent asset-axis reductions in
# ``_common_betas_one``: assets with no complete (factor, return) pairs vanish at
# the valid-mask group-by; assets with fewer than MIN_COMMON_BETA_PERIODS_HARD complete pairs
# are filtered; assets with zero factor time-variation yield a null slope that
# is dropped. The cross-asset consumers aggregate over the survivors, so the
# drop rate is measured against the raw universe.
_COMMON_BETA_DROP_REASON = (
    f"per-asset history below MIN_COMMON_BETA_PERIODS_HARD ({MIN_COMMON_BETA_PERIODS_HARD}), zero factor "
    f"variation, or no finite (factor, return) pairs"
)


def _residual_mean_pairwise_corr(
    data: pl.DataFrame,
    betas: pl.DataFrame,
    factor_col: str,
    return_col: str,
) -> float | None:
    r"""Mean off-diagonal correlation of the per-asset regression residuals.

    The cross-asset $t$-test downstream treats the per-asset $\hat\beta_i$ as
    independent draws. Assets loading on a common component do not give
    independent betas, and $\mathrm{std}(\beta)/\sqrt{N}$ then understates
    $\mathrm{Var}(\bar\beta)$ without bound: at $N = 8$ and a residual
    correlation of 0.5 the test rejected 44.8% of true nulls at a nominal 5%,
    and 79.2% at 0.9.

    $\bar r$ is the quantity that fixes it — the same role the within-period
    ICC plays for the pooled event statistics — so it is estimated here, where
    the residuals exist, and carried on the frame for
    :func:`~factrix.metrics.common_beta.common_beta` to deflate with.
    Residuals are $\varepsilon_{it} = R_{it} - \hat\alpha_i -
    \hat\beta_i F_t$; correlations are taken over the dates where **every**
    surviving asset has a finite residual (a rectangular panel — pairwise
    deletion can produce a non-PSD matrix whose mean off-diagonal is not a
    valid design input).

    Returns ``None`` when it cannot be estimated: fewer than two assets, or
    fewer than three shared dates.
    """
    if betas.height < 2:
        return None
    wide = (
        data.join(betas.select("asset_id", "alpha", "beta"), on="asset_id", how="inner")
        .with_columns(
            (
                pl.col(return_col)
                - pl.col("alpha")
                - pl.col("beta") * pl.col(factor_col)
            ).alias("_resid")
        )
        .select("date", "asset_id", "_resid")
        .pivot(index="date", on="asset_id", values="_resid")
        .drop("date")
        .drop_nulls()
    )
    matrix = wide.to_numpy()
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None
    # A constant residual series has no correlation with anything; excluding it
    # keeps corrcoef from emitting NaN rows.
    keep = matrix.std(axis=0) > EPSILON
    matrix = matrix[:, keep]
    if matrix.shape[1] < 2:
        return None
    corr = np.corrcoef(matrix, rowvar=False)
    off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
    if not np.isfinite(off_diagonal).any():
        return None
    return float(np.nanmean(off_diagonal))


@metric(
    cell=cell(FactorScope.COMMON, FactorDensity.DENSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.TS_THEN_CS,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
    batchable=True,
)
def compute_common_betas(
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
        ``asset_id``, a broadcast ``residual_mean_pairwise_corr`` column (the
        cross-asset residual correlation ``common_beta`` deflates its t with —
        see :func:`_residual_mean_pairwise_corr`; ``null`` when it cannot be
        estimated), plus a broadcast ``_drop_stats`` carrier column on the
        assets axis (see :func:`_attach_drop_stats`) so cross-asset consumers
        can surface how much of the universe was silently dropped. An asset is
        emitted only with at least ``MIN_COMMON_BETA_PERIODS_HARD`` **finite**
        pairs and non-zero factor time-variation (zero-variance assets have no
        identifiable slope and are dropped). ``beta`` is always finite;
        ``t_stat`` may be **null** — see Notes.

    Notes:
        Sample definition: "complete pair" means *finite* pair — non-null and
        neither ``NaN`` nor ``±inf``. polars keeps ``NaN`` as an ordinary float
        (unlike pandas, where ``NaN`` is the missing marker), so a single
        ``NaN`` cell used to leave ``Var(F) > 0`` while ``Cov(F, R)`` went
        ``NaN``; the asset then emitted a ``NaN`` beta that survived
        ``drop_nulls`` and poisoned every cross-asset aggregate downstream.
        Row-wise dropping (rather than voiding the whole asset) matches a
        pandas ``dropna()`` per-asset OLS, and ``n_obs`` is the finite-pair
        count actually regressed.

        ``t_stat`` is **null when the standard error is undefined**: an exact
        fit (``SSR = 0``, ``R² = 1``) or exhausted residual dof. The obvious
        alternative — the previous ``t_stat = 0.0`` — reads downstream as
        "maximally insignificant", the exact inverse of a perfectly determined
        slope, and it does so silently because 0.0 is a legal t. A null forces
        the consumer to decide; :mod:`factrix.metrics.common_beta` drops such
        assets from any t-based aggregate while keeping their ``beta``.
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")

    return {f: _common_betas_one(data, f, return_col) for f in cols}


def _common_betas_one(
    data: pl.DataFrame, factor_col: str, return_col: str
) -> pl.DataFrame:
    # In-line asset count vs the raw universe, captured before the valid-mask
    # filter so the carried drop rate reflects the silent reduction the
    # cross-asset consumers see — including assets dropped for having no
    # complete (factor, return) pairs at all.
    n_assets_in = data["asset_id"].n_unique()

    # Restrict every moment to the pairwise-FINITE (factor, return) set so cov
    # and var share one sample (polars cov pairwise-drops nulls; bare var would
    # not, and NEITHER skips float NaN), matching a per-asset OLS on the
    # complete observations. Without the finite test a single NaN cell left
    # ``_var_x > 0`` true while ``_cov`` went NaN, so the asset emitted a NaN
    # beta that sailed straight through ``drop_nulls("beta")``.
    valid_mask = _finite_expr(factor_col) & _finite_expr(return_col)

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
        .filter(pl.col("n_obs") >= MIN_COMMON_BETA_PERIODS_HARD)
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
            # ``None`` (not 0.0) when the SE is undefined: an exact fit
            # (SSR = 0, R² = 1) or exhausted dof gives no sampling variation
            # to divide by. Reporting t=0 there reads as "maximally
            # insignificant" for what is in fact a perfectly determined slope
            # — the exact inversion of the truth. A null propagates as
            # "undefined" and is dropped by the cross-asset consumers.
            pl.when((dof > 0) & (ss_res / dof > EPSILON) & (se_beta > EPSILON))
            .then(pl.col("beta") / se_beta)
            .otherwise(None)
            .cast(pl.Float64)
            .alias("t_stat"),
        )
        # Contract: "finite beta or no row". The finite-pair mask above already
        # rules NaN out of the moments; this makes the guarantee explicit.
        .filter(pl.col("beta").is_not_null() & pl.col("beta").is_finite())
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
    r_bar = _residual_mean_pairwise_corr(
        data.filter(valid_mask), result, factor_col, return_col
    )
    result = result.with_columns(
        pl.lit(r_bar, dtype=pl.Float64).alias("residual_mean_pairwise_corr")
    )
    return _attach_drop_stats(
        result,
        axis="assets",
        n_in=n_assets_in,
        drop_reason=_COMMON_BETA_DROP_REASON,
    )
