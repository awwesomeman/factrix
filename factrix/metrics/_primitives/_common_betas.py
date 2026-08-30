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
from factrix._stats.ols import _ols_nw_slope_se
from factrix._types import EPSILON
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _attach_drop_stats,
    _finite_expr,
    _validate_factor_cols,
)

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


def _ew_portfolio_slope(
    data: pl.DataFrame,
    betas: pl.DataFrame,
    factor_col: str,
    return_col: str,
    overlap_periods: int,
) -> tuple[float | None, float | None, int]:
    r"""Calendar-time equal-weight portfolio slope on the common factor.

    The cross-asset test downstream treats the per-asset $\hat\beta_i$ as
    independent draws. Assets loading on a common component do not give
    independent betas, and $\mathrm{std}(\beta)/\sqrt{N}$ then understates
    $\mathrm{Var}(\bar\beta)$ without bound: at $N = 8$ and a residual
    correlation of 0.5 the iid test rejected 44.8% of true nulls at a nominal
    5%, and 79.2% at 0.9.

    The calendar-time form fixes it without an equicorrelation estimate
    ([Fama (1998)][fama-1998] portfolio approach, the same construction
    ``caar`` uses): with one regressor shared by every asset,
    $\bar{\hat\beta} = \mathrm{mean}_i \hat\beta_i$ is exactly the OLS slope of
    the equal-weight portfolio return $\bar R_t = \mathrm{mean}_i R_{it}$ on
    $F_t$ whenever the assets share the regressor sample, and that
    regression's residual $\bar\varepsilon_t$ carries whatever cross-asset
    residual covariance and heteroskedasticity there is. Its Newey-West SE
    with an $h - 1$ bandwidth (the overlap of $h$-period forward returns) is
    the sampling variance of the mean beta *given* the betas.

    Dates are those on which at least one surviving asset has a finite
    ``(factor, return)`` pair; on an unbalanced panel the portfolio holds the
    assets present that date and its slope can differ from the mean of the
    per-asset slopes — both are reported so the gap is visible.

    Returns ``(slope, var, n_periods)``; ``(None, None, n_periods)`` when
    fewer than three dates carry a pair or the factor has no time variation.
    """
    if betas.height < 1:
        return None, None, 0
    # Pivot to a date x asset grid and average in numpy: a polars group_by
    # mean sums in a thread-dependent order, and the last-ulp drift breaks the
    # batch == single-factor equality the primitive promises.
    survivors = data.join(betas.select("asset_id"), on="asset_id", how="inner")
    wide_y = (
        survivors.select("date", "asset_id", return_col)
        .pivot(index="date", on="asset_id", values=return_col)
        .sort("date")
    )
    wide_x = (
        survivors.select("date", "asset_id", factor_col)
        .pivot(index="date", on="asset_id", values=factor_col)
        .sort("date")
    )
    y = np.nanmean(wide_y.drop("date").to_numpy().astype(np.float64), axis=1)
    x = np.nanmean(wide_x.drop("date").to_numpy().astype(np.float64), axis=1)
    n_periods = len(y)
    if n_periods < 3:
        return None, None, n_periods
    slope, se, _ = _ols_nw_slope_se(y, x, lags=max(overlap_periods - 1, 0))
    # NaN (unformable fit) or ~0 (perfect fit): no usable variance either way.
    if not np.isfinite(se) or se < EPSILON:
        return None, None, n_periods
    return slope, se * se, n_periods


def _validate_compute_common_betas(m: MetricBase) -> None:
    """Knob bounds for ``compute_common_betas``, applied at construction."""
    _validate_factor_cols(
        m.factor_cols,  # type: ignore[attr-defined]
        func_name="compute_common_betas",
        docs_path="api/metrics/common_beta",
    )


@metric(
    cell=cell(FactorScope.COMMON, FactorDensity.DENSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.TS_THEN_CS,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
    batchable=True,
    validate=_validate_compute_common_betas,
)
def compute_common_betas(
    data: pl.DataFrame,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
    overlap_periods: int = 5,
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
        overlap_periods: Overlap horizon of ``return_col``; sets the
            Newey-West bandwidth ($h - 1$) of the calendar-time slope
            variance below. Injected by ``evaluate`` from the data's stamped
            horizon.

    Returns:
        Dict mapping each factor name to a DataFrame with columns
        ``asset_id, beta, alpha, t_stat, r_squared, n_obs`` sorted by
        ``asset_id``, three broadcast calendar-time columns —
        ``ew_portfolio_beta`` / ``ew_portfolio_beta_var`` (the equal-weight
        portfolio's slope on the factor and its Newey-West variance, the
        sampling variance ``common_beta`` tests the mean beta with — see
        :func:`_ew_portfolio_slope`; ``null`` when they cannot be estimated)
        and ``ew_portfolio_periods`` (dates behind them) — plus a broadcast
        ``_drop_stats`` carrier column on the
        assets axis (see :func:`_attach_drop_stats`) so cross-asset consumers
        can surface how much of the universe was silently dropped. An asset is
        emitted only with at least ``MIN_COMMON_BETA_PERIODS_HARD`` **finite**
        pairs and non-zero factor time-variation (zero-variance assets have no
        identifiable slope and are dropped). ``beta`` is always finite;
        ``t_stat`` may be **null** — see Notes.

    Notes:
        **``t_stat`` is homoskedastic and overlap-uncorrected.** It is the
        textbook OLS $t$ from the closed forms above: no heteroskedasticity
        correction and, more importantly here, no HAC term. With
        ``overlap_periods > 1`` each asset's forward-return series carries
        MA($h-1$) dependence by construction (Hansen-Hodrick 1980), so this
        $t$ is inflated by roughly $\sqrt{h}$ and its $p$-value is not usable
        as a per-asset significance test at a long horizon. It is published as
        a descriptive fit diagnostic — how well this asset's returns track the
        factor, alongside ``r_squared`` — and ``common_beta_profile`` surfaces
        it as such. For an actual single-asset slope test with a Newey-West
        SE and an $h-1$ bandwidth floor, use
        :func:`~factrix.metrics.predictive_beta.predictive_beta`. The
        cross-asset headline, ``common_beta``, consumes only the $\beta$
        vector and is unaffected.

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

    return {f: _common_betas_one(data, f, return_col, overlap_periods) for f in cols}


def _common_betas_one(
    data: pl.DataFrame, factor_col: str, return_col: str, overlap_periods: int
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
    ew_beta, ew_var, ew_periods = _ew_portfolio_slope(
        data.filter(valid_mask), result, factor_col, return_col, overlap_periods
    )
    result = result.with_columns(
        pl.lit(ew_beta, dtype=pl.Float64).alias("ew_portfolio_beta"),
        pl.lit(ew_var, dtype=pl.Float64).alias("ew_portfolio_beta_var"),
        pl.lit(ew_periods, dtype=pl.Int64).alias("ew_portfolio_periods"),
    )
    return _attach_drop_stats(
        result,
        axis="assets",
        n_in=n_assets_in,
        drop_reason=_COMMON_BETA_DROP_REASON,
    )
