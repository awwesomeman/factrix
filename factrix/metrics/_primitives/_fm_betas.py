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
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _attach_drop_stats,
    _finite_expr,
    _validate_factor_cols,
)

# Minimum complete (factor, return) pairs per date to estimate a slope.
# Two parameters (intercept + slope) leave one residual degree of freedom
# at three observations.
MIN_FM_ASSETS_HARD: int = 3
MIN_FM_ASSETS_WARN: int = 10


def _validate_compute_fm_betas(m: MetricBase) -> None:
    """Knob bounds for ``compute_fm_betas``, applied at construction."""
    _validate_factor_cols(
        m.factor_cols,  # type: ignore[attr-defined]
        func_name="compute_fm_betas",
        docs_path="api/metrics/fm_beta",
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
    validate=_validate_compute_fm_betas,
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
        when it has at least ``MIN_FM_ASSETS_HARD`` **finite**
        ``(factor, return)`` pairs and a non-degenerate cross-sectional
        spread; dates with zero factor variance (no identifiable slope)
        are dropped. ``n_assets`` is the finite-pair count actually used
        by the regression. Dates below ``MIN_FM_ASSETS_WARN`` survive but
        are surfaced by downstream FM consumers as thin-cross-section
        warnings. ``_drop_stats`` records the per-factor aggregate drop
        count, which now also covers dates lost to non-finite cells.

    Notes:
        "Complete pair" means *finite* pair here — non-null **and** neither
        NaN nor ±inf. The mainstream pandas/statsmodels convention treats
        NaN as the sole missing marker and never sees a distinction; polars
        keeps NaN as a real float value, so a single NaN cell previously
        left ``var(x) > 0`` while ``cov(x, y)`` went NaN and the date
        emitted a NaN beta that survived ``drop_nulls``. Dropping the
        offending rows (rather than voiding the whole date, the other
        defensible choice) keeps the estimator's sample definition
        identical to a pandas ``dropna()`` cross-section and is why the
        pair count doubles as the effective ``n``.
    """
    cols = list(factor_cols)

    agg_exprs: list[pl.Expr] = []
    for f in cols:
        # Restrict every moment to the pairwise-FINITE (factor, return) set so
        # the slope numerator and denominator share one sample — polars' ``cov``
        # already pairwise-drops nulls, but ``var`` would otherwise keep
        # factor-present / return-null rows and bias the ratio, and neither
        # skips float NaN: a single NaN cell makes ``cov`` NaN while ``var``
        # stays > 0, so ``beta`` came out NaN and survived ``drop_nulls``.
        both = _finite_expr(f) & _finite_expr(return_col)
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
        f"n_assets below MIN_FM_ASSETS_HARD ({MIN_FM_ASSETS_HARD}) after "
        f"dropping non-finite (factor, return) pairs, or degenerate "
        f"cross-sectional variance"
    )

    return {
        f: _attach_drop_stats(
            wide.select(
                pl.col("date"),
                pl.col(f"_cnt__{f}").alias("_cnt"),
                pl.col(f"_beta__{f}").alias("beta"),
            )
            .filter(pl.col("_cnt") >= MIN_FM_ASSETS_HARD)
            # Belt-and-braces: the finite-pair mask already rules NaN out of
            # the moments, but the emitted contract is "finite beta or no row".
            .filter(pl.col("beta").is_not_null() & pl.col("beta").is_finite())
            .select("date", "beta", pl.col("_cnt").alias("n_assets")),
            n_in=n_periods_in,
            drop_reason=drop_reason,
        )
        for f in cols
    }
