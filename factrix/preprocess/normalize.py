"""Preprocessing Step 4-5: factor value normalization.

Step 4 — MAD Winsorize: per-date robust outlier clipping
Step 5 — Cross-sectional Z-score: MAD-based robust standardization

All functions expect canonical column names (date, asset_id).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import polars as pl

from factrix._codes import WarningCode
from factrix._errors import UserInputError
from factrix._types import (
    EPSILON,
    MAD_CONSISTENCY_CONSTANT,
    MIN_SCALE_ASSETS_HARD,
)

__all__ = ["cross_sectional_zscore", "mad_winsorize"]

_DOCS_MAD_WINSORIZE = "api/preprocess#factrix.preprocess.mad_winsorize"
_DOCS_ZSCORE = "api/preprocess#factrix.preprocess.cross_sectional_zscore"

# Where the per-date centre is taken. ``"median"`` is the robust default;
# ``"mean"`` buys an exactly mean-zero score at the cost of outlier
# sensitivity (see :func:`cross_sectional_zscore` Notes).
Center = Literal["median", "mean"]

# Croux-Rousseeuw (1992) finite-sample correction factors ``b_n`` for the
# MAD scale estimator. ``1.4826`` is the *asymptotic* consistency constant
# (1/Phi^-1(0.75)); at finite ``n`` the MAD is biased low, and the bias is
# ``n``-dependent — 18% at n=5, 9% at n=10, 4% at n=20. Left uncorrected,
# an unbalanced panel (delistings, staggered entry, event-triggered sparse
# dates) hands thin cross-sections systematically larger z-scores, so
# anything that pools or weights by z over-weights the thinnest, noisiest
# dates. The tabulated factors cover ``n <= 9``; above that the authors'
# asymptotic expansion ``n / (n - 0.8)`` is accurate to well under 1%.
_CR_SMALL_SAMPLE_FACTORS: dict[int, float] = {
    3: 1.495,
    4: 1.363,
    5: 1.206,
    6: 1.200,
    7: 1.140,
    8: 1.129,
    9: 1.107,
}


def _validate_n_mad(n_mad: object) -> float:
    """``n_mad`` must be a finite non-negative number.

    ``float("nan")`` used to pass the bare ``n_mad <= 0`` guard, produce NaN
    clip bounds, and be treated by polars as a no-op — winsorization silently
    skipped with no trace in the output. Validate in the style of
    ``_validate_forward_periods`` / ``_validate_winsorize_bounds``.
    """
    if isinstance(n_mad, bool) or not isinstance(n_mad, int | float):
        raise UserInputError(
            func_name="mad_winsorize",
            field="n_mad",
            value=n_mad,
            expected="a finite non-negative number of MAD units, e.g. 3.0",
            docs_path=_DOCS_MAD_WINSORIZE,
        )
    value = float(n_mad)
    if value != value or value in (float("inf"), float("-inf")) or value < 0.0:
        raise UserInputError(
            func_name="mad_winsorize",
            field="n_mad",
            value=n_mad,
            expected="a finite number >= 0 (0 disables winsorization)",
            docs_path=_DOCS_MAD_WINSORIZE,
        )
    return value


def _validate_center(center: object, func_name: str, docs_path: str) -> Center:
    if center not in ("median", "mean"):
        raise UserInputError(
            func_name=func_name,
            field="center",
            value=center,
            expected="'median' (robust, default) or 'mean' (mean-zero output)",
            docs_path=docs_path,
        )
    return center


def _finite(factor_col: str) -> pl.Expr:
    """``factor_col`` with non-finite entries (NaN / +-Inf) blanked to null.

    polars aggregations skip nulls but *propagate* float NaN, so a single
    NaN on a date would otherwise poison that date's median / MAD / std and
    silently take the whole cross-section out of the pipeline. Blanking
    non-finite values to null makes the per-date statistics ignore them, in
    line with the library convention (consumers use
    ``drop_nulls().drop_nans()``; producers never impute).
    """
    col = pl.col(factor_col)
    return pl.when(col.is_finite()).then(col).otherwise(None)


def _mad_correction_expr(n_col: str) -> pl.Expr:
    """Croux-Rousseeuw (1992) ``b_n`` keyed on the per-date finite count."""
    n = pl.col(n_col)
    expr: Any = pl.when(n >= 10).then(n / (n - 0.8))
    for size, factor in _CR_SMALL_SAMPLE_FACTORS.items():
        expr = expr.when(n == size).then(pl.lit(factor))
    return cast(pl.Expr, expr.otherwise(pl.lit(None, dtype=pl.Float64)))


def _per_date_scale(
    data: pl.DataFrame, factor_col: str, center: Center
) -> pl.DataFrame:
    """Per-date centre / scale / regime table for ``factor_col``.

    Columns: ``date``, ``_n_finite``, ``_center``, ``_scale``,
    ``_scale_method``, ``_sparse_zero``. ``_scale_method`` names which branch
    of the ladder produced ``_scale`` so every regime-driven switch is visible
    to the caller instead of being folded into one number:

    ``"mad"``
        ``b_n * 1.4826 * MAD`` — the robust, small-sample-consistent scale.
    ``"std"``
        The MAD collapsed to 0 (>50% ties at the median) and the per-date
        sample standard deviation (``ddof=1``) stood in for it.
    ``"constant"``
        No dispersion at all; ``_scale`` is ``0.0``.
    ``"insufficient_assets"``
        Fewer than ``MIN_SCALE_ASSETS_HARD`` finite values on the date —
        no robust scale exists, ``_scale`` is null.

    The MAD is always taken about the **median**, whatever ``center`` is:
    it is a dispersion about the median by definition, and its consistency
    constant is calibrated for that.
    """
    clean = _finite(factor_col)
    medians = data.group_by("date").agg(
        clean.count().alias("_n_finite"),
        clean.median().alias("_median"),
        (clean.mean() if center == "mean" else clean.median()).alias("_center"),
        clean.std(ddof=1).alias("_std"),
        (clean == 0).sum().alias("_n_zero"),
    )
    mad = (
        data.join(medians.select("date", "_median"), on="date", how="left")
        .group_by("date")
        .agg((clean - pl.col("_median")).abs().median().alias("_mad"))
    )
    stats = medians.join(mad, on="date", how="left")

    scale = (
        pl.when(pl.col("_n_finite") < MIN_SCALE_ASSETS_HARD)
        .then(pl.lit(None, dtype=pl.Float64))
        .when(pl.col("_mad") > EPSILON)
        .then(
            pl.col("_mad")
            * MAD_CONSISTENCY_CONSTANT
            * _mad_correction_expr("_n_finite")
        )
        .when(pl.col("_std") > EPSILON)
        .then(pl.col("_std"))
        .otherwise(pl.lit(0.0))
    )
    method = (
        pl.when(pl.col("_n_finite") < MIN_SCALE_ASSETS_HARD)
        .then(pl.lit("insufficient_assets"))
        .when(pl.col("_mad") > EPSILON)
        .then(pl.lit("mad"))
        .when(pl.col("_std") > EPSILON)
        .then(pl.lit("std"))
        .otherwise(pl.lit("constant"))
    )
    # A ``{0, R}`` sparse trigger column: the median is 0 and the MAD is 0 by
    # construction once fewer than half the assets fire, so the std fallback
    # is driven *by the triggers themselves* and collapses toward zero as the
    # trigger rate falls.
    sparse_zero = (
        (pl.col("_median") == 0)
        & (pl.col("_mad") <= EPSILON)
        & (pl.col("_n_zero") > 0)
        & (pl.col("_n_zero") < pl.col("_n_finite"))
    )
    return stats.select(
        "date",
        "_n_finite",
        pl.col("_center"),
        scale.alias("_scale"),
        method.alias("_scale_method"),
        sparse_zero.alias("_sparse_zero"),
    )


def _warn_scale_regimes(
    stats: pl.DataFrame, func_name: str, *, sparse_skipped: bool
) -> None:
    """Emit one ``UserWarning`` per regime-driven scale switch seen in ``stats``.

    The project convention is that a sample-regime-driven switch of estimator
    never happens silently. These functions return a bare DataFrame (there is
    no ``MetricResult`` to hang ``warning_codes`` on), so the
    :class:`~factrix._codes.WarningCode` value is carried in the warning text.
    """
    n_dates = stats.height
    if not n_dates:
        return
    counts: dict[str, int] = {}
    for method in stats["_scale_method"].to_list():
        counts[method] = counts.get(method, 0) + 1

    n_std = counts.get("std", 0)
    n_sparse = int(stats["_sparse_zero"].sum()) if sparse_skipped else 0
    n_std_non_sparse = n_std - n_sparse
    if n_sparse:
        warnings.warn(
            f"{func_name}: {WarningCode.SPARSE_WINSORIZE_SKIPPED.value} — "
            f"{n_sparse}/{n_dates} dates carry a sparse {{0, R}} factor whose "
            "median and MAD are both 0. The standard-deviation fallback there "
            "is driven by the triggers themselves and shrinks with the trigger "
            "rate, so clipping would destroy event magnitudes; those dates are "
            "left unwinsorized.",
            UserWarning,
            stacklevel=3,
        )
    if n_std_non_sparse > 0:
        warnings.warn(
            f"{func_name}: {WarningCode.ZERO_MAD_STD_FALLBACK.value} — "
            f"{n_std_non_sparse}/{n_dates} dates had a zero MAD (>50% ties at "
            "the median) and fell back to the non-robust per-date standard "
            "deviation (ddof=1). Robust and non-robust dates are mixed in one "
            "output column.",
            UserWarning,
            stacklevel=3,
        )
    n_thin = counts.get("insufficient_assets", 0)
    if n_thin:
        warnings.warn(
            f"{func_name}: {WarningCode.INSUFFICIENT_SCALE_ASSETS.value} — "
            f"{n_thin}/{n_dates} dates carry fewer than {MIN_SCALE_ASSETS_HARD} "
            "finite factor values; no robust scale exists there, so those dates "
            "are left unscaled (z-score null, clip skipped).",
            UserWarning,
            stacklevel=3,
        )


def _warn_non_finite(data: pl.DataFrame, factor_col: str, func_name: str) -> None:
    """Report the count of NaN / +-Inf inputs blanked to null."""
    n_bad = int(
        data.select(
            (pl.col(factor_col).is_not_null() & ~pl.col(factor_col).is_finite()).sum()
        ).item()
    )
    if n_bad:
        warnings.warn(
            f"{func_name}: {WarningCode.NON_FINITE_INPUT_DROPPED.value} — "
            f"{n_bad} row(s) carry NaN / +-Inf in {factor_col!r} and come back "
            "null. A non-finite tick is a data error, not an extreme value: "
            "clipping it into the band would manufacture a plausible-looking "
            "number that survives every downstream drop_nulls().drop_nans().",
            UserWarning,
            stacklevel=3,
        )


def mad_winsorize(
    data: pl.DataFrame,
    factor_col: str = "factor",
    n_mad: float = 3.0,
    *,
    center: Center = "median",
) -> pl.DataFrame:
    """Step 4: Per-date MAD-based winsorization on factor values.

    Clips factor values to ``[centre +- n_mad * b_n * 1.4826 * MAD]`` within
    each cross-section, where ``b_n`` is the Croux-Rousseeuw (1992)
    finite-sample correction for the per-date finite count.

    Args:
        factor_col: Column to winsorize in place.
        n_mad: Number of MAD units for clipping (default 3.0). Must be a
            finite number ``>= 0``; ``0`` disables the step.
        center: Where the clip band is centred — ``"median"`` (default,
            robust) or ``"mean"``. The MAD itself is always taken about the
            median.

    Raises:
        UserInputError: ``n_mad`` is not a finite number ``>= 0``, or
            ``center`` is not ``"median"`` / ``"mean"``.

    Returns:
        DataFrame with ``factor_col`` clipped in-place. Non-finite inputs
        come back **null**.

    Notes:
        **Small-sample MAD scaling.** ``1.4826`` is the asymptotic
        consistency constant; at finite ``n`` it under-estimates sigma by
        18% at n=5 and 9% at n=10, which makes ``n_mad=3.0`` an effective
        2.46 sigma clip on a five-name cross-section — over-winsorizing
        exactly the tail a factor's signal lives in. The Croux-Rousseeuw
        (1992) factors (tabulated for ``n <= 9``, ``n/(n-0.8)`` above)
        restore ``n_mad`` to its nominal sigma meaning at every
        cross-section size, so an unbalanced panel is clipped consistently
        across thin and wide dates.

        **Zero-MAD dates.** More than 50% ties on a date drives the MAD to
        exactly 0. Two sub-regimes are handled differently, and both raise a
        ``UserWarning``:

        - A sparse ``{0, R}`` trigger column (median 0, MAD 0) is **left
          unwinsorized** (``WarningCode.SPARSE_WINSORIZE_SKIPPED``). Its
          standard deviation is produced *by the triggers*, so
          ``3 * std`` collapses as the trigger rate falls — at one trigger
          in fifty names a unit event was clipped to 0.42, destroying 58%
          of the magnitude the metric downstream is trying to measure.
        - Any other zero-MAD date (a bucketed or binary factor) falls back
          to the per-date sample standard deviation (``ddof=1``), which
          keeps the factor finite and rank-preserving
          (``WarningCode.ZERO_MAD_STD_FALLBACK``).

        A cross-section with no dispersion at all has ``scale = 0`` and is
        left untouched by the clip. A date with fewer than
        ``MIN_SCALE_ASSETS_HARD`` finite values has no estimable robust
        scale and is left untouched too
        (``WarningCode.INSUFFICIENT_SCALE_ASSETS``).

        **Non-finite input.** NaN / +-Inf values are excluded from the
        per-date statistics *and blanked to null in the output*, matching
        :func:`cross_sectional_zscore`. Clipping them into the band (the
        previous behaviour) turned a hard data error into a plausible
        finite extreme that survived every downstream
        ``drop_nulls().drop_nans()`` and put the asset at the top of that
        date's ranking.

    References:
        - Croux, C. & Rousseeuw, P. J. (1992). "Time-Efficient Algorithms
          for Two Highly Robust Estimators of Scale." Finite-sample
          correction factors for the MAD.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import mad_winsorize
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> clipped = mad_winsorize(raw, n_mad=3.0)
        >>> clipped.columns == raw.columns
        True
        >>> clipped.height == raw.height
        True
    """
    n_mad = _validate_n_mad(n_mad)
    center = _validate_center(center, "mad_winsorize", _DOCS_MAD_WINSORIZE)
    if n_mad == 0.0:
        return data

    _warn_non_finite(data, factor_col, "mad_winsorize")
    stats = _per_date_scale(data, factor_col, center)
    _warn_scale_regimes(stats, "mad_winsorize", sparse_skipped=True)

    clean = _finite(factor_col)
    half_width = pl.col("_scale") * n_mad
    clipped = (
        pl.when(pl.col("_sparse_zero") | (pl.col("_scale").fill_null(0.0) <= EPSILON))
        .then(clean)
        .otherwise(
            clean.clip(pl.col("_center") - half_width, pl.col("_center") + half_width)
        )
        .alias(factor_col)
    )
    return (
        data.join(stats, on="date", how="left", maintain_order="left")
        .with_columns(clipped)
        .drop(stats.columns[1:])
    )


def cross_sectional_zscore(
    data: pl.DataFrame,
    factor_col: str = "factor",
    *,
    center: Center = "median",
) -> pl.DataFrame:
    """Step 5: MAD-robust z-score within each cross-section (date).

    ``z = (x - centre(x)) / (b_n * 1.4826 * MAD(x))``

    Args:
        factor_col: Column to standardize.
        center: Where the score is centred.

            - ``"median"`` (default) — robust to outliers, but **not
              mean-zero**. For any skewed factor (sparse event triggers,
              log-scaled fundamentals) ``mean(z) != 0``, so weights
              ``w ~ z`` carry a persistent net long or short leg and the
              book is charged market beta the caller did not intend. On a
              2-of-20 sparse trigger column the net exposure is ``+0.32``.
            - ``"mean"`` — subtracts the per-date mean instead, so the
              output is exactly mean-zero and ``w ~ z`` is dollar-neutral
              by construction, at the cost of a centre that a single
              outlier can move. The **scale** stays the robust MAD either
              way.

    Returns:
        DataFrame with a ``f"{factor_col}_zscore"`` column appended (so
        standardizing several factors in one panel no longer collides on a
        single hard-coded name).

    Notes:
        **Small-sample MAD scaling.** The scale carries the Croux-Rousseeuw
        (1992) finite-sample factor ``b_n`` on top of the asymptotic
        ``1.4826``. Without it the output of a function called
        "z-score" is not unit-scale at small ``n`` — its expected
        cross-sectional standard deviation is 1.67 at n=5 and 1.18 at
        n=10 — and, worse, the inflation is ``n``-dependent, so on an
        unbalanced panel thin dates produce systematically larger z and
        every pooled or z-weighted statistic over-weights the noisiest
        cross-sections. ``b_n`` removes the bias in the *scale estimator*
        (``E[b_n · 1.4826 · MAD] = σ`` to within 1% at every ``n >= 3``),
        which brings the expected ``sd(z)`` to 1.38 at n=5 and 1.08 at
        n=10; the residual excess is Jensen curvature in ``E[σ̂ / scale]``
        and no fixed constant removes it. The constant targets the
        *normal-consistent* scale: on heavier tails it overstates the
        distribution's own MAD-implied scale (about +12% at n=3 on a
        t(5), under +1% by n=50). See :func:`mad_winsorize`.

        **Zero-MAD fallback.** With more than 50% ties on a date the MAD is
        exactly 0, so the naive ratio is ``+-inf`` for every non-median
        value (and ``0/0 = NaN`` at the median). We fall back to the
        per-date sample standard deviation (``ddof=1``) as the scale, which
        keeps the z finite and rank-preserving — and raise a
        ``UserWarning`` carrying ``WarningCode.ZERO_MAD_STD_FALLBACK``,
        because robust and non-robust dates otherwise end up mixed into one
        column and one downstream statistic with no way to tell them apart.

        **Thin cross-sections.** A date with fewer than
        ``MIN_SCALE_ASSETS_HARD`` finite values yields **null**, not
        ``0.0``. At n=1 the old chain fell through to ``0.0`` — an
        "average" fabricated from a single observation — and at n=2 it
        returned ``+-0.6745`` whatever the two values were, a constant
        carrying no information but indistinguishable downstream from a
        real score. ``WarningCode.INSUFFICIENT_SCALE_ASSETS`` is raised.
        A fully constant cross-section of at least that size does have a
        genuine "no spread" reading and still maps to ``0.0``.

        **Nulls stay null.** Per the library convention (producers never
        impute; consumers drop with ``drop_nulls().drop_nans()``), a null
        input yields a null z-score, and so does a non-finite (NaN / +-Inf)
        input.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import cross_sectional_zscore
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> standardized = cross_sectional_zscore(raw)
        >>> "factor_zscore" in standardized.columns
        True
    """
    center = _validate_center(center, "cross_sectional_zscore", _DOCS_ZSCORE)
    _warn_non_finite(data, factor_col, "cross_sectional_zscore")
    stats = _per_date_scale(data, factor_col, center)
    _warn_scale_regimes(stats, "cross_sectional_zscore", sparse_skipped=False)

    clean = _finite(factor_col)
    zscore = (
        pl.when(clean.is_null() | pl.col("_scale").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(pl.col("_scale") > EPSILON)
        .then((clean - pl.col("_center")) / pl.col("_scale"))
        .otherwise(pl.lit(0.0))
    )
    return (
        data.join(stats, on="date", how="left", maintain_order="left")
        .with_columns(zscore.alias(f"{factor_col}_zscore"))
        .drop(stats.columns[1:])
    )
