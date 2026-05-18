"""IC (Information Coefficient) computation for cross-sectional panels.

Notes:
    **Pipeline.** Per-date Spearman rank IC (cross-section step) → IC
    time series, then non-overlapping cross-asset t or Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) t on its
    mean; the regime variant slices the same pipeline.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.

    **Output.** Time-indexed IC series (``date, ic``) that can be fed
    into any ``series/`` tool (oos, trend, significance, hit_rate).
"""

from __future__ import annotations

import math
import warnings as _warnings
from collections.abc import Sequence

import polars as pl

from factrix._axis import FactorScope, Metric, Mode, Signal, Visibility
from factrix._metric_index import MetricSpec, cell
from factrix._stats import (
    _calc_t_stat,
    _newey_west_t_test,
    _p_value_from_t,
    _significance_marker,
)
from factrix._types import (
    EPSILON,
    MIN_ASSETS_PER_DATE_IC,
    MetricOutput,
)
from factrix.metrics._helpers import (
    TIE_RATIO_WARN_THRESHOLD,
    _sample_non_overlapping,
    _scaled_min_periods,
    _short_circuit_output,
)
from factrix.metrics._metric_capabilities import per_date_series_rename

__all__ = [  # noqa: RUF022 (teaching order, see #322 SSOT note)
    "compute_ic",
    "ic",
    "ic_newey_west",
    "ic_ir",
]

_IC_CELL = cell(
    FactorScope.INDIVIDUAL,
    Signal.CONTINUOUS,
    metric=Metric.IC,
    mode=Mode.PANEL,
)
_IC_PRIMITIVES = (
    "_newey_west_t_test",
    "_calc_t_stat",
    "_p_value_from_t",
    "_significance_marker",
    "_sample_non_overlapping",
    "_short_circuit_output",
)
_IC_INFERENCE = "NW HAC / cross-asset t"

# Slice-test contract (#153 §5): IC is per-date Spearman rank
# correlation, not a bucketed metric — slice tests never need to
# downscale `n_groups`. The min-cross-section-per-date constraint
# (Spearman ρ asymptotic distribution requires ≥ 30 obs per date,
# Hollander-Wolfe-Chicken §8.6) lives in the procedure as
# `MIN_ASSETS_PER_DATE_IC` short-circuit, parallel to (not exposed
# via) this attribute.
min_assets_per_group: int | None = None
per_date_series = per_date_series_rename("ic")


def _median_tie_ratio(ic_df: pl.DataFrame) -> float:
    """Median of the per-date ``tie_ratio`` column, or ``nan`` if absent/empty."""
    if "tie_ratio" not in ic_df.columns:
        return float("nan")
    med = ic_df["tie_ratio"].median()
    return float("nan") if med is None else float(med)  # type: ignore[arg-type]


def _warn_if_high_ic_tie_ratio(ic_df: pl.DataFrame, metric_name: str) -> float:
    """Emit ``UserWarning`` when median tie_ratio exceeds the global threshold.

    Returns the median for caller to stash in metadata. The Spearman ρ on
    average ranks is biased relative to the tie-corrected formula
    (Kendall-Stuart §31) at high tie densities — a bucketed / categorical
    factor will look like it has IC ≈ 0 even if the bucketing is
    informative. Threshold reuses the global ``TIE_RATIO_WARN_THRESHOLD``
    (0.3) shared with the quantile-bucketing diagnostics.
    """
    med = _median_tie_ratio(ic_df)
    if not math.isnan(med) and med > TIE_RATIO_WARN_THRESHOLD:
        _warnings.warn(
            f"{metric_name}: median tie_ratio={med:.3f} exceeds "
            f"{TIE_RATIO_WARN_THRESHOLD:.2f}. Spearman ρ on average ranks is "
            f"biased on bucketed / categorical factors; treat the IC "
            f"magnitude as a lower bound and consider a tie-corrected "
            f"correlation or a continuous transform of the factor.",
            UserWarning,
            stacklevel=2,
        )
    return med


def compute_ic(
    df: pl.DataFrame,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
) -> dict[str, pl.DataFrame]:
    r"""Per-date Spearman Rank information coefficient (IC).

    Args:
        df: Panel with ``date``, ``asset_id``, every name in
            ``factor_cols``, and ``return_col``.
        factor_cols: Factor column names to score. All factors run in a
            single polars query (one ``with_columns`` + one
            ``group_by("date").agg(...)`` + one ``collect``) regardless
            of N. The N=1 case is just the general path specialised —
            no fast/slow path divergence.
        return_col: Forward-return column shared across factors.

    Returns:
        Dict mapping each factor name to a DataFrame with columns
        ``date, ic, tie_ratio`` sorted by date. Dates with fewer than
        ``MIN_ASSETS_PER_DATE_IC`` assets are dropped. ``tie_ratio`` is
        the per-date factor tie density
        $1 - n_{\mathrm{unique}} / n$ in $[0, 1]$.

    Notes:
        Per-date Spearman IC is
        $\mathrm{IC}_t = \mathrm{corr}(\mathrm{rank}(f_t), \mathrm{rank}(r_t))$
        over the cross-section at date $t$; rank ties are broken with
        average rank (``method="average"``).

        At high tie rates Spearman $\rho$ on average ranks is biased
        relative to the tie-corrected formula (Kendall-Stuart §31). The
        per-date factor ``tie_ratio`` is surfaced alongside ``ic`` so
        downstream callers can detect bucketed / categorical signals
        without re-inspecting the input; ``ic`` / ``ic_newey_west`` /
        ``ic_ir`` aggregate it as the median across dates and stash it in
        ``MetricOutput.metadata["tie_ratio"]``. When the median exceeds
        ``TIE_RATIO_WARN_THRESHOLD`` (0.3) those aggregators also emit a
        ``UserWarning``: treat the IC magnitude as a lower bound and
        consider a tie-corrected correlation or a continuous transform
        of the factor.

        factrix drops dates whose cross-section has fewer than
        ``MIN_ASSETS_PER_DATE_IC`` assets — undersized panels yield
        rank-correlation estimates with degenerate variance.

    References:
        [Grinold 1989][grinold-1989]:
        $\mathrm{IR} \approx \mathrm{IC} \times \sqrt{\mathrm{breadth}}$
        motivates
        IC as the canonical signal-quality measure. The appraisal-ratio
        single-asset ancestor is [Treynor-Black
        1973][treynor-black-1973]; the breadth identity itself is
        Grinold's generalisation.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=120, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_by_factor = compute_ic(panel)
        >>> ic_df = ic_by_factor["factor"]
        >>> set(ic_df.columns) >= {"date", "ic", "tie_ratio"}
        True
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")

    # The "_rank__<col>" / "_ic__<col>" / "_tie__<col>" aliases are
    # internal to this query and never escape — per-factor DataFrames
    # are renamed back to the canonical ``ic`` / ``tie_ratio`` columns
    # before being placed in the returned dict.
    rank_exprs: list[pl.Expr] = [
        pl.col(return_col).rank(method="average").over("date").alias("_rank_return"),
        *[
            pl.col(f).rank(method="average").over("date").alias(f"_rank__{f}")
            for f in cols
        ],
    ]
    agg_exprs: list[pl.Expr] = [pl.len().alias("n")]
    for f in cols:
        agg_exprs.append(pl.corr(f"_rank__{f}", "_rank_return").alias(f"_ic__{f}"))
        agg_exprs.append((1.0 - pl.col(f).n_unique() / pl.len()).alias(f"_tie__{f}"))

    wide = (
        df.lazy()
        .with_columns(rank_exprs)
        .group_by("date")
        .agg(agg_exprs)
        .filter(pl.col("n") >= MIN_ASSETS_PER_DATE_IC)
        .sort("date")
        .collect()
    )

    return {
        f: wide.select(
            pl.col("date"),
            pl.col(f"_ic__{f}").alias("ic"),
            pl.col(f"_tie__{f}").alias("tie_ratio"),
        )
        for f in cols
    }


def ic(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> MetricOutput:
    r"""Information coefficient (IC) mean significance: is mean IC significantly different from zero?

    Args:
        ic_df: Output of ``compute_ic()``.
        forward_periods: Sampling interval for non-overlapping dates.

    Returns:
        MetricOutput with value=mean IC, t_stat from non-overlapping sampling.

    Notes:
        Given the per-date IC series $\mathrm{IC}_t$, significance is
        $t = \mathrm{mean}(\mathrm{IC}) / (\mathrm{std}(\mathrm{IC}) / \sqrt{n})$
        computed on a non-overlapping subsample (every
        ``forward_periods``-th date). $H_0: \mathbb{E}[\mathrm{IC}] = 0$.

        factrix uses non-overlapping resampling rather than Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC)
        for the default ``ic`` test to avoid the lag floor implied by
        overlapping forward returns; the HAC route is offered separately
        as ``ic_newey_west`` for callers who prefer to keep every sample.

    References:
        [Grinold 1989][grinold-1989]: IC as the canonical signal-quality
        measure under the Fundamental Law of Active Management.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: K-period overlapping
        returns carry MA(K-1) autocorrelation — the motivation for the
        non-overlap stride used here.

    Examples:
        Chain from :func:`compute_ic` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic, ic
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic(ic_df, forward_periods=5)
        >>> result.name
        'ic'
    """
    median_tie = _warn_if_high_ic_tie_ratio(ic_df, "ic")
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    raw_min = _scaled_min_periods(MIN_ASSETS_PER_DATE_IC, forward_periods)
    if n < raw_min:
        return _short_circuit_output(
            "ic",
            "insufficient_ic_periods",
            n_obs=n,
            min_required=raw_min,
            forward_periods=forward_periods,
        )

    mean_ic = float(ic_vals.mean())  # type: ignore[arg-type]
    sampled = _sample_non_overlapping(ic_df, forward_periods)["ic"].drop_nulls()
    n_sampled = len(sampled)
    if n_sampled < MIN_ASSETS_PER_DATE_IC:
        return _short_circuit_output(
            "ic",
            "insufficient_sampled_ic_periods",
            n_obs=n_sampled,
            min_required=MIN_ASSETS_PER_DATE_IC,
            forward_periods=forward_periods,
        )
    t = _calc_t_stat(float(sampled.mean()), float(sampled.std()), n_sampled)  # type: ignore[arg-type]
    p = _p_value_from_t(t, n_sampled)

    return MetricOutput(
        name="ic",
        value=mean_ic,
        stat=t,
        significance=_significance_marker(p),
        metadata={
            "n_periods": n,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "non-overlapping t-test",
            "tie_ratio": median_tie,
        },
    )


def ic_newey_west(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> MetricOutput:
    r"""Information coefficient (IC) mean significance via Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) $t$-test on the overlapping series.

    Sibling of ``ic()``: same null hypothesis ($H_0$: mean IC = 0), but
    keeps every observation and absorbs the autocorrelation induced by
    overlapping ``forward_periods``-day returns through HAC standard
    errors rather than dropping samples.

    Notes:
        $t = \mathrm{mean}(\mathrm{IC}) / \mathrm{SE}_{\mathrm{NW}}(\mathrm{IC})$
        on the full overlapping IC series. Lag selection:
        $L = \max(\lfloor T^{1/3} \rfloor, h - 1)$ (with $h$ = ``forward_periods``)
        — the [Andrews (1991)][andrews-1991] Bartlett growth rate, floored against the
        Hansen-Hodrick MA($h-1$) overlap horizon so the kernel covers
        the induced dependence.

        factrix uses the Andrews fixed-rate rule rather than the
        [Newey-West (1994)][newey-west-1994] data-adaptive bandwidth — simpler, deterministic
        across reruns, and adequate at the typical $T$ of factor research.

    References:
        [Newey-West 1987][newey-west-1987]: HAC variance estimator.
        [Andrews 1991][andrews-1991]: optimal Bartlett growth rate
        $T^{1/3}$ underlying the default lag rule.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: ``forward_periods - 1``
        floor for overlapping returns.
        [Newey-West 1994][newey-west-1994]: data-adaptive lag-selection
        alternative; cited as background.

    Examples:
        Chain from :func:`compute_ic` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic, ic_newey_west
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic_newey_west(ic_df, forward_periods=5)
        >>> result.name
        'ic_newey_west'
    """
    median_tie = _warn_if_high_ic_tie_ratio(ic_df, "ic_newey_west")
    ic_vals = ic_df["ic"].drop_nulls().to_numpy()
    n = len(ic_vals)
    if n < MIN_ASSETS_PER_DATE_IC:
        return _short_circuit_output(
            "ic_newey_west",
            "insufficient_ic_periods",
            n_obs=n,
            min_required=MIN_ASSETS_PER_DATE_IC,
        )

    from factrix._stats import _resolve_nw_lags

    lags = _resolve_nw_lags(n, lags=None, forward_periods=forward_periods)
    t, p, sig = _newey_west_t_test(ic_vals, forward_periods=forward_periods)
    return MetricOutput(
        name="ic_newey_west",
        value=float(ic_vals.mean()),
        stat=t,
        significance=sig,
        metadata={
            "n_periods": n,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "Newey-West HAC t-test on overlapping IC series",
            "newey_west_lags": lags,
            "forward_periods": forward_periods,
            "tie_ratio": median_tie,
        },
    )


def ic_ir(
    ic_df: pl.DataFrame,
) -> MetricOutput:
    r"""$\mathrm{ICIR} = \mathrm{mean}(\mathrm{IC}) / \mathrm{std}(\mathrm{IC})$.

    Signed ratio — positive when information coefficient (IC) is consistently positive, negative
    when consistently negative.  Analogous to a Sharpe ratio for the
    factor signal.

    This is a **descriptive statistic**, not a hypothesis test (t_stat=None).
    For significance testing, use ``ic()``.

    Args:
        ic_df: Output of ``compute_ic()``.

    Returns:
        MetricOutput with value=IC_IR (signed), t_stat=None.

    Notes:
        $\mathrm{ICIR} = \mathrm{mean}(\mathrm{IC}) / \mathrm{std}(\mathrm{IC})$
        over the per-date IC series — a Sharpe-style ratio describing
        time-series stability of the signal. Reported as a descriptive
        statistic; no inference is attached because the heteroskedasticity-and-autocorrelation-consistent (HAC)-corrected
        significance test on $\mathrm{mean}(\mathrm{IC})$ lives in ``ic``
        / ``ic_newey_west``.

    References:
        [Grinold 1989][grinold-1989]: ICIR is the time-stability
        normalisation that completes the IR decomposition.

    Examples:
        Chain from :func:`compute_ic` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic, ic_ir
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic_ir(ic_df)
        >>> result.name
        'ic_ir'
    """
    median_tie = _warn_if_high_ic_tie_ratio(ic_df, "ic_ir")
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    if n < MIN_ASSETS_PER_DATE_IC:
        return _short_circuit_output(
            "ic_ir",
            "insufficient_ic_periods",
            n_obs=n,
            min_required=MIN_ASSETS_PER_DATE_IC,
        )

    mean_ic = float(ic_vals.mean())  # type: ignore[arg-type]
    std_ic = float(ic_vals.std())  # type: ignore[arg-type]

    if std_ic < EPSILON:
        return _short_circuit_output(
            "ic_ir",
            "degenerate_ic_variance",
            std_ic=std_ic,
        )

    ratio = mean_ic / std_ic

    return MetricOutput(
        name="ic_ir",
        value=ratio,
        metadata={
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "n_periods": n,
            "tie_ratio": median_tie,
        },
    )


__metric_specs__ = (
    MetricSpec(
        name="compute_ic",
        cell=_IC_CELL,
        family="cs-first",
        inference=_IC_INFERENCE,
        primitives=_IC_PRIMITIVES,
        visibility=Visibility.INTERNAL,
        batchable=True,
    ),
    MetricSpec(
        name="ic",
        cell=_IC_CELL,
        family="cs-first",
        inference=_IC_INFERENCE,
        primitives=_IC_PRIMITIVES,
        requires={"ic_df": compute_ic},
    ),
    MetricSpec(
        name="ic_newey_west",
        cell=_IC_CELL,
        family="cs-first",
        inference=_IC_INFERENCE,
        primitives=_IC_PRIMITIVES,
        requires={"ic_df": compute_ic},
    ),
    MetricSpec(
        name="ic_ir",
        cell=_IC_CELL,
        family="cs-first",
        inference=_IC_INFERENCE,
        primitives=_IC_PRIMITIVES,
        requires={"ic_df": compute_ic},
    ),
)
