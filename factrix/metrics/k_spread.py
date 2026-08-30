"""Fixed-K Top-K vs Bottom-K long-short spread for small cross-sections.

Notes:
    **Pipeline.** Per non-overlapping date, select the top ``k`` and
    bottom ``k`` names by factor rank, take the mean-return difference
    (cross-section step), then test the per-period spread series across
    time. Small cross-sections switch the headline test to a
    ``FEW_ASSETS`` advisory.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.

    **Output.** Mean spread, with the per-period cross-sectional return
    dispersion reported alongside.

    The small-`n_assets` counterpart of
    :func:`~factrix.metrics.quantile.quantile_spread`. Quantile bucketing
    (``n_groups=5`` ⇒ quintiles) degrades when ``n_assets < 30``: each bucket
    holds only a handful of names, so the spread is dominated by
    individual assets and the quintile breakpoints are unstable. Fixing
    the **count** ``k`` per leg keeps each leg's composition stable
    regardless of ``n_assets``, and the metric reports the contemporaneous
    cross-sectional dispersion so the spread can be read relative to the
    typical spread of returns that period.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import DDOF, MIN_PORTFOLIO_PERIODS_HARD, TiePolicy
from factrix.inference import (
    NEWEY_WEST,
    NON_OVERLAPPING,
    STATIONARY_BOOTSTRAP,
    NeweyWest,
    NonOverlapping,
    StationaryBootstrap,
)
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _all_dates_degenerate,
    _check_applicable_inference,
    _compute_tie_ratio,
    _degenerate_test_fields,
    _enforce_scaled_floor,
    _finite_expr,
    _finite_values,
    _no_signal_zero_variance,
    _sample_non_overlapping,
    _scaled_periods_threshold,
    _short_circuit_output,
    _spread_significance_with_inference,
    _surface_null_drop,
    _validate_choice,
    _warn_high_tie_ratio,
)

__all__ = [
    "k_spread",
]

# Inference allowlist: like ``quantile_spread``, a vetting record rather than a
# dispatch constraint — the non-overlap t-test, the Newey-West HAC and the
# stationary-bootstrap empirical p are the members measured on a spread series.
# Anything else (``HansenHodrick``, a non-``Inference`` object) is rejected.
applicable_inference: frozenset[NonOverlapping | NeweyWest | StationaryBootstrap] = (
    frozenset({NON_OVERLAPPING, NEWEY_WEST, STATIONARY_BOOTSTRAP})
)

_K_SPREAD_PERIODS_FLOOR = _scaled_periods_threshold(MIN_PORTFOLIO_PERIODS_HARD)


def _finite_mean(series: pl.Series) -> float:
    """Mean over the finite observations of *series*; NaN when there are none."""
    vals = _finite_values(series)
    return float("nan") if len(vals) == 0 else float(np.mean(vals.to_numpy()))


def _median_per_date_count(panel: pl.DataFrame) -> int:
    """Median per-period row count of the cleaned (finite factor+return) panel.

    The cross-section size that actually forms one date's legs. The
    ``FEW_ASSETS`` advisory keys on this rather than on
    ``asset_id.n_unique()`` over the whole panel: what matters is per period
    (``k`` names per leg out of *today's* names), so a universe that
    rotated through many tickers but only ever lists a handful at a time must
    read as thin, not wide.
    """
    if panel.is_empty():
        return 0
    med = panel.group_by("date").len()["len"].median()
    return 0 if med is None else int(med)  # type: ignore[arg-type]


def _k_spread_threshold(self) -> SampleThreshold:
    periods = _K_SPREAD_PERIODS_FLOOR(self)
    return SampleThreshold(
        min_periods=periods.min_periods,
        warn_periods=periods.warn_periods,
        min_assets=2 * self.k,
    )


def _build_k_spread_series(
    panel: pl.DataFrame,
    k: int,
    factor_col: str,
    return_col: str,
    tie_policy: TiePolicy = "ordinal",
) -> tuple[pl.DataFrame | None, pl.DataFrame]:
    """Per-period Top-K/Bottom-K spread series from a (possibly sampled) panel.

    Returns ``(series, clean)``: ``series`` has ``date, top_return,
    bottom_return, xs_dispersion, spread`` (``None`` when no date clears the
    ``2*k`` floor), and ``clean`` is the null-filtered panel for the
    short-circuit diagnostics / ``n_assets`` count. Shared by the
    non-overlap path (sampled panel) and the HAC path (full panel).
    """
    # NaN is dropped alongside null on BOTH columns. ``rank(descending=True)``
    # sorts NaN as the largest value, so a NaN factor would take rank 1 and put
    # a name with no signal at the head of the long leg; a NaN return would
    # propagate through the leg ``mean`` into the spread. Ranking therefore only
    # ever sees finite factor values, and ``_n_date`` — the bottom-leg cutoff —
    # counts exactly those rows.
    clean = panel.filter(_finite_expr(factor_col) & _finite_expr(return_col))
    if clean.is_empty():
        return None, clean
    # Constant factor: skip ranking (ordinal ties would manufacture a spurious
    # top/bottom split) and emit a spread=0 series; the body returns no-signal.
    if _all_dates_degenerate(clean, factor_col):
        series = (
            clean.group_by("date")
            .agg(
                pl.col(return_col).mean().alias("top_return"),
                pl.col(return_col).mean().alias("bottom_return"),
                pl.col(return_col).std(ddof=DDOF).alias("xs_dispersion"),
            )
            .with_columns(pl.lit(0.0).alias("spread"))
            .sort("date")
        )
        return series, clean
    # ``tie_policy`` is the caller's, not a hard-coded "ordinal". On a
    # discrete signal — a 3-level CTA event score with k=5, say — ordinal
    # tie-breaking fills the legs by row order among tied names and reports an
    # arbitrary split as a spread. ``"average"`` gives tied names a shared rank
    # so neither leg can be filled by sort artefacts; leg sizes then vary.
    ranked = clean.with_columns(
        pl.col(factor_col)
        .rank(method=tie_policy, descending=True)
        .over("date")
        .alias("_rank"),
        pl.len().over("date").alias("_n_date"),
    ).filter(pl.col("_n_date") >= 2 * k)

    if ranked.height == 0:
        return None, clean

    series = (
        ranked.group_by("date")
        .agg(
            pl.col(return_col).filter(pl.col("_rank") <= k).mean().alias("top_return"),
            pl.col(return_col)
            .filter(pl.col("_rank") > pl.col("_n_date") - k)
            .mean()
            .alias("bottom_return"),
            pl.col(return_col).std(ddof=DDOF).alias("xs_dispersion"),
        )
        .with_columns((pl.col("top_return") - pl.col("bottom_return")).alias("spread"))
        .sort("date")
    )
    return series, clean


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    # Periods floor scales with the non-overlap stride (see ``quantile``): the
    # spread series is sub-sampled at ``overlap_periods``, so pre-flight and the
    # in-body gate share ``MIN_PORTFOLIO_PERIODS_HARD`` + ``_scaled_min_periods``.
    sample_threshold=_k_spread_threshold,
)
def k_spread(
    data: pl.DataFrame,
    overlap_periods: int = 5,
    k: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    tie_policy: TiePolicy = "ordinal",
    inference: NonOverlapping | NeweyWest | StationaryBootstrap = NON_OVERLAPPING,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Fixed-K Top-K vs Bottom-K long-short spread.

    Per non-overlapping date, the long leg is the mean forward return of
    the ``k`` highest-factor names and the short leg the mean of the
    ``k`` lowest; the spread is their difference. The mean spread is
    tested across time.

    Args:
        data: Panel with ``date, asset_id``, ``factor_col`` and
            ``return_col``.
        overlap_periods: Sampling stride for non-overlapping dates;
            match the forward-return horizon.
        k: Number of names per leg (fixed count, not a quantile
            fraction). A date needs at least ``2 * k`` names to form
            disjoint legs — dates with fewer are dropped.
        factor_col: Ranking column (default ``"factor"``).
        return_col: Realised-return column (default ``"forward_return"``).
        tie_policy: Tie-break policy for the leg ranking — the same knob and
            the same values as ``quantile_spread`` / ``quantile_spread_vw`` /
            ``monotonicity``. Either ``"ordinal"`` or ``"average"``; anything else raises ``UserInputError``.
            ``"ordinal"`` (default) breaks ties by row order,
            which keeps both legs exactly ``k`` names wide but fills them
            arbitrarily among tied values; ``"average"`` gives tied names a
            shared rank so a discrete signal cannot be split by sort artefacts.
            Note what that means for a *fixed-k* leg: a tied block wider than
            ``k`` puts no name inside the cutoff, so the leg is empty and the
            date drops out (the drop-rate advisory reports it). That is the
            honest answer — five names cannot be picked out of a ten-way tie
            without an arbitrary rule — but it costs sample, so on a heavily
            tied factor prefer ``quantile_spread`` with ``tie_policy="average"``,
            whose legs are defined by rank *fraction* rather than count. The realised per-period tie
            ratio is reported in ``metadata["tie_ratio"]`` and a median above
            the shared threshold raises a ``UserWarning``. Only
            ``_all_dates_degenerate`` used to guard this, and it fires only
            when the factor is constant everywhere — a 3-level signal ranked
            ordinally into fixed-``k`` legs reported an arbitrary split as a
            spread with no tie diagnostic at all.
        inference: Headline significance method. ``fx.inference.NON_OVERLAPPING``
            (default) runs the OLS t-test on the non-overlap stride;
            ``fx.inference.NEWEY_WEST`` keeps every date and HAC-corrects the
            MA(h-1) SE. The small-cross-section block bootstrap still takes
            precedence over either when it fires (HAC corrects autocorrelation,
            not heavy tails); the override is flagged in ``metadata``.

    Returns:
        MetricResult with value = mean spread, ``stat`` = ``t`` on the
        spread series, p-value from the cross-section-aware significance
        path. ``metadata["cross_sectional_dispersion"]`` carries the
        mean per-period cross-sectional standard deviation of returns.

    Notes:
        Per qualifying date ``t`` (universe size ``n_assets_t >= 2 * k``), with
        $\mathrm{top}_k$ / $\mathrm{bot}_k$ the names ranked $1..k$ /
        ``n_assets_t - k + 1 .. n_assets_t`` by factor:

        $$\text{spread}_t = \frac1k \sum_{i \in \mathrm{top}_k} r_{i,t}
        - \frac1k \sum_{i \in \mathrm{bot}_k} r_{i,t}.$$

        ``value = mean_t spread_t``. The headline test is the non-overlapping
        ``t`` on the strided spread series (or Newey-West HAC on the full
        series under ``inference=NEWEY_WEST``); a thin cross-section attaches
        ``FEW_ASSETS`` and changes nothing else. The contemporaneous
        cross-sectional dispersion $\mathrm{std}_i(r_{i,t})$ is averaged
        over dates and reported so the spread can be judged against the
        period's return spread.

        **What "small" counts.** The advisory reads the median *per-period*
        number of usable names (``metadata["median_cross_section"]``), not
        the count of distinct ``asset_id`` values in the panel: the legs
        are formed date by date, so a universe listing 12 names at a time
        while rotating through 200 over the sample is thin. The
        universe-wide count is still what the ``2 * k`` feasibility
        short-circuit uses — that one is about whether the legs can exist
        at all.

        **Non-finite observations.** Rows with a null or NaN factor or
        return are dropped before ranking: ``rank(descending=True)`` sorts
        NaN as the largest value, so a NaN factor would otherwise take
        rank 1 and head the long leg. Every series column collapsed to a
        scalar is then filtered with ``drop_nulls().drop_nans()`` — one
        NaN in the spread series would make ``_calc_t_stat`` return NaN —
        withholding the test as ``degenerate_variance`` — or make the
        bootstrap raise.

        **Which sample each count describes.** ``value``, ``stat``,
        ``p_value`` and ``n_obs`` all come from the sample the selected
        inference ran on. ``n_obs`` == ``metadata["n_periods"]`` is the
        strided count under ``NON_OVERLAPPING`` (and under a bootstrap
        override) but the *full overlapping* count under ``NEWEY_WEST``,
        which never touches the strided series;
        ``metadata["n_periods_strided"]`` always carries the non-overlap
        count and ``metadata["n_periods_full"]`` the overlapping one on
        the HAC path. The ``n_dropped`` / ``n_periods_in`` /
        ``n_periods_out`` keys describe the null-drop on the strided
        series.

    References:
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: overlapping-return
        autocorrelation, motivating the non-overlap stride.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.k_spread import k_spread
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=20, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> result = k_spread(panel, overlap_periods=5, k=3)
        >>> result.name == ""
        True
    """
    _validate_choice(
        tie_policy,
        TiePolicy,
        func_name="k_spread",
        field="tie_policy",
        docs_path="api/metrics/k_spread",
    )
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    _check_applicable_inference(inference, applicable_inference, func_name="k_spread")
    if return_col not in data.columns:
        return _short_circuit_output(
            "k_spread",
            "no_return_column",
            missing_column=return_col,
        )

    # Drop rows with no factor or no realised return BEFORE ranking: ``rank``
    # skips nulls but ``pl.len`` would still count them, so ``_n_date`` would
    # overcount and the bottom-leg cutoff (``_rank > _n_date - k``) would point
    # past the last real rank — silently shrinking or emptying the short leg.
    # forward_return is null on the last ``overlap_periods`` rows per asset.
    sampled = _sample_non_overlapping(data, overlap_periods)
    tie_ratio = _compute_tie_ratio(sampled, factor_col)
    high_tie_ratio = _warn_high_tie_ratio(
        tie_ratio,
        "k_spread",
        tie_policy,
        expected_warnings=expected_warnings,
    )
    series, clean = _build_k_spread_series(
        sampled, k, factor_col, return_col, tie_policy
    )
    n_assets = clean["asset_id"].n_unique()
    # Real per-period asset count (not the universe-wide unique count): both
    # insufficient-assets short-circuits report it under the same reason.
    max_per_date_value = (
        clean.group_by("date").len()["len"].max() if not clean.is_empty() else None
    )
    max_per_date = 0 if max_per_date_value is None else cast(int, max_per_date_value)
    if n_assets < 2 * k:
        return _short_circuit_output(
            "k_spread",
            "insufficient_assets_for_k_legs",
            n_obs=n_assets,
            n_obs_axis="assets",
            k=k,
            min_required=2 * k,
            max_assets_per_date=max_per_date,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
        )

    if series is None:
        return _short_circuit_output(
            "k_spread",
            "insufficient_assets_for_k_legs",
            n_obs=0,
            n_obs_axis="periods",
            k=k,
            min_required=2 * k,
            max_assets_per_date=max_per_date,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
        )

    spread_vals = _finite_values(series["spread"])
    n_strided = len(spread_vals)
    sc = _enforce_scaled_floor(
        "k_spread",
        data["date"].n_unique(),
        MIN_PORTFOLIO_PERIODS_HARD,
        overlap_periods,
        "insufficient_portfolio_periods",
        k=k,
        tie_ratio=tie_ratio,
        tie_policy=tie_policy,
    )
    if sc is not None:
        return sc
    # ``_build_k_spread_series`` forces spread=0 for a constant factor, so a
    # degenerate panel is detected once here and returned as no-signal.
    if _all_dates_degenerate(clean, factor_col):
        return _no_signal_zero_variance(
            n_strided,
            k=k,
            cross_sectional_dispersion=_finite_mean(series["xs_dispersion"]),
            top_return=_finite_mean(series["top_return"]),
            bottom_return=_finite_mean(series["bottom_return"]),
        )
    if n_strided == 0:
        return _short_circuit_output(
            "k_spread",
            "insufficient_portfolio_periods",
            n_obs=0,
            n_obs_axis="periods",
            k=k,
            n_periods_in=series.height,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
        )

    strided_series = series.select("date", pl.col("spread").cast(pl.Float64)).filter(
        _finite_expr("spread")
    )
    # A member that consumes the full overlapping series needs every period;
    # build it once on the unsampled panel. Non-finite spreads are filtered out
    # before it for the same reason the strided series is cleaned.
    full_series: pl.DataFrame | None = None
    if inference.consumes_full_series:
        full_series, _ = _build_k_spread_series(
            data, k, factor_col, return_col, tie_policy
        )
        if full_series is not None:
            full_series = full_series.filter(_finite_expr("spread"))
    # Thin-cross-section switch keys on the median per-period count of usable
    # names, not the universe-wide unique asset count (see
    # ``_median_per_date_count``).
    median_xs = _median_per_date_count(clean)
    mean_spread, t, p, sig_method, sig_stat_type, sig_extra, sig_codes = (
        _spread_significance_with_inference(
            inference,
            strided_spread=strided_series,
            full_spread=full_series,
            overlap_periods=overlap_periods,
            n_assets=median_xs,
        )
    )
    # Sample the headline stat/p actually ran on: full overlapping series under
    # HAC, strided otherwise. ``value``/``stat``/``p_value``/``n_obs`` all
    # describe it.
    n = int(
        cast(
            int,
            sig_extra.get(
                "n_periods_tested", sig_extra.get("n_periods_full", n_strided)
            ),
        )
    )

    mean_dispersion = _finite_mean(series["xs_dispersion"])
    mean_top = _finite_mean(series["top_return"])
    mean_bottom = _finite_mean(series["bottom_return"])

    metadata: dict[str, object] = {
        "n_periods": n,
        "n_periods_strided": n_strided,
        "median_cross_section": median_xs,
        "k": k,
        "tie_ratio": tie_ratio,
        "tie_policy": tie_policy,
        "stat_type": sig_stat_type,
        "h0": "mu=0",
        "method": sig_method,
        "cross_sectional_dispersion": mean_dispersion,
        "top_return": mean_top,
        "bottom_return": mean_bottom,
        **sig_extra,
    }
    warning_codes = list(sig_codes)
    if high_tie_ratio:
        warning_codes.append(WarningCode.HIGH_TIE_RATIO.value)
    # Drop stats describe the strided series this consumer collapsed, whatever
    # sample the headline test ended up running on.
    _surface_null_drop(
        n_periods_in=series.height,
        n_periods_out=n_strided,
        drop_reason="null / NaN value observations in the series",
        metric_name="k_spread",
        metadata=metadata,
        warning_codes=warning_codes,
        expected_warnings=expected_warnings,
    )
    # A NaN headline stat means the tested spread series carries no dispersion
    # (or the HAC SE collapsed): ``mean_spread`` still stands, the t does not.
    stat, p_out, alternative = _degenerate_test_fields(
        t, p, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        value=mean_spread,
        p_value=p_out,
        alternative=alternative,
        n_obs=n,
        n_obs_axis="periods",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
