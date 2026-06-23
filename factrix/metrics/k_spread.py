"""Fixed-K Top-K vs Bottom-K long-short spread for small cross-sections.

Notes:
    **Pipeline.** Per non-overlapping date, select the top ``k`` and
    bottom ``k`` names by factor rank, take the mean-return difference
    (cross-section step), then test the per-date spread series across
    time. Small cross-sections switch the headline test to a
    block-bootstrap CI.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.

    **Output.** Mean spread, with the per-date cross-sectional return
    dispersion reported alongside.

    The small-N counterpart of
    :func:`~factrix.metrics.quantile.quantile_spread`. Quantile bucketing
    (``n_groups=5`` ⇒ quintiles) degrades when ``N < 30``: each bucket
    holds only a handful of names, so the spread is dominated by
    individual assets and the quintile breakpoints are unstable. Fixing
    the **count** ``k`` per leg keeps each leg's composition stable
    regardless of ``N``, and the metric reports the contemporaneous
    cross-sectional dispersion so the spread can be read relative to the
    typical spread of returns that period.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import DDOF, MIN_PORTFOLIO_PERIODS_HARD
from factrix.inference import NON_OVERLAPPING, NeweyWest, NonOverlapping
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _enforce_min_floor,
    _sample_non_overlapping,
    _short_circuit_output,
    _spread_significance_with_inference,
    _surface_null_drop,
)

__all__ = [
    "k_spread",
]


def _build_k_spread_series(
    panel: pl.DataFrame, k: int, factor_col: str, return_col: str
) -> tuple[pl.DataFrame | None, pl.DataFrame]:
    """Per-date Top-K/Bottom-K spread series from a (possibly sampled) panel.

    Returns ``(series, clean)``: ``series`` has ``date, top_return,
    bottom_return, xs_dispersion, spread`` (``None`` when no date clears the
    ``2*k`` floor), and ``clean`` is the null-filtered panel for the
    short-circuit diagnostics / ``n_assets`` count. Shared by the
    non-overlap path (sampled panel) and the HAC path (full panel).
    """
    clean = panel.filter(
        pl.col(factor_col).is_not_null() & pl.col(return_col).is_not_null()
    )
    ranked = clean.with_columns(
        pl.col(factor_col)
        .rank(method="ordinal", descending=True)
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
    sample_threshold=SampleThreshold(min_periods=MIN_PORTFOLIO_PERIODS_HARD),
)
def k_spread(
    df: pl.DataFrame,
    forward_periods: int = 5,
    k: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    rng_seed: int = 0,
    inference: NonOverlapping | NeweyWest = NON_OVERLAPPING,
) -> MetricResult:
    r"""Fixed-K Top-K vs Bottom-K long-short spread.

    Per non-overlapping date, the long leg is the mean forward return of
    the ``k`` highest-factor names and the short leg the mean of the
    ``k`` lowest; the spread is their difference. The mean spread is
    tested across time.

    Args:
        df: Panel with ``date, asset_id``, ``factor_col`` and
            ``return_col``.
        forward_periods: Sampling stride for non-overlapping dates;
            match the forward-return horizon.
        k: Number of names per leg (fixed count, not a quantile
            fraction). A date needs at least ``2 * k`` names to form
            disjoint legs — dates with fewer are dropped.
        factor_col: Ranking column (default ``"factor"``).
        return_col: Realised-return column (default ``"forward_return"``).
        rng_seed: Seed for the small-N block-bootstrap branch
            (reproducible by default).
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
        mean per-date cross-sectional standard deviation of returns.

    Notes:
        Per qualifying date $t$ (universe size $N_t \geq 2k$), with
        $\mathrm{top}_k$ / $\mathrm{bot}_k$ the names ranked $1..k$ /
        $N_t-k+1..N_t$ by factor:

        $$\text{spread}_t = \frac1k \sum_{i \in \mathrm{top}_k} r_{i,t}
        - \frac1k \sum_{i \in \mathrm{bot}_k} r_{i,t}.$$

        ``value = mean_t spread_t``. The headline test follows the shared
        small-cross-section policy: with ``n_assets < MIN_ASSETS_WARN``
        the per-date spread is heavy-tailed (few names per leg), so the
        ``t``-test is replaced by a block-bootstrap CI; otherwise the
        non-overlapping ``t`` applies. ``metadata["method"]`` records
        which ran. The contemporaneous cross-sectional dispersion
        $\mathrm{std}_i(r_{i,t})$ is averaged over dates and reported so
        the spread can be judged against the period's return spread.

    References:
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: overlapping-return
        autocorrelation, motivating the non-overlap stride.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.k_spread import k_spread
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=20, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = k_spread(panel, forward_periods=5, k=3)
        >>> result.name == ""
        True
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if return_col not in df.columns:
        return _short_circuit_output(
            "k_spread",
            "no_return_column",
            missing_column=return_col,
        )

    # Drop rows with no factor or no realised return BEFORE ranking: ``rank``
    # skips nulls but ``pl.len`` would still count them, so ``_n_date`` would
    # overcount and the bottom-leg cutoff (``_rank > _n_date - k``) would point
    # past the last real rank — silently shrinking or emptying the short leg.
    # forward_return is null on the last ``forward_periods`` rows per asset.
    sampled = _sample_non_overlapping(df, forward_periods)
    series, clean = _build_k_spread_series(sampled, k, factor_col, return_col)

    if series is None:
        per_date_counts = clean.group_by("date").len()["len"].to_numpy()
        max_per_date = int(np.max(per_date_counts)) if per_date_counts.size else 0
        return _short_circuit_output(
            "k_spread",
            "insufficient_assets_for_k_legs",
            n_obs=0,
            k=k,
            min_required=2 * k,
            max_assets_per_date=max_per_date,
        )

    spread_vals = series["spread"].drop_nulls()
    n = len(spread_vals)
    sc = _enforce_min_floor(
        k_spread, "k_spread", n, "insufficient_portfolio_periods", k=k
    )
    if sc is not None:
        return sc

    arr = spread_vals.to_numpy()
    n_assets = clean["asset_id"].n_unique()
    # The HAC path needs the full overlapping spread series (every date);
    # build it once on the unsampled panel.
    full_series: pl.DataFrame | None = None
    if isinstance(inference, NeweyWest):
        full_series, _ = _build_k_spread_series(df, k, factor_col, return_col)
    mean_spread, t, p, sig_method, sig_extra, sig_codes = (
        _spread_significance_with_inference(
            inference,
            strided_spread=arr,
            full_spread=full_series,
            forward_periods=forward_periods,
            n_assets=n_assets,
            rng_seed=rng_seed,
        )
    )

    mean_dispersion = float(np.mean(series["xs_dispersion"].drop_nulls().to_numpy()))
    mean_top = float(np.mean(series["top_return"].drop_nulls().to_numpy()))
    mean_bottom = float(np.mean(series["bottom_return"].drop_nulls().to_numpy()))

    metadata: dict[str, object] = {
        "n_periods": n,
        "k": k,
        "stat_type": "t",
        "h0": "mu=0",
        "method": sig_method,
        "cross_sectional_dispersion": mean_dispersion,
        "top_return": mean_top,
        "bottom_return": mean_bottom,
        **sig_extra,
    }
    warning_codes = list(sig_codes)
    _surface_null_drop(
        n_periods_in=series.height,
        n_periods_out=n,
        drop_reason="null spread observations in the series",
        metric_name="k_spread",
        metadata=metadata,
        warning_codes=warning_codes,
    )
    return MetricResult(
        value=mean_spread,
        p_value=p,
        n_obs=n,
        stat=t,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
