"""Preprocessing Step 1-3: forward return computation and adjustment.

Step 1 — Forward Return: (price[t+1+forward_periods] / price[t+1] - 1) / forward_periods
Step 2 — Winsorize Forward Return: per-date percentile clip
Step 3 — Abnormal Return: forward_return - cross-sectional mean

All functions expect canonical column names (date, asset_id, price).
Use ``adapt()`` to rename before calling.
"""

import warnings
from collections.abc import Iterable

import numpy as np
import polars as pl

from factrix._codes import WarningCode
from factrix._errors import UserInputError

_DOCS_FORWARD_RETURN = "api/preprocess#factrix.preprocess.compute_forward_return"
_DOCS_EVALUATION_GRID = "api/preprocess#evaluating-on-a-coarser-grid"
_DOCS_WINSORIZE_FORWARD_RETURN = (
    "api/preprocess#factrix.preprocess.winsorize_forward_return"
)


def _validate_forward_periods(forward_periods: object) -> int:
    if not isinstance(forward_periods, int) or isinstance(forward_periods, bool):
        raise UserInputError(
            func_name="compute_forward_return",
            field="forward_periods",
            value=forward_periods,
            expected="a positive int row horizon, e.g. 5",
            docs_path=_DOCS_FORWARD_RETURN,
        )
    if forward_periods <= 0:
        raise UserInputError(
            func_name="compute_forward_return",
            field="forward_periods",
            value=forward_periods,
            expected="a positive int row horizon (> 0)",
            docs_path=_DOCS_FORWARD_RETURN,
        )
    return forward_periods


def _validate_winsorize_bounds(lower: object, upper: object) -> tuple[float, float]:
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, int | float)
        or not isinstance(upper, int | float)
    ):
        raise UserInputError(
            func_name="winsorize_forward_return",
            field="bounds",
            value={"lower": lower, "upper": upper},
            expected="numeric quantile bounds satisfying 0 <= lower <= upper <= 1",
            docs_path=_DOCS_WINSORIZE_FORWARD_RETURN,
        )
    lower_f = float(lower)
    upper_f = float(upper)
    if not 0.0 <= lower_f <= upper_f <= 1.0:
        raise UserInputError(
            func_name="winsorize_forward_return",
            field="bounds",
            value={"lower": lower, "upper": upper},
            expected="0 <= lower <= upper <= 1",
            docs_path=_DOCS_WINSORIZE_FORWARD_RETURN,
        )
    return lower_f, upper_f


def _warn_if_ragged(indexed: pl.DataFrame, n_periods: int) -> None:
    """Flag per-asset date grids that disagree with the panel's own grid.

    The horizon is a step along each asset's period index, so a gap is no
    longer able to stretch one asset's window past another's — but the asset
    with the gap has no observation to pair at ``i + 1 + h`` and simply
    contributes fewer rows. Callers comparing horizons across names need to
    know their grids differ.
    """
    per_asset = indexed.group_by("asset_id").agg(
        pl.col("_period_index").n_unique().alias("_n_periods")
    )
    n_ragged = int((per_asset["_n_periods"] < n_periods).sum())
    if n_ragged:
        warnings.warn(
            f"compute_forward_return: {WarningCode.RAGGED_PERIOD_GRID.value} — "
            f"{n_ragged} of {per_asset.height} assets are missing periods that "
            f"others have (panel grid: {n_periods} periods). The horizon is "
            "measured on the panel's period grid, so those assets simply have "
            "no observation to pair at the exit period rather than a stretched "
            "window; reindex onto a common grid if the horizons must be "
            "comparable across names.",
            UserWarning,
            stacklevel=3,
        )


def _warn_if_uneven(kept_index: list[int]) -> None:
    """Flag a caller-chosen evaluation grid with a non-constant stride.

    ``dates=`` may name any subset of the panel's period grid, and nothing
    downstream resamples it. The series-mean paths are calibrated on an
    unevenly spaced grid; several other paths read a constant spacing into
    the grid they are handed, so the caller has to know which one it passed.
    """
    strides = np.unique(np.diff(np.asarray(sorted(kept_index), dtype=np.int64)))
    if strides.size > 1:
        warnings.warn(
            f"compute_forward_return: {WarningCode.UNEVEN_EVALUATION_GRID.value} "
            f"— the evaluation grid passed as dates= has {strides.size} distinct "
            f"spacings between adjacent kept rows ({int(strides.min())} to "
            f"{int(strides.max())} periods). Series-mean inference "
            "(NonOverlapping, NeweyWest) is calibrated on such a grid; the "
            "regression HAC tests (predictive_beta, spanning_alpha, the "
            "K-restriction Wald / slice period joint tests), the ADF / "
            "autocorrelation persistence screens, event-study estimation "
            "windows and offsets, and adjacent-period metrics (turnover / rank "
            "autocorrelation, rolling windows) assume a constant spacing — read "
            "their results on this grid with that in mind. The remedy is on "
            "your side: pass dates= at a constant stride on the panel's period "
            "grid if those paths must be calibrated; factrix does not resample. "
            "Separately, on an uneven grid overlap_periods is the maximum "
            "overlap, so NonOverlapping strides at that maximum and discards "
            "more of the series than it would on a constant-stride grid, while "
            "NeweyWest's 1.3*sqrt(T) base bandwidth is insensitive to the "
            "unevenness (measured 5.5% at a nominal 5%) and keeps the full "
            "series. That is sample efficiency only: switching inference= does "
            "not recalibrate the paths named above.",
            UserWarning,
            stacklevel=3,
        )


def _resolve_evaluation_grid(
    grid: pl.DataFrame, dates: object
) -> tuple[pl.Series, list[int]]:
    """Map caller-chosen evaluation dates onto the panel's period grid.

    Every value must be a member of the panel's distinct-date grid — nothing
    is snapped. Snapping backward would evaluate a factor observed after the
    entry period (look-ahead); snapping forward would silently move the entry
    period. Returns the matched dates (in the grid's dtype) and their sorted
    period indices on the *full* grid.
    """
    if isinstance(dates, pl.Series):
        wanted = dates
    elif isinstance(dates, str | bytes) or not isinstance(dates, Iterable):
        raise UserInputError(
            func_name="compute_forward_return",
            field="dates",
            value=type(dates).__name__,
            expected="a polars Series or a sequence of dates on the panel's grid",
            docs_path=_DOCS_EVALUATION_GRID,
        )
    else:
        wanted = pl.Series("date", list(dates))
    if wanted.len() == 0:
        raise UserInputError(
            func_name="compute_forward_return",
            field="dates",
            value=[],
            expected="at least one evaluation date on the panel's grid",
            docs_path=_DOCS_EVALUATION_GRID,
        )
    grid_dtype = grid.schema["date"]
    if wanted.dtype != grid_dtype:
        # A Date value names a Datetime period unambiguously (its midnight),
        # and a Datetime in another time unit re-expresses the same instant
        # (a Python datetime list arrives as microseconds; membership is
        # still checked exactly after the cast). Any other mismatch —
        # Datetime -> Date would truncate, i.e. snap; a String would be
        # parsed — is rejected rather than coerced.
        same_zone_datetime = (
            isinstance(wanted.dtype, pl.Datetime)
            and isinstance(grid_dtype, pl.Datetime)
            and wanted.dtype.time_zone == grid_dtype.time_zone
        )
        if same_zone_datetime or (
            wanted.dtype == pl.Date and isinstance(grid_dtype, pl.Datetime)
        ):
            wanted = wanted.cast(grid_dtype)
        else:
            raise UserInputError(
                func_name="compute_forward_return",
                field="dates",
                value=str(wanted.dtype),
                expected=(
                    f"dates with the panel's date dtype ({grid_dtype}); values "
                    "are matched exactly against the panel's grid, never "
                    "parsed or snapped"
                ),
                docs_path=_DOCS_EVALUATION_GRID,
            )
    wanted = wanted.unique().alias("date")
    on_grid = wanted.is_in(grid["date"].implode())
    if not on_grid.all():
        missing = wanted.filter(~on_grid).sort().to_list()
        raise UserInputError(
            func_name="compute_forward_return",
            field="dates",
            value=missing[:5],
            expected=(
                f"every evaluation date to be a member of the panel's distinct-"
                f"date grid ({len(missing)} are not). Dates are matched "
                "exactly and never snapped: snapping backward would read a "
                "factor observed after the entry period, snapping forward "
                "would move the entry period. Pick the evaluation dates from "
                "the panel's own dates."
            ),
            docs_path=_DOCS_EVALUATION_GRID,
        )
    matched = grid.join(wanted.to_frame(), on="date", how="inner").sort("_period_index")
    return matched["date"], matched["_period_index"].to_list()


def _overlap_on_grid(kept_index: list[int], forward_periods: int) -> int:
    """Overlap of adjacent observations on an evaluation grid.

    A row at period index ``i`` carries the return over ``(i + 1, i + 1 + h]``
    on the full grid, so two kept rows ``i < j`` overlap exactly when
    ``j - i < h``. The overlap statistic is::

        overlap_periods = 1 + max_i #{ j kept : 0 < j - i < h }

    i.e. one more than the largest number of later kept rows any kept row
    shares future periods with. On the full grid this is ``h``; at a constant
    stride ``s >= h`` it is 1; at stride ``s < h`` it is ``ceil(h / s)``.

    ``max`` rather than a typical (median) count is deliberate. Under uneven
    spacing on the period grid, a count that under-states the true overlap
    by one leaves an MA term the non-overlapping stride does not remove, and
    the stride t-test over-rejects (measured 12-13% at a nominal 5%); a count
    that over-states it by one only thins the strided sample (size intact,
    some power lost). The Newey-West path is insensitive either way — its
    bandwidth is dominated by the ``1.3 * sqrt(T)`` base.
    """
    idx = np.asarray(sorted(kept_index), dtype=np.int64)
    if idx.size == 0:
        return 1
    # #{j : 0 < idx_j - idx_i < h} == searchsorted(idx, idx_i + h) - (i + 1)
    upper = np.searchsorted(idx, idx + forward_periods, side="left")
    counts = upper - (np.arange(idx.size) + 1)
    return int(1 + counts.max())


def compute_forward_return(
    data: pl.DataFrame,
    forward_periods: int = 5,
    *,
    dates: pl.Series | Iterable[object] | None = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Step 1: Compute per-period forward return per asset.

    ``forward_return = (price[t+1+forward_periods] / price[t+1] - 1) / forward_periods``

    Entry at t+1 (next bar after density), exit at t+1+forward_periods.

    WHY t+1 entry: The density at t is computed using data up to and
    including price[t]. Using price[t] as both density input and entry
    price assumes you can trade at the same price used to generate the
    density — unrealistic in practice. Entry at t+1 enforces a strict
    causal boundary: density → wait → trade → measure.

    This also keeps the return window cleanly separated from the
    estimation window in event studies (BMP test), eliminating the
    need for ad-hoc shift corrections.

    Dividing by ``forward_periods`` normalizes returns to a per-period basis, making
    different forward_periods directly comparable on a scale basis
    (see Notes for the scope boundary).

    Args:
        data: Must contain ``date``, ``asset_id``, ``price``, with one row per
            ``(date, asset_id)`` and a temporal ``date`` column. Both are
            enforced (see Raises); neither used to be.
        forward_periods: Holding horizon in **periods of the panel's own
            grid** — the distinct sorted ``date`` values present in the panel,
            not calendar time and not row position within an asset (default
            5). On a daily panel this is 5 trading days; on a weekly panel, 5
            weeks; on 1-min bars, 5 minutes. Which frequency those periods
            represent is the caller's responsibility.
        dates: Optional evaluation grid — the dates to keep, as a polars
            ``Series`` or a sequence. Every value must be a member of the
            panel's distinct-date grid (matched exactly, never snapped; see
            Raises). The return is still computed on the full grid at
            ``forward_periods``; only rows on these dates are returned, and
            the ``overlap_periods`` stamp is *derived* from their spacing on
            the full period grid (see Notes) instead of being set to the
            horizon. ``None`` (default) keeps every date and stamps
            ``overlap_periods = forward_periods``.
        overwrite: Allow recomputation when ``data`` already carries a
            ``forward_return`` column. ``False`` (default) raises rather
            than silently overwrite — the function is **not idempotent**:
            the previous call already dropped the last ``forward_periods + 1``
            rows per asset, so recomputing on the result drops a *further*
            tail. To change the horizon, recompute from the original
            (pre-forward-return) panel; ``overwrite=True`` recomputes in
            place anyway, accepting the additional truncation.

    Raises:
        UserInputError: ``forward_periods`` is not a positive ``int``;
            ``data`` carries duplicate ``(date, asset_id)`` rows or a
            non-temporal ``date`` column; ``data`` already has a
            ``forward_return`` column and ``overwrite`` is ``False``; the
            horizon / price data leaves no finite forward returns after
            filtering; or ``dates`` is empty, has a dtype the panel's
            ``date`` cannot match exactly, or contains a value that is not on
            the panel's distinct-date grid.

    Returns:
        Input DataFrame with ``forward_return`` column appended and two
        reserved stamp columns — ``_forward_periods`` (the return horizon,
        the hypothesis) and ``_overlap_periods`` (the overlap of adjacent
        observations on the evaluation grid, the quantity inference consumes;
        equal to the horizon unless ``dates`` was given). They are the
        single source of truth ``factrix.evaluate`` reads (it strips both
        before dispatch, so they never reach a metric or ``to_frame``). Rows
        where forward return is not finite (tail nulls, NaN, +inf, -inf) are
        dropped; with ``dates``, rows off the evaluation grid are dropped
        too.

    Notes:
        **The horizon is measured on the panel's period grid.** The distinct
        sorted ``date`` values in the panel are indexed 0, 1, 2, ..., and each
        asset's forward return pairs its row at period index ``i + 1`` with its
        row at ``i + 1 + forward_periods``. This used to be a positional
        ``shift`` within each asset, which equals a time horizon only on a
        complete per-asset panel: with asset A missing 20 periods mid-sample, a
        "5-period" return silently spanned 25 real periods, contaminating both
        the return and the overlap horizon stamped in ``_forward_periods`` that
        every downstream HAC inference reads. Suspensions, halts,
        delist-relist and staggered entry are ordinary in regional equity data,
        and sparse event panels are ragged by construction.

        A ragged grid (an asset missing periods that others have) raises
        ``WarningCode.RAGGED_PERIOD_GRID``: the pairing is now correct for
        every asset, but an asset with a gap simply has no observation at
        ``i + 1 + h``, so it contributes fewer rows than a complete one.
        Reindex onto a common grid if the horizons must be comparable across
        names.

        **Evaluating on a coarser grid.** A factor is often observed on a
        finer grid than it is traded on — the price grid supplies the return
        horizon, while the evaluation happens on a caller-chosen rebalance
        grid. Two separate quantities then matter. ``forward_periods`` is the
        economic horizon and names the hypothesis; it is the same on any
        grid. ``overlap_periods`` is how many adjacent *evaluation*
        observations share future periods, which is what the HAC bandwidth
        and effective degrees of freedom, the non-overlapping stride and the
        stride-scaled sample floors consume. On the full grid the two
        coincide, so a panel sub-sampled by hand *after* this function keeps
        a stale ``overlap_periods = forward_periods`` stamp: a 60-period
        return evaluated every 60 periods is treated as 60-fold overlapping
        and the stride-scaled floor rejects it outright. Pass the evaluation
        grid as ``dates`` instead and the overlap is derived here, on the
        full grid, as::

            overlap_periods = 1 + max_i #{ j in dates : 0 < idx(j) - idx(i) < h }

        with ``idx`` the period index on the panel's *full* distinct-date
        grid. The row at ``i`` covers ``(i + 1, i + 1 + h]``, so two kept
        rows overlap exactly when they are fewer than ``h`` periods apart.
        Stride 60 at ``h = 60`` gives 1; stride 20 gives 3; the full grid
        gives ``h``. The maximum, not a typical count, is used because the
        evaluation grid may be spaced unevenly on the period grid:
        under-stating the overlap by one leaves dependence the stride does
        not remove and the stride t-test over-rejects (12-13% at a nominal
        5% in simulation), while over-stating it by one only thins the
        strided sample. The derivation must run here rather than on the
        returned panel: the ``is_finite`` filter below removes any period on
        which every asset's return is non-finite, so an index rebuilt from
        the output's own dates would compress the gaps and under-count.

        The unit of ``forward_return`` does not change with the grid: it is
        the return per period **of the horizon** (the ``/ forward_periods``
        below), not per evaluation period. Every p-value on the affected
        paths is scale-invariant, and a per-evaluation-period rate is not
        defined on an unevenly spaced grid.

        **Non-finite prices are blanked before the division, not after.** The
        filter used to be applied to the *quotient*: with a ``+Inf``
        denominator, ``finite / inf`` is ``0.0``, so the result was a perfectly
        finite fabricated ``-100%`` return that sailed straight through
        ``is_finite()``. (An ``inf`` numerator gives ``inf`` and was correctly
        dropped — the leak was asymmetric, and so easy to miss.)

        The ``/ forward_periods`` per-period normalization is a *scale* choice with
        three caveats the caller should know:

        1. **Arithmetic, not summed-log-return.** This is the
           arithmetic per-period mean of a simple return, not the
           academic-standard direct long-horizon regression of summed
           log returns on the predictor (the latter is
           linear-additive across horizons by construction).
        2. **Compounding bias.** Compounding at the arithmetic mean
           is an upward-biased estimator of cumulative wealth; the
           bias grows with ``forward_periods`` and per-bar return variance.
           Negligible for rank-based information coefficient (IC); not negligible for
           signed-return mean and t-tests at large ``forward_periods``.
        3. **Scale, not inference.** ``/ forward_periods`` aligns the *scale* across
           horizons — it does *not* address the inference problem.
           Overlap is handled by heteroskedasticity-and-autocorrelation-consistent (HAC) (see
           :class:`factrix.inference.NeweyWest`); across-horizon
           selection is handled by a declared multiple-testing family in
           :func:`factrix.multi_factor.bhy` (BHY controls FDR, not FWER). The three concerns
           (scale, overlap, cross-horizon selection) are addressed
           at separate layers; overlap and across-horizon dependence
           share a common source in the persistent regressor, but
           each requires its own tool.

    References:
        - [Fama & French (1988)][fama-french-1988]. "Dividend Yields
          and Expected Stock Returns." Journal of Financial
          Economics, 22(1), 3–25. Direct summed-log-return
          long-horizon regression — the academic-standard
          alternative to factrix's ``÷N``.
        - [Jacquier, Kane & Marcus (2003)][jacquier-kane-marcus-2003].
          "Geometric or Arithmetic Mean: A Reconsideration."
          Financial Analysts Journal, 59(6), 46–53. Compounding bias
          of the arithmetic mean and the unbiased horizon-weighted
          blend.
        - [Boudoukh, Richardson & Whitelaw (2008)][boudoukh-richardson-whitelaw-2008].
          "The Myth of Long-Horizon Predictability." Review of
          Financial Studies, 21(4), 1577–1605. Documents that
          across-horizon regression statistics share information
          through the persistent regressor — separate from any
          per-period scaling choice, and the reason inference across
          horizons is not addressed by normalization.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> "forward_return" in panel.columns
        True
        >>> panel["forward_return"].null_count() == 0
        True

        The output panel is the canonical input to ``fx.evaluate``:

        >>> from factrix.metrics import ic
        >>> results = fx.evaluate(
        ...     panel, metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
        ... )
        >>> isinstance(results, dict) and "factor" in results
        True

        A 60-period return evaluated every 60 periods: the horizon stays 60
        (the hypothesis), while the derived overlap is 1, so inference treats
        the sampled observations as non-overlapping:

        >>> raw = fx.datasets.make_cs_panel(n_assets=30, n_dates=1500)
        >>> grid = raw["date"].unique().sort()
        >>> sampled = compute_forward_return(
        ...     raw, forward_periods=60, dates=grid.gather_every(60)
        ... )
        >>> sampled["date"].n_unique()
        24
        >>> result = fx.evaluate(sampled, metrics={"ic": ic()}, factor_cols=["factor"])
        >>> (result["factor"].forward_periods, result["factor"].overlap_periods)
        (60, 1)
    """
    forward_periods = _validate_forward_periods(forward_periods)

    if "forward_return" in data.columns:
        if not overwrite:
            raise UserInputError(
                func_name="compute_forward_return",
                field="data",
                value=list(data.columns),
                expected=(
                    "a panel without 'forward_return'. This function is not "
                    "idempotent — a prior call already dropped the last "
                    "forward_periods+1 rows per asset, so recomputing drops a "
                    "further tail and silently shrinks the data. To change the "
                    "horizon, recompute from the original (pre-forward-return) "
                    "panel; pass overwrite=True to recompute in place anyway."
                ),
                docs_path=_DOCS_FORWARD_RETURN,
            )
        data = data.drop("forward_return")

    from factrix._data_input import _normalize_panel, _stamp_horizons

    # One structural gate: temporal ``date``, unique ``(date, asset_id)``, and
    # non-finite numerics blanked to null *before* any arithmetic. A duplicated
    # key made the "next period" the same date's twin and manufactured a 0.0
    # return; a +Inf denominator made ``finite / inf = 0`` and manufactured a
    # finite -100% return that an output-side is_finite() filter cannot catch.
    data = _normalize_panel(data)

    # Period index on the panel's own grid, shared by every asset. Shifting by
    # row position within an asset only equals a time horizon on a complete
    # panel — see Notes.
    grid = (
        data.select("date")
        .unique()
        .sort("date")
        .with_row_index("_period_index")
        .with_columns(pl.col("_period_index").cast(pl.Int64))
    )
    n_periods = grid.height
    indexed = data.join(grid, on="date", how="left")

    # The evaluation grid is resolved against the FULL period grid, before the
    # is_finite filter below can remove a period on which every asset's return
    # is non-finite — an index rebuilt from the output's own dates would
    # compress those gaps and under-count the overlap.
    evaluation_grid: tuple[pl.Series, list[int]] | None = (
        _resolve_evaluation_grid(grid, dates) if dates is not None else None
    )

    _warn_if_ragged(indexed, n_periods)

    # Entry at period i+1, exit at period i+1+h, both looked up by index rather
    # than by row offset, so a gap in one asset's history cannot stretch its
    # horizon.
    prices = indexed.select(
        "asset_id",
        pl.col("_period_index"),
        pl.col("price").alias("_price_at"),
    )
    out = (
        indexed.with_columns(
            (pl.col("_period_index") + 1).alias("_entry_index"),
            (pl.col("_period_index") + 1 + forward_periods).alias("_exit_index"),
        )
        .join(
            prices.rename({"_period_index": "_entry_index", "_price_at": "_entry"}),
            on=["asset_id", "_entry_index"],
            how="left",
        )
        .join(
            prices.rename({"_period_index": "_exit_index", "_price_at": "_exit"}),
            on=["asset_id", "_exit_index"],
            how="left",
        )
        .with_columns(
            ((pl.col("_exit") / pl.col("_entry") - 1) / forward_periods).alias(
                "forward_return"
            )
        )
        .drop("_period_index", "_entry_index", "_exit_index", "_entry", "_exit")
        .filter(pl.col("forward_return").is_finite())
    )
    overlap_periods = forward_periods
    if evaluation_grid is not None:
        kept_dates, kept_index = evaluation_grid
        out = out.filter(pl.col("date").is_in(kept_dates.implode()))
        overlap_periods = _overlap_on_grid(kept_index, forward_periods)
        _warn_if_uneven(kept_index)
    if out.is_empty():
        raise UserInputError(
            func_name="compute_forward_return",
            field="data",
            value=f"{data.height} rows",
            expected=(
                "at least one finite forward_return after applying the row "
                f"horizon forward_periods={forward_periods}; the panel may be "
                "too short, price contains only non-finite returns, or every "
                "evaluation date in dates= falls inside the horizon's tail"
            ),
            docs_path=_DOCS_FORWARD_RETURN,
        )
    # Stamp both horizons as the single source of truth for the data; evaluate
    # reads them instead of taking them at the metric / call layer (the layers
    # could silently diverge — see compute_forward_return docs).
    return _stamp_horizons(
        out, forward_periods=forward_periods, overlap_periods=overlap_periods
    )


def winsorize_forward_return(
    data: pl.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """Step 2: Per-date percentile clip on forward returns.

    Args:
        lower: Lower quantile bound (default 0.01 = 1st percentile).
            Must satisfy ``0 <= lower <= upper <= 1``.
        upper: Upper quantile bound (default 0.99 = 99th percentile).
            Must satisfy ``0 <= lower <= upper <= 1``. Set to
            ``(0.0, 1.0)`` to disable.

    Raises:
        UserInputError: ``lower`` / ``upper`` are not numeric quantile
            bounds satisfying ``0 <= lower <= upper <= 1``.

    Returns:
        DataFrame with ``forward_return`` clipped in-place.

    Notes:
        **Quantile interpolation.** The per-date bounds are computed with
        ``interpolation="linear"`` — the numpy / pandas / alphalens default —
        rather than polars' own default of ``"nearest"``. With ``"nearest"``
        the bound snaps to an actually observed value, so on a small
        cross-section the winsorisation can be a complete no-op: for
        ``[0, 1, ..., 8, 100]`` at ``upper=0.95`` the nearest-rank quantile is
        ``100`` itself and the outlier survives untouched, while the linear
        quantile is ``58.6`` and clips it. Linear interpolation makes the clip
        level a continuous function of the requested percentile, which is what
        a per-date winsoriser needs in order to behave consistently as the
        cross-section size changes.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import (
        ...     compute_forward_return,
        ...     winsorize_forward_return,
        ... )
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> clipped = winsorize_forward_return(panel, lower=0.01, upper=0.99)
        >>> clipped.height == panel.height
        True
        >>> clipped["forward_return"].max() <= panel["forward_return"].max()
        True
    """
    lower, upper = _validate_winsorize_bounds(lower, upper)
    if lower <= 0.0 and upper >= 1.0:
        return data

    # WHY: interpolation="linear" matches numpy / pandas / alphalens. polars
    # defaults to "nearest", which on small cross-sections snaps the bound to
    # an existing observation and can fail to clip at all (see Notes).
    lb = pl.col("forward_return").quantile(lower, interpolation="linear").over("date")
    ub = pl.col("forward_return").quantile(upper, interpolation="linear").over("date")

    return data.with_columns(
        pl.col("forward_return").clip(lb, ub).alias("forward_return")
    )


def compute_abnormal_return(data: pl.DataFrame) -> pl.DataFrame:
    """Step 3: Cross-sectional abnormal return.

    ``abnormal_return = forward_return - mean(forward_return) per date``

    Returns:
        DataFrame with ``abnormal_return`` column appended.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import (
        ...     compute_abnormal_return,
        ...     compute_forward_return,
        ... )
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> adjusted = compute_abnormal_return(panel)
        >>> "abnormal_return" in adjusted.columns
        True
    """
    return data.with_columns(
        (pl.col("forward_return") - pl.col("forward_return").mean().over("date")).alias(
            "abnormal_return"
        )
    )
