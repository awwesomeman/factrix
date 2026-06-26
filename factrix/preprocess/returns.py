"""Preprocessing Step 1-3: forward return computation and adjustment.

Step 1 — Forward Return: (price[t+1+N] / price[t+1] - 1) / N
Step 2 — Winsorize Forward Return: per-date percentile clip
Step 3 — Abnormal Return: forward_return - cross-sectional mean

All functions expect canonical column names (date, asset_id, price).
Use ``adapt()`` to rename before calling.
"""

import polars as pl

from factrix._errors import UserInputError

_DOCS_FORWARD_RETURN = "api/preprocess#compute_forward_return"
_DOCS_WINSORIZE_FORWARD_RETURN = "api/preprocess#winsorize_forward_return"


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


def compute_forward_return(
    data: pl.DataFrame,
    forward_periods: int = 5,
    *,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Step 1: Compute per-period forward return per asset.

    ``forward_return = (price[t+1+N] / price[t+1] - 1) / N``

    Entry at t+1 (next bar after density), exit at t+1+N.

    WHY t+1 entry: The density at t is computed using data up to and
    including price[t]. Using price[t] as both density input and entry
    price assumes you can trade at the same price used to generate the
    density — unrealistic in practice. Entry at t+1 enforces a strict
    causal boundary: density → wait → trade → measure.

    This also keeps the return window cleanly separated from the
    estimation window in event studies (BMP test), eliminating the
    need for ad-hoc shift corrections.

    Dividing by N normalizes returns to a per-period basis, making
    different forward_periods directly comparable on a scale basis
    (see Notes for the scope boundary).

    Args:
        data: Must contain ``date``, ``asset_id``, ``price``. Must already
            be sorted with **regular spacing per asset** on the time axis;
            this function shifts by row count and does not inspect ``date``.
        forward_periods: Holding horizon in **rows** of the time axis,
            not calendar time (default 5). On a daily panel this is 5
            trading days; on a weekly panel, 5 weeks; on 1-min bars,
            5 minutes. Frequency is the caller's responsibility.
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
            ``data`` already has a ``forward_return`` column and
            ``overwrite`` is ``False``; or the row horizon / price data
            leaves no finite forward returns after filtering.

    Returns:
        Input DataFrame with ``forward_return`` column appended and the
        overlap horizon ``forward_periods`` stamped on as a reserved column —
        the single source of truth ``factrix.evaluate`` reads (it strips the
        column before dispatch, so it never reaches a metric or ``to_frame``).
        Rows where forward return is not finite (tail nulls, NaN, +inf, -inf)
        are dropped.

    Notes:
        The ``÷N`` per-period normalization is a *scale* choice with
        three caveats the caller should know:

        1. **Arithmetic, not summed-log-return.** This is the
           arithmetic per-period mean of a simple return, not the
           academic-standard direct long-horizon regression of summed
           log returns on the predictor (the latter is
           linear-additive across horizons by construction).
        2. **Compounding bias.** Compounding at the arithmetic mean
           is an upward-biased estimator of cumulative wealth; the
           bias grows with ``N`` and per-bar return variance.
           Negligible for rank-based information coefficient (IC); not negligible for
           signed-return mean and t-tests at large ``N``.
        3. **Scale, not inference.** ``÷N`` aligns the *scale* across
           horizons — it does *not* address the inference problem.
           Overlap is handled by heteroskedasticity-and-autocorrelation-consistent (HAC) (see
           :class:`factrix.inference.NeweyWest`); across-horizon
           selection is handled by the family-wise error rate (FWER) correction in
           :func:`factrix.multi_factor.bhy`. The three concerns
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

    from factrix._data_input import _stamp_forward_periods

    out = (
        data.sort(["asset_id", "date"])
        .with_columns(
            (
                (
                    pl.col("price").shift(-(forward_periods + 1)).over("asset_id")
                    / pl.col("price").shift(-1).over("asset_id")
                    - 1
                )
                / forward_periods
            ).alias("forward_return")
        )
        .filter(pl.col("forward_return").is_finite())
    )
    if out.is_empty():
        raise UserInputError(
            func_name="compute_forward_return",
            field="data",
            value=f"{data.height} rows",
            expected=(
                "at least one finite forward_return after applying the row "
                f"horizon forward_periods={forward_periods}; the panel may be "
                "too short, or price contains only non-finite returns"
            ),
            docs_path=_DOCS_FORWARD_RETURN,
        )
    # Stamp the overlap horizon as the single source of truth for the data;
    # evaluate reads it instead of taking forward_periods at the metric / call
    # layer (the three could silently diverge — see compute_forward_return docs).
    return _stamp_forward_periods(out, forward_periods)


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

    lb = pl.col("forward_return").quantile(lower).over("date")
    ub = pl.col("forward_return").quantile(upper).over("date")

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
