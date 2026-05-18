"""Preprocessing Step 1-3: forward return computation and adjustment.

Step 1 — Forward Return: (price[t+1+N] / price[t+1] - 1) / N
Step 2 — Winsorize Forward Return: per-date percentile clip
Step 3 — Abnormal Return: forward_return - cross-sectional mean

All functions expect canonical column names (date, asset_id, price).
Use ``adapt()`` to rename before calling.
"""

import polars as pl


def compute_forward_return(
    df: pl.DataFrame,
    forward_periods: int = 5,
) -> pl.DataFrame:
    """Step 1: Compute per-period forward return per asset.

    ``forward_return = (price[t+1+N] / price[t+1] - 1) / N``

    Entry at t+1 (next bar after signal), exit at t+1+N.

    WHY t+1 entry: The signal at t is computed using data up to and
    including price[t]. Using price[t] as both signal input and entry
    price assumes you can trade at the same price used to generate the
    signal — unrealistic in practice. Entry at t+1 enforces a strict
    causal boundary: signal → wait → trade → measure.

    This also keeps the return window cleanly separated from the
    estimation window in event studies (BMP test), eliminating the
    need for ad-hoc shift corrections.

    Dividing by N normalizes returns to a per-period basis, making
    different forward_periods directly comparable on a scale basis
    (see Notes for the scope boundary).

    Args:
        df: Must contain ``date``, ``asset_id``, ``price``. Must already
            be sorted with **regular spacing per asset** on the time axis;
            this function shifts by row count and does not inspect ``date``.
        forward_periods: Holding horizon in **rows** of the time axis,
            not calendar time (default 5). On a daily panel this is 5
            trading days; on a weekly panel, 5 weeks; on 1-min bars,
            5 minutes. Frequency is the caller's responsibility.

    Returns:
        Input DataFrame with ``forward_return`` column appended.
        Rows where forward return is null (end of series) are dropped.

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
           :class:`factrix.stats.NeweyWest`); across-horizon
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

        >>> cfg = fx.AnalysisConfig.individual_continuous(forward_periods=5)
        >>> profile = fx.evaluate(panel, cfg)["factor"]
        >>> isinstance(profile, fx.FactorProfile)
        True
    """
    return (
        df.sort(["asset_id", "date"])
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
        .filter(pl.col("forward_return").is_not_null())
    )


def winsorize_forward_return(
    df: pl.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """Step 2: Per-date percentile clip on forward returns.

    Args:
        lower: Lower quantile bound (default 0.01 = 1st percentile).
        upper: Upper quantile bound (default 0.99 = 99th percentile).
            Set to (0.0, 1.0) to disable.

    Returns:
        DataFrame with ``forward_return`` clipped in-place.
    """
    if lower <= 0.0 and upper >= 1.0:
        return df

    lb = pl.col("forward_return").quantile(lower).over("date")
    ub = pl.col("forward_return").quantile(upper).over("date")

    return df.with_columns(
        pl.col("forward_return").clip(lb, ub).alias("forward_return")
    )


def compute_abnormal_return(df: pl.DataFrame) -> pl.DataFrame:
    """Step 3: Cross-sectional abnormal return.

    ``abnormal_return = forward_return - mean(forward_return) per date``

    Returns:
        DataFrame with ``abnormal_return`` column appended.
    """
    return df.with_columns(
        (pl.col("forward_return") - pl.col("forward_return").mean().over("date")).alias(
            "abnormal_return"
        )
    )
