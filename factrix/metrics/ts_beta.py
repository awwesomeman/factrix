"""Time-series beta metrics for macro common factors.

macro_common factors (VIX, gold, USD index) are a single time series
shared across all assets. Per-asset time-series regression measures
each asset's sensitivity (β) to the common factor.

``compute_ts_betas``: per-asset full-sample TS regression → ``{factor: per-asset DataFrame}``.
``ts_beta``: cross-sectional test on the β distribution.
``mean_r_squared``: average explanatory power across assets.
``compute_rolling_mean_beta``: rolling window mean β for stability analysis.

Notes:
    **Pipeline.** Per-asset full-sample ordinary least squares (OLS) β (time-series step), then
    cross-asset t on the β distribution; rolling-window variant slices
    the time axis before the per-asset step.
"""

from __future__ import annotations

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
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import (
    _calc_t_stat,
    _p_value_from_t,
)
from factrix._types import DDOF
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _enforce_min_floor, _surface_drop_stats
from factrix.metrics._primitives import compute_ts_betas

__all__ = [  # noqa: RUF022 (teaching order, see SSOT note)
    "ts_beta",
    "mean_r_squared",
    "ts_beta_sign_consistency",
    "compute_rolling_mean_beta",
]

_TSB_CELL = cell(FactorScope.COMMON, FactorDensity.DENSE, structure=DataStructure.PANEL)


@metric(
    cell=_TSB_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    input_shape=InputShape.SERIES,
    requires={"ts_betas_df": compute_ts_betas},
    sample_threshold=SampleThreshold(min_assets=3),
)
def ts_beta(ts_betas_df: pl.DataFrame) -> MetricResult:
    r"""Test $H_0: \mathrm{mean}(\beta) = 0$ across assets.

    Uses the cross-sectional distribution of per-asset betas.

    Notes:
        Stage 2 of the BJS-style aggregation order:
        $\overline{\beta} = \mathrm{mean}_i \beta_i$;
        $t = \overline{\beta} / (\mathrm{std}(\beta) / \sqrt{N})$
        with $H_0: \mathbb{E}[\beta] = 0$ across assets. The std is the
        sample cross-sectional std with ``ddof=1``.

        factrix uses an iid cross-asset t at this stage rather than a
        clustered/heteroskedasticity-and-autocorrelation-consistent (HAC) variant: per-asset betas come from non-overlapping
        time-series fits in ``compute_ts_betas``, so the betas are
        approximately independent across assets unless a strong
        latent common factor links them.

    References:
        [Black-Jensen-Scholes 1972][black-jensen-scholes-1972]:
        beta-sorted-portfolio time-series CAPM tests. factrix's
        cross-asset t on mean β is a simplified analogue of the BJS
        aggregation order, not a replication of the grouped-portfolio
        intercept test BJS run on assets sorted into beta deciles.

    Examples:
        Chain from :func:`compute_ts_betas` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ts_beta import compute_ts_betas, ts_beta
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ts_betas_df = compute_ts_betas(panel)["factor"]
        >>> result = ts_beta(ts_betas_df)
        >>> result.name == ""
        True
    """
    betas = ts_betas_df["beta"].drop_nulls().to_numpy()
    n = len(betas)

    sc = _enforce_min_floor(ts_beta, "ts_beta", n, "insufficient_assets", axis="assets")
    if sc is not None:
        return sc

    mean_b = float(np.mean(betas))
    std_b = float(np.std(betas, ddof=DDOF))
    t = _calc_t_stat(mean_b, std_b, n)
    p = _p_value_from_t(t, n)

    metadata: dict[str, object] = {
        "stat_type": "t",
        "h0": "mean(β)=0",
        "method": "cross-sectional t-test on per-asset TS betas",
        "n_assets": n,
        "beta_std": std_b,
        "median_beta": float(np.median(betas)),
    }
    warning_codes: list[str] = []
    _surface_drop_stats(ts_betas_df, "ts_beta", metadata, warning_codes, axis="assets")
    return MetricResult(
        p_value=p,
        value=mean_b,
        n_obs=n,
        n_obs_axis="assets",
        stat=t,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


# ---------------------------------------------------------------------------
# Mean R²
# ---------------------------------------------------------------------------


@metric(
    cell=_TSB_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    input_shape=InputShape.SERIES,
    requires={"ts_betas_df": compute_ts_betas},
    sample_threshold=SampleThreshold(min_assets=1),
)
def mean_r_squared(ts_betas_df: pl.DataFrame) -> MetricResult:
    r"""Average $R^2$ across per-asset TS regressions — ``value`` $= \mathrm{mean}_i R^2_i$.

    $R^2_i$ comes from asset $i$'s regression
    $R_{i,t} = \alpha_i + \beta_i \cdot F_t + \varepsilon$ (computed
    upstream in ``compute_ts_betas``). Metadata carries
    ``median_r_squared`` as well — useful when a few high-$R^2$ assets
    pull the mean. Low values ($< 0.05$) indicate the factor is too
    weak or noisy to drive individual-asset returns even when its
    cross-asset mean $\beta$ looks nonzero.

    Short-circuits to NaN when no assets have a non-null $R^2$.

    Notes:
        ``value`` $= \mathrm{mean}_i R^2_i$ and ``median_r_squared``
        $= \mathrm{median}_i R^2_i$ on the per-asset ordinary least squares (OLS) fits from
        ``compute_ts_betas``. Pure descriptive statistic — no formal
        $H_0$.

        factrix reports both mean and median because a few high-$R^2$
        assets can dominate the mean; large mean-vs-median gaps density
        the factor explains a small subset of assets rather than the
        cross-section as a whole.

    Examples:
        Chain from :func:`compute_ts_betas` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ts_beta import compute_ts_betas, mean_r_squared
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ts_betas_df = compute_ts_betas(panel)["factor"]
        >>> result = mean_r_squared(ts_betas_df)
        >>> result.name == ""
        True
    """
    r2_vals = ts_betas_df["r_squared"].drop_nulls().to_numpy()
    n = len(r2_vals)

    sc = _enforce_min_floor(
        mean_r_squared,
        "mean_r_squared",
        n,
        "no_asset_r_squared_observations",
        axis="assets",
    )
    if sc is not None:
        return sc

    metadata: dict[str, object] = {
        "n_assets": n,
        "median_r_squared": float(np.median(r2_vals)),
        "min_r_squared": float(np.min(r2_vals)),
        "max_r_squared": float(np.max(r2_vals)),
    }
    warning_codes: list[str] = []
    _surface_drop_stats(
        ts_betas_df, "mean_r_squared", metadata, warning_codes, axis="assets"
    )
    return MetricResult(
        value=float(np.mean(r2_vals)),
        n_obs=n,
        n_obs_axis="assets",
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


# ---------------------------------------------------------------------------
# Rolling mean beta for stability / OOS analysis
# ---------------------------------------------------------------------------


@metric(
    cell=_TSB_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
    # Pipeline producer: window-specific eligibility is enforced in-body; no
    # static panel-shape floor can pre-flight how many rolling dates survive.
    sample_threshold=SampleThreshold(),
)
def compute_rolling_mean_beta(
    data: pl.DataFrame,
    *,
    window: int = 60,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Rolling-window mean β across assets — time-series input for out-of-sample (OOS) / trend.

    Formula (per date t ≥ ``window``):
        For each asset i, take the trailing ``window`` rows ending at t.
        If ≥ 10 valid (factor, return) pairs, run ordinary least squares (OLS):
            R_{i,s} = α_i + β_i·F_s + ε   (s in window)
        β_t = mean_i β_i   (cross-asset mean of this window's βs)

    Dates with fewer than ``window`` trailing rows are skipped. Assets
    with < 10 valid obs in the window are dropped from that date's β
    calculation. If no asset qualifies at a given date, that date is
    absent from the output entirely.

    Returns:
        DataFrame with ``date, value`` where ``value`` is the rolling
        cross-asset mean β. Shape compatible with ``oos`` / ``ic_trend``.

    Notes:
        Per date ``t >= window``, run the per-asset TS OLS over the
        trailing ``window`` rows and compute ``value_t = mean_i beta_i``.
        Output schema matches the time-series tools (``oos`` /
        ``ic_trend``), so callers can pipe rolling betas into stability
        and trend diagnostics.

        factrix requires at least 10 valid rows per asset within each
        rolling window; below that, the asset is dropped from that
        date's mean rather than imputed — keeps each ``value_t`` an
        average over identifiable per-asset slopes.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ts_beta import compute_rolling_mean_beta
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> rolling = compute_rolling_mean_beta(panel, window=60)
        >>> set(rolling.columns) >= {"date", "value"}
        True
    """
    dates = data["date"].unique().sort()
    if len(dates) < window:
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Datetime("ms")),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )

    # Partition by asset once into date-sorted numpy arrays, dropping rows with a
    # null factor or return up front: an incomplete pair is unobserved, and
    # leaving it in would feed a NaN into the per-asset OLS and poison that
    # asset's slope (and the cross-asset mean). The trailing date window for each
    # ``t`` is the closed interval ``[dates[i-window], dates[i-1]]`` — every
    # asset row whose date lands in it, located by ``searchsorted`` on the
    # asset's sorted dates — which replaces the per-date ``is_in`` filter and the
    # per-asset ``asset_id ==`` filter the loop used to run.
    valid = data.filter(
        pl.col(factor_col).is_not_null() & pl.col(return_col).is_not_null()
    )
    asset_arrays: dict[object, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, a_data in (
        valid.sort("date")
        .partition_by("asset_id", as_dict=True, maintain_order=True)
        .items()
    ):
        asset_arrays[key[0]] = (
            a_data["date"].to_numpy(),
            a_data[factor_col].to_numpy().astype(np.float64),
            a_data[return_col].to_numpy().astype(np.float64),
        )

    date_vals = dates.to_numpy()
    rows: list[dict] = []
    for i in range(window, len(dates)):
        lo = date_vals[i - window]  # first date in the trailing window (inclusive)
        hi = date_vals[i - 1]  # last date in the trailing window (inclusive)

        betas_per_asset: list[float] = []
        for a_dates, x_all, y_all in asset_arrays.values():
            left = int(np.searchsorted(a_dates, lo, side="left"))
            right = int(np.searchsorted(a_dates, hi, side="right"))
            n = right - left
            if n < 10:
                continue
            x = x_all[left:right]
            y = y_all[left:right]
            X = np.column_stack([np.ones(n), x])
            try:
                b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                betas_per_asset.append(float(b[1]))
            except np.linalg.LinAlgError:
                continue

        if betas_per_asset:
            rows.append(
                {
                    "date": dates[i],
                    "value": float(np.mean(betas_per_asset)),
                }
            )

    if not rows:
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Datetime("ms")),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# β sign consistency (per-asset version)
# ---------------------------------------------------------------------------


@metric(
    cell=_TSB_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    input_shape=InputShape.SERIES,
    requires={"ts_betas_df": compute_ts_betas},
    sample_threshold=SampleThreshold(min_assets=2),
)
def ts_beta_sign_consistency(ts_betas_df: pl.DataFrame) -> MetricResult:
    """Symmetric sign-agreement across per-asset βs — `value = max(pos, 1−pos)` where `pos = mean_i 1{β_i > 0}`.

    Range [0.5, 1.0]: 0.5 = βs evenly split (no directional consensus);
    1.0 = all βs share one sign. Unlike
    ``fm_beta.fm_beta_sign_consistency`` this is **direction-agnostic**
    — it does not require a prior on the factor's expected sign.

    Requires ``n_assets >= 2``: a single β is trivially "100% consistent with
    itself" (the max collapses to 1.0 for any nonzero β), which would
    read as strong evidence on a dashboard but carries zero information.
    Short-circuits to NaN in that case so the degenerate value never
    leaks into downstream inference.

    Notes:
        ``pos = mean_i 1{beta_i > 0}``; ``value = max(pos, 1 - pos)``.
        Direction-agnostic: returns 1 when all assets have positive
        beta or all negative.

        factrix gates this metric at ``n_assets >= 2`` so a single-asset
        ``max(pos, 1-pos) = 1.0`` cannot leak into downstream
        inference as spurious "perfect agreement". Pair with
        ``fm_beta.fm_beta_sign_consistency`` when a directional prior
        is available.

    Examples:
        Chain from :func:`compute_ts_betas` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ts_beta import (
        ...     compute_ts_betas,
        ...     ts_beta_sign_consistency,
        ... )
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ts_betas_df = compute_ts_betas(panel)["factor"]
        >>> result = ts_beta_sign_consistency(ts_betas_df)
        >>> result.name == ""
        True
    """
    betas = ts_betas_df["beta"].drop_nulls().to_numpy()
    n = len(betas)
    sc = _enforce_min_floor(
        ts_beta_sign_consistency,
        "ts_beta_sign_consistency",
        n,
        "insufficient_assets_for_sign_consistency",
        axis="assets",
    )
    if sc is not None:
        return sc

    positive = float(np.mean(betas > 0))
    consistency = max(positive, 1.0 - positive)

    metadata: dict[str, object] = {
        "n_assets": n,
        "fraction_positive": positive,
    }
    warning_codes: list[str] = []
    _surface_drop_stats(
        ts_betas_df, "ts_beta_sign_consistency", metadata, warning_codes, axis="assets"
    )
    return MetricResult(
        value=consistency,
        n_obs=n,
        n_obs_axis="assets",
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
