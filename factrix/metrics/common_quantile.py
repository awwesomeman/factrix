"""Time-series quantile bucketing + monotonicity test.

Diagnostic for the `(COMMON, DENSE, PANEL)` cell (needs `n_assets >= 2`
assets; raises at `n_assets == 1`): bucket factor history into quantiles and check the conditional
mean forward return per bucket. Catches U-shape / inverted-U /
extreme-only signals that ordinary least squares (OLS) β assumes away (linear) and reports
pass / fail on as a single slope.

SPARSE / binary signals are out of scope; the input gate redirects to
`event_quality` helpers.

Notes:
    **Pipeline.** Per-date aggregation to a common ``(_f, _r)`` series
    (cross-section step), then quantile-bucketed Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) OLS on that
    time series; Wald (finite-sample F) on the top-bottom bucket spread.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import (
    _ols_nw_multivariate,
    _resolve_nw_lags,
    _wald_p_linear,
)
from factrix._types import EPSILON, MIN_PORTFOLIO_PERIODS_HARD
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _aggregate_to_per_date,
    _enforce_min_floor,
    _short_circuit_output,
)

__all__ = [
    "common_quantile_spread",
]


@metric(
    cell=cell(FactorScope.COMMON, FactorDensity.DENSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.CS_THEN_TS,
    sample_threshold=SampleThreshold(min_periods=MIN_PORTFOLIO_PERIODS_HARD),
)
def common_quantile_spread(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    n_groups: int = 5,
    forward_periods: int | None = None,
    nw_lags: int | None = None,
) -> MetricResult:
    """Bucket time-series factor by historical quantiles, test conditional means.

    Reported:

    - ``value`` = top-bottom spread (β_{K-1} - β_0)
    - ``stat``  = Wald on ``H0: β_{K-1} = β_0`` → two-sided p in metadata
    - ``metadata["spearman_rho"]`` / ``spearman_p`` = small-sample
      monotonicity diagnostic across the K bucket means
    - ``metadata["buckets"]`` = per-bucket ``{idx, mean_return, n}``

    Gate for conditional means/piecewise slopes: ``n_unique(factor) >= n_groups * 2``. Below the
    gate the factor cannot sustain quantile cuts — short-circuits with
    a redirect to ``event_quality.*`` for binary / sparse signals.

    Args:
        data: Long panel; aggregated to per-date ``(_f, _r)`` internally.
        factor_col: Column carrying the factor.
        return_col: Column carrying the forward return.
        n_groups: Number of quantile buckets ``K`` to cut the factor
            history into.
        forward_periods: Overlap horizon of the forward return; floors
            the Newey-West (NW) bandwidth.
        nw_lags: Override for the NW lag count. ``None`` resolves to
            the standard rule given ``forward_periods`` and ``T``.

    Returns:
        ``MetricResult`` whose ``value`` is the top-bottom bucket
        spread; bucket detail and the Spearman monotonicity diagnostic
        live in ``metadata``. Short-circuits with a reason code when
        input shape is insufficient (no ``date`` / factor / return
        column, fewer than ``MIN_PORTFOLIO_PERIODS_HARD`` rows, or factor
        variation below ``n_groups * 2`` distinct values).

    Notes:
        Aggregate the panel to per-date ``(_f, _r)``, ordinal-rank into
        ``K = n_groups`` buckets by historical ``_f`` quantile, run
        ``r_t = sum_k beta_k * I(bucket_t = k) + eps`` with NW heteroskedasticity-and-autocorrelation-consistent (HAC)
        covariance, and form the spread ``value = beta_{K-1} - beta_0``
        with Wald p-value on ``H0: beta_{K-1} = beta_0``. A
        ``Spearman(0..K-1, beta)`` rank-monotonicity diagnostic across
        buckets is reported alongside.

        factrix uses NW HAC + Wald rather than Welch t for cross-method
        comparability with ``common_asymmetry`` / ``common_beta`` and
        because ``forward_periods > 1`` breaks the iid assumption Welch
        relies on.

    References:
        [Newey-West 1987][newey-west-1987]: HAC covariance under-pinning
        the Wald test.
        [Newey-West 1994][newey-west-1994]: automatic Bartlett bandwidth
        used by the default lag resolver.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: ``forward_periods - 1``
        floor for overlapping returns.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.common_quantile import common_quantile_spread
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = common_quantile_spread(panel, n_groups=5)
        >>> result.name == ""
        True
    """
    if "date" not in data.columns:
        return _short_circuit_output(
            "common_quantile_spread",
            "no_date_column",
        )
    for col in (factor_col, return_col):
        if col not in data.columns:
            return _short_circuit_output(
                "common_quantile_spread",
                f"no_{col}_column",
            )

    per_date = _aggregate_to_per_date(
        data,
        factor_col=factor_col,
        return_col=return_col,
    )
    n_periods = len(per_date)

    sc = _enforce_min_floor(
        common_quantile_spread,
        "common_quantile_spread",
        n_periods,
        "insufficient_portfolio_periods",
        n_groups=n_groups,
    )
    if sc is not None:
        return sc

    n_distinct = int(per_date["_f"].n_unique())
    if n_distinct < n_groups * 2:
        return _short_circuit_output(
            "common_quantile_spread",
            "insufficient_factor_variation",
            n_distinct=n_distinct,
            n_groups=n_groups,
            n_periods=n_periods,
            hint=(
                "factor has too few distinct values for quantile cuts. "
                "Reduce n_groups, or for binary / sparse signals use "
                "factrix.metrics.event_quality.* "
                "(event_hit_rate / event_ic / profit_factor / event_skewness)."
            ),
        )

    per_bucket_periods = n_periods // n_groups
    if per_bucket_periods < 5:
        warnings.warn(
            f"common_quantile_spread: median {per_bucket_periods} periods per "
            f"bucket (T={n_periods}, n_groups={n_groups}). Each bucket mean "
            f"sits on a thin sample; consider reducing n_groups.",
            UserWarning,
            stacklevel=2,
        )

    r = per_date["_r"].to_numpy()

    # Ordinal rank → each row in exactly one bucket; clip handles rank=T edge.
    ranks = per_date["_f"].rank(method="ordinal").to_numpy().astype(np.int64)
    bucket_idx = np.minimum(
        ((ranks - 1) * n_groups) // n_periods,
        n_groups - 1,
    ).astype(np.int64)

    X = np.zeros((n_periods, n_groups))
    X[np.arange(n_periods), bucket_idx] = 1.0

    lags = _resolve_nw_lags(n_periods, nw_lags, forward_periods)
    beta, V_hac, _ = _ols_nw_multivariate(r, X, lags=lags)

    R = np.zeros((1, n_groups))
    R[0, n_groups - 1] = 1.0
    R[0, 0] = -1.0
    spread_value = float(beta[n_groups - 1] - beta[0])
    # Finite-sample F_{r, T-k} reference (k = n_groups regressors), matching the
    # cluster-Wald paths; the asymptotic χ² over-rejects on short T.
    _, p_spread = _wald_p_linear(beta, V_hac, R, q=0.0, df_denom=n_periods - n_groups)

    spread_var = float((R @ V_hac @ R.T)[0, 0])
    spread_t = (
        spread_value / float(np.sqrt(spread_var)) if spread_var >= EPSILON else 0.0
    )

    counts = np.bincount(bucket_idx, minlength=n_groups).astype(int)

    # Spearman across K bucket means: non-parametric shape check, K small.
    if n_groups >= 3:
        rho_res = sp_stats.spearmanr(np.arange(n_groups), beta)
        rho = float(rho_res.statistic)
        rho_p = float(rho_res.pvalue) if not np.isnan(rho_res.pvalue) else 1.0
    else:
        rho, rho_p = float("nan"), 1.0

    return MetricResult(
        p_value=p_spread,
        value=spread_value,
        n_obs=n_periods,
        n_obs_axis="periods",
        stat=spread_t,
        metadata={
            "stat_type": "wald (NW HAC)",
            "h0": "beta_top = beta_bottom",
            "method": "OLS on bucket dummies + Newey-West HAC",
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "n_groups": n_groups,
            "n_periods": n_periods,
            "n_distinct_factor": n_distinct,
            "nw_lags_used": lags,
            "buckets": [
                {
                    "idx": int(k),
                    "mean_return": float(beta[k]),
                    "n": int(counts[k]),
                }
                for k in range(n_groups)
            ],
        },
    )
