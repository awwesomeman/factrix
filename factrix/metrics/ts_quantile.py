"""Time-series quantile bucketing + monotonicity test (issue #5).

Aggregation: per-date aggregation to a common ``(_f, _r)`` series
(cross-section step), then quantile-bucketed NW HAC OLS on that time
series; Wald χ² on the top-bottom bucket spread.

Diagnostic for `(COMMON, CONTINUOUS, *)` and single-asset TIMESERIES
cells: bucket factor history into quantiles and check the conditional
mean forward return per bucket. Catches U-shape / inverted-U /
extreme-only signals that OLS β assumes away (linear) and reports
pass / fail on as a single slope.

Standalone metric — does not enter the registry. See
`ARCHITECTURE.md` §"Registry procedure vs standalone metric" for the
distinction. SPARSE / binary signals are out of scope; the input gate
redirects to `event_quality` helpers.

Matrix-row: ts_quantile_spread | (COMMON, CONTINUOUS, *, PANEL) | CS-first | NW HAC Wald | _significance_marker, _short_circuit_output, _aggregate_to_per_date, _ols_nw_multivariate, _wald_p_linear
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._stats import (
    _ols_nw_multivariate,
    _resolve_nw_lags,
    _significance_marker,
    _wald_p_linear,
)
from factrix._types import EPSILON, MIN_PORTFOLIO_PERIODS, MetricOutput
from factrix.metrics._helpers import (
    _aggregate_to_per_date,
    _short_circuit_output,
)


def ts_quantile_spread(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    n_groups: int = 5,
    forward_periods: int | None = None,
    nw_lags: int | None = None,
) -> MetricOutput:
    """Bucket time-series factor by historical quantiles, test conditional means.

    Reported:

    - ``value`` = top-bottom spread (β_{K-1} - β_0)
    - ``stat``  = Wald on ``H0: β_{K-1} = β_0`` → two-sided p in metadata
    - ``metadata["spearman_rho"]`` / ``spearman_p`` = small-sample
      monotonicity diagnostic across the K bucket means
    - ``metadata["buckets"]`` = per-bucket ``{idx, mean_return, n}``

    Gate (issue #5): ``n_unique(factor) >= n_groups * 2``. Below the
    gate the factor cannot sustain quantile cuts — short-circuits with
    a redirect to ``event_quality.*`` for binary / sparse signals.

    Args:
        df: Long panel; aggregated to per-date ``(_f, _r)`` internally.
        factor_col: Column carrying the factor.
        return_col: Column carrying the forward return.
        n_groups: Number of quantile buckets ``K`` to cut the factor
            history into.
        forward_periods: Overlap horizon of the forward return; floors
            the NW bandwidth.
        nw_lags: Override for the NW lag count. ``None`` resolves to
            the standard rule given ``forward_periods`` and ``T``.

    Returns:
        ``MetricOutput`` whose ``value`` is the top-bottom bucket
        spread; bucket detail and the Spearman monotonicity diagnostic
        live in ``metadata``. Short-circuits with a reason code when
        input shape is insufficient (no ``date`` / factor / return
        column, fewer than ``MIN_PORTFOLIO_PERIODS`` rows, or factor
        variation below ``n_groups * 2`` distinct values).

    Notes:
        Aggregate the panel to per-date ``(_f, _r)``, ordinal-rank into
        ``K = n_groups`` buckets by historical ``_f`` quantile, run
        ``r_t = sum_k beta_k * I(bucket_t = k) + eps`` with NW HAC
        covariance, and form the spread ``value = beta_{K-1} - beta_0``
        with Wald p-value on ``H0: beta_{K-1} = beta_0``. A
        ``Spearman(0..K-1, beta)`` rank-monotonicity diagnostic across
        buckets is reported alongside.

        factrix uses NW HAC + Wald rather than Welch t for cross-method
        comparability with ``ts_asymmetry`` / ``ts_beta_t_nw`` and
        because ``forward_periods > 1`` breaks the iid assumption Welch
        relies on.

    References:
        [Newey-West 1987][newey-west-1987]: HAC covariance under-pinning
        the Wald test.
        [Andrews 1991][andrews-1991]: Bartlett growth rate ``T^(1/3)``
        used for the default lag.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: ``forward_periods - 1``
        floor for overlapping returns.
    """
    if "date" not in df.columns:
        return _short_circuit_output(
            "ts_quantile_spread", "no_date_column",
        )
    for col in (factor_col, return_col):
        if col not in df.columns:
            return _short_circuit_output(
                "ts_quantile_spread", f"no_{col}_column",
            )

    per_date = _aggregate_to_per_date(
        df, factor_col=factor_col, return_col=return_col,
    )
    n_periods = len(per_date)

    if n_periods < MIN_PORTFOLIO_PERIODS:
        return _short_circuit_output(
            "ts_quantile_spread", "insufficient_portfolio_periods",
            n_observed=n_periods, min_required=MIN_PORTFOLIO_PERIODS,
            n_groups=n_groups,
        )

    n_distinct = int(per_date["_f"].n_unique())
    if n_distinct < n_groups * 2:
        return _short_circuit_output(
            "ts_quantile_spread", "insufficient_factor_variation",
            n_distinct=n_distinct, n_groups=n_groups, n_periods=n_periods,
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
            f"ts_quantile_spread: median {per_bucket_periods} periods per "
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
    _, p_spread = _wald_p_linear(beta, V_hac, R, q=0.0)

    spread_var = float((R @ V_hac @ R.T)[0, 0])
    spread_t = (
        spread_value / float(np.sqrt(spread_var))
        if spread_var >= EPSILON else 0.0
    )

    counts = np.bincount(bucket_idx, minlength=n_groups).astype(int)

    # Spearman across K bucket means: non-parametric shape check, K small.
    if n_groups >= 3:
        rho_res = sp_stats.spearmanr(np.arange(n_groups), beta)
        rho = float(rho_res.statistic)
        rho_p = float(rho_res.pvalue) if not np.isnan(rho_res.pvalue) else 1.0
    else:
        rho, rho_p = float("nan"), 1.0

    return MetricOutput(
        name="ts_quantile_spread",
        value=spread_value,
        stat=spread_t,
        significance=_significance_marker(p_spread),
        metadata={
            "p_value": p_spread,
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
