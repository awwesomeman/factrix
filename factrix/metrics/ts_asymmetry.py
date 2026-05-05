"""Long-side / short-side asymmetry test (issue #5).

Aggregation: per-date aggregation of factor and forward return to a
common ``(_f, _r)`` series (cross-section step), then NW HAC OLS with
sign-asymmetric slopes on the resulting time series; Wald χ² on the
slope difference.

Diagnostic for `(COMMON, CONTINUOUS, *)` and single-asset TIMESERIES
cells. OLS β reports a single slope and assumes the response is
symmetric around zero — `β > 0` could be "rises more on positive
factor" *or* "falls less on negative factor", and a strategy team
needs to know which.

Two methods, both fit by OLS with Newey-West HAC covariance and
tested by Wald χ² so cross-method p-values stay comparable and the
overlapping-forward-return autocorrelation is handled the same way
as `ts_beta_t_nw`. Welch t is intentionally avoided — its iid
assumption breaks under `forward_periods > 1`.

- Method A (conditional means): `r = β_long·I(f>0) + β_short·I(f<0)
  + β_zero·I(f=0)`. H0: `β_long + β_short = 0` — symmetric magnitude.
- Method B (piecewise slopes): `r = α + β_pos·max(f,0) + β_neg·min(f,0)`.
  H0: `β_pos = β_neg` — slope on the positive side equals slope on
  the negative side.

Gates (issue #5):
- Gate B (mandatory for either method): factor must have both
  positive and negative observations.
- Gate C (method B only): each side needs ≥ 2 distinct factor
  values to identify a slope. Below the gate, method B is skipped
  and `metadata["method_b_skipped"]` records the reason.

Standalone metric — does not enter the registry.

Matrix-row: ts_asymmetry | (COMMON, CONTINUOUS, *, PANEL) | CS-first | NW HAC Wald | _significance_marker, _short_circuit_output, _aggregate_to_per_date, _ols_nw_multivariate, _wald_p_linear
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._stats import (
    _ols_nw_multivariate,
    _resolve_nw_lags,
    _significance_marker,
    _wald_p_linear,
)
from factrix._types import MIN_PORTFOLIO_PERIODS, MetricOutput
from factrix.metrics._helpers import (
    _aggregate_to_per_date,
    _short_circuit_output,
)


def ts_asymmetry(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    forward_periods: int | None = None,
    nw_lags: int | None = None,
) -> MetricOutput:
    """Long/short asymmetry of factor → return relationship.

    Reported headline:

    - ``value`` = method-A magnitude ``β_long + β_short`` (0 under
      perfect symmetry; positive = long side stronger)
    - ``stat``  = ``value`` / NW HAC SE
    - ``metadata["p_value"]`` = method-A Wald p (two-sided)

    Method B (Gate C passing) populates ``beta_pos`` / ``beta_neg`` /
    ``p_wald_slopes``; otherwise ``method_b_skipped`` carries the
    reason.

    Args:
        df: Long panel; aggregated to per-date ``(_f, _r)`` internally.
        factor_col: Column carrying the factor.
        return_col: Column carrying the forward return.
        forward_periods: Overlap horizon of the forward return; used
            to floor the NW bandwidth so the kernel is consistent
            with the autocorrelation it must absorb.
        nw_lags: Override for the NW lag count. ``None`` resolves to
            the standard rule given ``forward_periods`` and ``T``.

    Returns:
        ``MetricOutput`` whose ``value`` is the method-A magnitude;
        diagnostic statistics live in ``metadata``. Short-circuits
        with a reason code when input shape is insufficient (no
        ``date`` column, missing ``factor`` / return column, fewer
        than ``MIN_PORTFOLIO_PERIODS`` per-date rows, or no two-sided
        factor variation).

    Notes:
        Aggregate to per-date ``(_f, _r)`` then fit two NW-HAC OLS
        specifications on the resulting time series::

            Method A: r_t = beta_long*I(f>0) + beta_short*I(f<0)
                          + beta_zero*I(f=0)
                      H0: beta_long + beta_short = 0   (Wald, two-sided)

            Method B: r_t = alpha + beta_pos*max(f, 0) + beta_neg*min(f, 0)
                      H0: beta_pos = beta_neg          (Wald)

        ``value = beta_long + beta_short`` (method A); 0 under perfect
        symmetry, positive when the long side dominates in magnitude.

        factrix runs both methods under NW HAC + Wald (not Welch t)
        because ``forward_periods > 1`` breaks the iid assumption Welch
        relies on, and using one estimator family across A and B keeps
        cross-method p-values comparable.

    References:
        [Newey-West 1987](../../reference/bibliography.md#newey-west-1987): HAC covariance underpinning
        the Wald tests for both methods.
        [Andrews 1991](../../reference/bibliography.md#andrews-1991): Bartlett growth rate ``T^(1/3)``.
        [Hansen-Hodrick 1980](../../reference/bibliography.md#hansen-hodrick-1980): ``forward_periods - 1``
        floor for overlapping returns.
    """
    if "date" not in df.columns:
        return _short_circuit_output(
            "ts_asymmetry", "no_date_column",
        )
    for col in (factor_col, return_col):
        if col not in df.columns:
            return _short_circuit_output(
                "ts_asymmetry", f"no_{col}_column",
            )

    per_date = _aggregate_to_per_date(
        df, factor_col=factor_col, return_col=return_col,
    )
    n_periods = len(per_date)

    if n_periods < MIN_PORTFOLIO_PERIODS:
        return _short_circuit_output(
            "ts_asymmetry", "insufficient_portfolio_periods",
            n_observed=n_periods, min_required=MIN_PORTFOLIO_PERIODS,
        )

    f = per_date["_f"].to_numpy()
    r = per_date["_r"].to_numpy()

    pos_mask = f > 0
    neg_mask = f < 0
    zero_mask = ~(pos_mask | neg_mask)

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    n_zero = int(zero_mask.sum())

    if n_pos == 0 or n_neg == 0:
        return _short_circuit_output(
            "ts_asymmetry", "no_two_sided_factor",
            n_pos=n_pos, n_neg=n_neg, n_zero=n_zero, n_periods=n_periods,
            hint=(
                "factor lacks one of {positive, negative} regions; "
                "long/short asymmetry is undefined. For unsigned event "
                "signals ({0,1}-style) use factrix.metrics.event_quality "
                "(event_hit_rate / event_ic / profit_factor)."
            ),
        )

    lags = _resolve_nw_lags(n_periods, nw_lags, forward_periods)

    # Drop the zero column when n_zero==0 to keep the design matrix full-rank.
    cols = [pos_mask.astype(float), neg_mask.astype(float)]
    if n_zero > 0:
        cols.append(zero_mask.astype(float))
    X_a = np.column_stack(cols)

    beta_a, V_a, _ = _ols_nw_multivariate(r, X_a, lags=lags)

    R_a = np.zeros((1, X_a.shape[1]))
    R_a[0, 0] = 1.0
    R_a[0, 1] = 1.0
    asym_value = float(beta_a[0] + beta_a[1])
    asym_var = float((R_a @ V_a @ R_a.T)[0, 0])
    asym_se = float(np.sqrt(asym_var)) if asym_var > 0 else 0.0
    asym_t = asym_value / asym_se if asym_se > 0 else 0.0
    _, p_a = _wald_p_linear(beta_a, V_a, R_a, q=0.0)

    e_long = float(beta_a[0])
    e_short = float(beta_a[1])
    # >1 → short side larger magnitude than long side.
    abs_ratio = float("nan") if e_long == 0.0 else abs(e_short) / abs(e_long)

    method_b: dict[str, object] = {}
    n_unique_pos = int(np.unique(f[pos_mask]).size)
    n_unique_neg = int(np.unique(f[neg_mask]).size)
    if n_unique_pos < 2 or n_unique_neg < 2:
        method_b["method_b_skipped"] = (
            f"each side needs >=2 distinct factor values to identify a "
            f"slope (got n_unique_pos={n_unique_pos}, n_unique_neg="
            f"{n_unique_neg}). Method A already carries the full information "
            f"for categorical / binary signals."
        )
    else:
        f_pos = np.where(pos_mask, f, 0.0)
        f_neg = np.where(neg_mask, f, 0.0)
        X_b = np.column_stack([np.ones(n_periods), f_pos, f_neg])
        beta_b, V_b, _ = _ols_nw_multivariate(r, X_b, lags=lags)
        R_b = np.array([[0.0, 1.0, -1.0]])
        _, p_b = _wald_p_linear(beta_b, V_b, R_b, q=0.0)
        method_b.update(
            intercept=float(beta_b[0]),
            beta_pos=float(beta_b[1]),
            beta_neg=float(beta_b[2]),
            p_wald_slopes=p_b,
            h0_method_b="beta_pos = beta_neg",
        )

    return MetricOutput(
        name="ts_asymmetry",
        value=asym_value,
        stat=asym_t,
        significance=_significance_marker(p_a),
        metadata={
            "p_value": p_a,
            "stat_type": "wald (NW HAC)",
            "h0": "beta_long + beta_short = 0",
            "method": "method A: dummy regression on sign(factor)",
            "beta_long": e_long,
            "beta_short": e_short,
            "abs_short_over_long": abs_ratio,
            **({"beta_zero": float(beta_a[2])} if n_zero > 0 else {}),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_zero": n_zero,
            "n_periods": n_periods,
            "nw_lags_used": lags,
            **method_b,
        },
    )
