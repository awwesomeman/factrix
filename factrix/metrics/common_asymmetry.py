"""Long-side / short-side asymmetry test.

Diagnostic for the `(COMMON, DENSE, PANEL)` cell (needs `n_assets >= 2`
assets; raises at `n_assets == 1`). Ordinary least squares (OLS) β reports a single slope and assumes the response is
symmetric around zero — `β > 0` could be "rises more on positive
factor" *or* "falls less on negative factor", and a strategy team
needs to know which.

Two methods, both fit by OLS with Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) covariance and
tested by a finite-sample Wald F so cross-method p-values stay comparable and the
overlapping-forward-return autocorrelation is handled the same way
as `common_beta`. Welch t is intentionally avoided — its iid
assumption breaks under `overlap_periods > 1`.

- Method A (conditional means): `r = β_long·I(f>0) + β_short·I(f<0)
  + β_zero·I(f=0)`. H0: `β_long + β_short = 0` — symmetric magnitude.
- Method B (piecewise slopes): `r = α + β_pos·max(f,0) + β_neg·min(f,0)`.
  H0: `β_pos = β_neg` — slope on the positive side equals slope on
  the negative side.

Applicability checks for conditional means/piecewise slopes:
- Mandatory for either method: factor must have both
  positive and negative observations.
- Method B only: each side needs ≥ 2 distinct factor
  values to identify a slope. Below that floor, method B is skipped
  and `metadata["method_b_skipped"]` records the reason.

Notes:
    **Pipeline.** Per-date aggregation of factor and forward return to
    a common ``(_f, _r)`` series (cross-section step), then NW HAC OLS
    with sign-asymmetric slopes on the resulting time series; Wald
    (finite-sample F) on the slope difference.
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
from factrix._stats import (
    _ols_scalar_wald_hac,
    _wald_p_linear,
)
from factrix._types import MIN_PORTFOLIO_PERIODS_HARD
from factrix.inference.series_mean import _persistent_array_beyond_horizon
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _aggregate_to_per_date,
    _degenerate_test_fields,
    _emit_scalar_har_warnings,
    _enforce_min_floor,
    _short_circuit_output,
)

__all__ = [
    "common_asymmetry",
]


@metric(
    cell=cell(FactorScope.COMMON, FactorDensity.DENSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.CS_THEN_TS,
    sample_threshold=SampleThreshold(min_periods=MIN_PORTFOLIO_PERIODS_HARD),
)
def common_asymmetry(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    overlap_periods: int | None = None,
    newey_west_lags: int | None = None,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    """Long/short asymmetry of factor → return relationship.

    Reported headline:

    - ``value`` = method-A magnitude ``β_long + β_short`` (0 under
      perfect symmetry; positive = long side stronger)
    - ``stat``  = ``value`` / Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) SE
    - ``p_value`` = method-A Wald p (two-sided)

    Method B, when each side has enough distinct values, populates
    ``beta_pos`` / ``beta_neg`` / ``p_wald_slopes``; otherwise
    ``method_b_skipped`` carries the reason.

    Args:
        data: Long panel; aggregated to per-date ``(_f, _r)`` internally.
        factor_col: Column carrying the factor.
        return_col: Column carrying the forward return.
        overlap_periods: Overlap horizon of the forward return; floors the
            Bartlett bandwidth at ``3 * (overlap_periods - 1)`` so the kernel
            covers the autocorrelation it must absorb, and caps the reference
            degrees of freedom at ``T / overlap_periods - 1``.
        newey_west_lags: Override for the Bartlett lag count. ``None``
            resolves to the standard rule given ``overlap_periods`` and
            ``T``; the finite-sample scale and reference df follow the
            resolved bandwidth either way.
        expected_warnings: Warning codes the caller declares; a declared code
            is still recorded, the ``UserWarning`` echo is silenced.

    Returns:
        ``MetricResult`` whose ``value`` is the method-A magnitude;
        diagnostic statistics live in ``metadata``. Short-circuits
        with a reason code when input shape is insufficient (no
        ``date`` column, missing ``factor`` / return column, fewer
        than ``MIN_PORTFOLIO_PERIODS_HARD`` per-date rows, or no two-sided
        factor variation).

    Notes:
        Aggregate to per-date ``(_f, _r)`` then fit two NW-HAC ordinary least squares (OLS)
        specifications on the resulting time series::

            Method A: r_t = beta_long*I(f>0) + beta_short*I(f<0)
                          + beta_zero*I(f=0)
                      H0: beta_long + beta_short = 0   (Wald, two-sided)

            Method B: r_t = alpha + beta_pos*max(f, 0) + beta_neg*min(f, 0)
                      H0: beta_pos = beta_neg          (Wald)

        ``value = beta_long + beta_short`` (method A); 0 under perfect
        symmetry, positive when the long side dominates in magnitude.

        Both restrictions are rank one, so both take the scalar HAR recipe
        (:func:`~factrix._stats.hac._resolve_scalar_wald_hac`): bandwidth,
        ``T / (T - L - 1)`` variance scale and fixed-``b`` effective degrees
        of freedom together. See ``inference-calibration`` for the
        measured size. The metric also reports a bandwidth estimated from too
        few lag products as ``hac_bandwidth_ill_conditioned`` from 30 periods;
        shorter samples follow this metric's effective-sample warning contract.
        Persistence beyond the overlap horizon remains
        ``serial_correlation_detected``.

        factrix runs both methods under NW HAC + Wald (not Welch t)
        because ``overlap_periods > 1`` breaks the iid assumption Welch
        relies on, and using one estimator family across A and B keeps
        cross-method p-values comparable.

    References:
        [Newey-West 1987][newey-west-1987]: HAC covariance underpinning
        the Wald tests for both methods.
        [Newey-West 1994][newey-west-1994]: automatic Bartlett bandwidth
        used by the default lag resolver.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: the MA(h-1) structure
        the overlap floor covers.
        [LLSW 2018][llsw-2018]: the HAR bandwidth and effective-df recipe the
        single-restriction contrasts read.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.common_asymmetry import common_asymmetry
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> result = common_asymmetry(panel)
        >>> result.name == ""
        True
    """
    # ``date`` / ``asset_id`` are gated by the direct-call key-column check in
    # ``MetricBase.__call__``, which also rejects a caller-named column the
    # frame does not carry (and ``evaluate`` validates the panel schema), so
    # what reaches here is the *default* column missing from the data — a
    # data-availability verdict, reported on the fixed ``reason`` tokens the
    # sibling metrics use so consumers can branch on them.
    if factor_col not in data.columns:
        return _short_circuit_output(
            "common_asymmetry",
            "no_factor_column",
            missing_column=factor_col,
        )
    if return_col not in data.columns:
        return _short_circuit_output(
            "common_asymmetry",
            "no_return_column",
            missing_column=return_col,
        )

    per_date = _aggregate_to_per_date(
        data,
        factor_col=factor_col,
        return_col=return_col,
    )
    n_periods = len(per_date)

    sc = _enforce_min_floor(
        common_asymmetry,
        "common_asymmetry",
        n_periods,
        "insufficient_portfolio_periods",
    )
    if sc is not None:
        return sc

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
            "common_asymmetry",
            "no_two_sided_factor",
            n_pos=n_pos,
            n_neg=n_neg,
            n_zero=n_zero,
            n_periods=n_periods,
            hint=(
                "factor lacks one of {positive, negative} regions; "
                "long/short asymmetry is undefined. For unsigned event "
                "signals ({0,1}-style) use factrix.metrics.event_quality "
                "(event_hit_rate / event_ic / profit_factor)."
            ),
        )

    # Both contrasts below are single restrictions, so both take the scalar
    # HAR recipe -- bandwidth, T/(T-L-1) variance scale and fixed-b effective
    # df together. The narrow K-restriction rule left this metric 15.3-34%
    # oversized at h>1; see ``_resolve_scalar_wald_hac`` for the table.
    # Drop the zero column when n_zero==0 to keep the design matrix full-rank.
    cols = [pos_mask.astype(float), neg_mask.astype(float)]
    if n_zero > 0:
        cols.append(zero_mask.astype(float))
    X_a = np.column_stack(cols)

    fit_a = _ols_scalar_wald_hac(
        r,
        X_a,
        lags=newey_west_lags,
        overlap_periods=overlap_periods,
    )
    beta_a, V_a = fit_a.beta, fit_a.covariance

    R_a = np.zeros((1, X_a.shape[1]))
    R_a[0, 0] = 1.0
    R_a[0, 1] = 1.0
    asym_value = float(beta_a[0] + beta_a[1])
    asym_var = float((R_a @ V_a @ R_a.T)[0, 0])
    _, p_a = _wald_p_linear(beta_a, V_a, R_a, q=0.0, df_denom=fit_a.dof)
    # A collapsed contrast variance admits no t and no Wald p: NaN here, and
    # the test fields are withheld below rather than reported as t=0 / p=1.
    asym_t = asym_value / float(np.sqrt(asym_var)) if np.isfinite(p_a) else float("nan")

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
        fit_b = _ols_scalar_wald_hac(
            r,
            X_b,
            lags=newey_west_lags,
            overlap_periods=overlap_periods,
        )
        beta_b, V_b = fit_b.beta, fit_b.covariance
        R_b = np.array([[0.0, 1.0, -1.0]])
        _, p_b = _wald_p_linear(beta_b, V_b, R_b, q=0.0, df_denom=fit_b.dof)
        method_b.update(
            method_b="method B: split-slope regression on signed factor",
            stat_type_method_b="wald (HAR HAC)",
            intercept=float(beta_b[0]),
            beta_pos=float(beta_b[1]),
            beta_neg=float(beta_b[2]),
            p_wald_slopes=p_b,
            h0_method_b="beta_pos = beta_neg",
        )

    warning_codes = _emit_scalar_har_warnings(
        metric_name="common_asymmetry",
        subject="the common-factor series",
        n_periods=n_periods,
        overlap_periods=overlap_periods,
        persistent=_persistent_array_beyond_horizon(f, overlap_periods),
        bandwidth_ill_conditioned=fit_a.bandwidth_ill_conditioned,
        expected_warnings=expected_warnings,
        stacklevel=2,
    )
    metadata: dict[str, object] = {
        "stat_type": "wald (HAR HAC)",
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
        "newey_west_lags_used": fit_a.lags,
        **method_b,
    }
    stat, p_out, alternative = _degenerate_test_fields(
        asym_t, p_a, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=asym_value,
        n_obs=n_periods,
        n_obs_axis="periods",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
