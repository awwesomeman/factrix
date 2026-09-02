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

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._codes import WarningCode, _emit_warning
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import (
    _ols_scalar_wald_hac,
    _wald_p_linear,
)
from factrix._types import (
    DEFAULT_N_GROUPS,
    MIN_PORTFOLIO_PERIODS_HARD,
)
from factrix.inference.series_mean import _persistent_array_beyond_horizon
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _aggregate_to_per_date,
    _degenerate_test_fields,
    _emit_scalar_har_warnings,
    _enforce_min_floor,
    _short_circuit_output,
    _validate_n_groups,
)

__all__ = [
    "common_quantile_spread",
]

_MIN_BUCKET_PERIODS_WARN = 5


def _validate_common_quantile_spread(m: MetricBase) -> None:
    """Bucketing floor for ``common_quantile_spread``.

    This metric buckets the factor *history* with its own ordinal cut rather
    than the panel kernel, so it states the shared floor here instead of
    inheriting it from the kernel: one bucket has no top-minus-bottom contrast
    to test.
    """
    _validate_n_groups(
        m.n_groups,  # type: ignore[attr-defined]
        func_name="common_quantile_spread",
        docs_path="api/metrics/common_quantile",
    )


@metric(
    cell=cell(FactorScope.COMMON, FactorDensity.DENSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.CS_THEN_TS,
    sample_threshold=SampleThreshold(min_periods=MIN_PORTFOLIO_PERIODS_HARD),
    validate=_validate_common_quantile_spread,
)
def common_quantile_spread(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    n_groups: int = DEFAULT_N_GROUPS,
    overlap_periods: int | None = None,
    newey_west_lags: int | None = None,
    expected_warnings: tuple[str, ...] = (),
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
            history into. At least :data:`~factrix._types.N_GROUPS_FLOOR` = 2
            (the top-minus-bottom contrast needs two buckets); the Spearman
            shape check is reported as NaN below ``K = 3``.
        overlap_periods: Overlap horizon of the forward return; floors the
            Bartlett bandwidth at ``3 * (overlap_periods - 1)`` and caps the
            reference degrees of freedom at ``T / overlap_periods - 1``.
        newey_west_lags: Override for the Bartlett lag count. ``None``
            resolves to the standard rule given ``overlap_periods`` and
            ``T``; the finite-sample scale and reference df follow the
            resolved bandwidth either way.

    Returns:
        ``MetricResult`` whose ``value`` is the top-bottom bucket
        spread; bucket detail and the Spearman monotonicity diagnostic
        live in ``metadata``. Short-circuits with a reason code when
        input shape is insufficient (no factor / return column, fewer
        than ``MIN_PORTFOLIO_PERIODS_HARD`` rows, or factor variation
        below ``n_groups * 2`` distinct values); a missing ``date`` /
        ``asset_id`` is a ``UserInputError`` from the panel key-column gate.

    Notes:
        Aggregate the panel to per-date ``(_f, _r)``, ordinal-rank into
        ``K = n_groups`` buckets by historical ``_f`` quantile, run
        ``r_t = sum_k beta_k * I(bucket_t = k) + eps`` with NW heteroskedasticity-and-autocorrelation-consistent (HAC)
        covariance, and form the spread ``value = beta_{K-1} - beta_0``
        with Wald p-value on ``H0: beta_{K-1} = beta_0``. That is a
        **single** restriction, so the bandwidth, the ``T / (T - L - 1)``
        variance scale and the fixed-``b`` effective degrees of freedom come
        from :func:`~factrix._stats.hac._resolve_scalar_wald_hac` — the same
        recipe the scalar series-mean HAR t-test uses, and the one this
        metric's size table is measured on (``inference-calibration``). A
        ``Spearman(0..K-1, beta)`` rank-monotonicity diagnostic across
        buckets is reported alongside.

        ``HAC_BANDWIDTH_ILL_CONDITIONED`` reports a resolved bandwidth
        estimated from too few lag products on samples of at least 30 periods.
        Shorter samples follow this metric's effective-sample warning contract.

        factrix uses NW HAC + Wald rather than Welch t for cross-method
        comparability with ``common_asymmetry`` / ``common_beta`` and
        because ``overlap_periods > 1`` breaks the iid assumption Welch
        relies on.

    References:
        [Newey-West 1987][newey-west-1987]: HAC covariance under-pinning
        the Wald test.
        [Newey-West 1994][newey-west-1994]: automatic Bartlett bandwidth
        used by the default lag resolver.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: the MA(h-1) structure
        the overlap floor covers.
        [LLSW 2018][llsw-2018]: the HAR bandwidth and effective-df recipe the
        single-restriction contrast reads.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.common_quantile import common_quantile_spread
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> result = common_quantile_spread(panel, n_groups=5)
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
            "common_quantile_spread",
            "no_factor_column",
            missing_column=factor_col,
        )
    if return_col not in data.columns:
        return _short_circuit_output(
            "common_quantile_spread",
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
    thin_bucket_periods = per_bucket_periods < _MIN_BUCKET_PERIODS_WARN
    if thin_bucket_periods:
        _emit_warning(
            WarningCode.THIN_QUANTILE_PERIODS,
            f"avg {per_bucket_periods} periods per "
            f"bucket (T={n_periods}, n_groups={n_groups}). Each bucket mean "
            f"sits on a thin sample; consider reducing n_groups.",
            label="common_quantile_spread",
            expected_warnings=expected_warnings,
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

    # Single restriction (top bucket minus bottom), so this takes the scalar
    # HAR recipe -- bandwidth, T/(T-L-1) variance scale and fixed-b effective
    # df together. The narrow K-restriction rule left this metric 9.7-21%
    # oversized at h=5; see ``_resolve_scalar_wald_hac`` for the table.
    hac = _ols_scalar_wald_hac(
        r,
        X,
        lags=newey_west_lags,
        overlap_periods=overlap_periods,
    )
    beta, V_hac = hac.beta, hac.covariance

    R = np.zeros((1, n_groups))
    R[0, n_groups - 1] = 1.0
    R[0, 0] = -1.0
    spread_value = float(beta[n_groups - 1] - beta[0])
    _, p_spread = _wald_p_linear(beta, V_hac, R, q=0.0, df_denom=hac.dof)

    # A collapsed contrast variance (identical bucket means every period, or
    # a singular HAC covariance) admits no t and no Wald p: NaN here, and the
    # test fields are withheld below rather than reported as t=0 / p=1.
    spread_var = float((R @ V_hac @ R.T)[0, 0])
    spread_t = (
        spread_value / float(np.sqrt(spread_var))
        if np.isfinite(p_spread)
        else float("nan")
    )

    counts = np.bincount(bucket_idx, minlength=n_groups).astype(int)

    # Spearman across K bucket means: non-parametric shape check, K small.
    # Not computable (K < 3, or identical bucket means) stays NaN in both
    # fields rather than a p = 1 that reads as "no monotonic shape".
    if n_groups >= 3:
        rho_res = sp_stats.spearmanr(np.arange(n_groups), beta)
        rho = float(rho_res.statistic)
        rho_p = float(rho_res.pvalue)
    else:
        rho, rho_p = float("nan"), float("nan")

    warning_codes = _emit_scalar_har_warnings(
        metric_name="common_quantile_spread",
        subject="the common-factor series",
        n_periods=n_periods,
        overlap_periods=overlap_periods,
        persistent=_persistent_array_beyond_horizon(
            per_date["_f"].to_numpy(), overlap_periods
        ),
        bandwidth_ill_conditioned=hac.bandwidth_ill_conditioned,
        expected_warnings=expected_warnings,
        stacklevel=2,
    )
    if thin_bucket_periods:
        _emit_warning(
            WarningCode.THIN_QUANTILE_PERIODS,
            "one or more time-series quantile buckets has too few periods "
            "for a stable conditional-mean estimate.",
            label="common_quantile_spread",
            expected_warnings=expected_warnings,
            warning_codes=warning_codes,
            stacklevel=2,
        )
    metadata: dict[str, object] = {
        "stat_type": "wald (HAR HAC)",
        "h0": "beta_top = beta_bottom",
        "method": "OLS on bucket dummies + Bartlett HAC (HAR reference)",
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "n_groups": n_groups,
        "n_periods": n_periods,
        "n_distinct_factor": n_distinct,
        "newey_west_lags_used": hac.lags,
        "buckets": [
            {
                "idx": int(k),
                "mean_return": float(beta[k]),
                "n": int(counts[k]),
            }
            for k in range(n_groups)
        ],
    }
    stat, p_out, alternative = _degenerate_test_fields(
        spread_t, p_spread, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=spread_value,
        n_obs=n_periods,
        n_obs_axis="periods",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
