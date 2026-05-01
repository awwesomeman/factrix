"""Statistical significance tools for any numeric series.

Provides t-statistic computation and significance markers
(``***``/``**``/``*``). Operates on raw numeric arrays — agnostic
to what the series represents.

BHY multiple-testing lives in ``factrix.stats.multiple_testing``;
it operates on *p-values* (profile-era) rather than the legacy
``bhy_threshold(t_stats)`` helper that was removed in the profile
migration.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from factrix._types import EPSILON, DDOF


def _calc_t_stat(mean: float, std: float, n: int) -> float:
    """Compute t-statistic with EPSILON guard against near-zero std.

    Args:
        mean: Sample mean.
        std: Sample standard deviation (ddof=1).
        n: Sample size.

    Returns:
        t-statistic, or 0.0 if std is near-zero or n ≤ 0.
    """
    if std > EPSILON and n > 0:
        return float(mean / (std / np.sqrt(n)))
    return 0.0


def _t_stat_from_array(values: np.ndarray) -> float:
    """Convenience: compute t-stat directly from a 1-D array.

    Args:
        values: 1-D numeric array with at least 2 elements.

    Returns:
        t-statistic of the mean, or 0.0 if insufficient data.
    """
    if len(values) < 2:
        return 0.0
    return _calc_t_stat(
        float(np.mean(values)),
        float(np.std(values, ddof=DDOF)),
        len(values),
    )


def _p_value_from_t(
    t_stat: float,
    n: int,
    alternative: str = "two-sided",
) -> float:
    """P-value from t-statistic using t-distribution.

    Args:
        alternative: "two-sided" (default), "less" (left-tail), "greater" (right-tail).
    """
    if n <= 1:
        return 1.0
    dof = n - 1
    if alternative == "less":
        return float(sp_stats.t.cdf(t_stat, dof))
    if alternative == "greater":
        return float(sp_stats.t.sf(t_stat, dof))
    return float(2 * sp_stats.t.sf(abs(t_stat), dof))


def _p_value_from_z(z: float) -> float:
    """Two-sided p-value from z-statistic using normal distribution."""
    return float(2 * sp_stats.norm.sf(abs(z)))


# Below this sample count, the normal approximation to the binomial
# systematically mis-sizes the test (≈5% actual α at nominal 5% only
# kicks in around n≥20; smaller n is liberal). Use exact binomial CDF
# when we fall below.
_BINOMIAL_EXACT_CUTOFF: int = 20


def _binomial_two_sided_p(hits: int, n: int, p0: float = 0.5) -> float:
    """Two-sided binomial test p-value for ``H₀: p = p0``.

    Uses the exact binomial CDF for ``n < _BINOMIAL_EXACT_CUTOFF`` and
    the normal-approximation ``z = (p̂ − p0) / √(p0(1−p0)/n)`` for larger
    samples. For p0 = 0.5 the two tails are symmetric; otherwise scipy's
    ``binomtest`` handles the asymmetric two-sided convention.
    """
    if n <= 0:
        return 1.0
    if n < _BINOMIAL_EXACT_CUTOFF:
        return float(sp_stats.binomtest(hits, n, p0).pvalue)
    rate = hits / n
    denom = float(np.sqrt(p0 * (1.0 - p0) / n))
    if denom < EPSILON:
        return 1.0
    z = (rate - p0) / denom
    return _p_value_from_z(z)


def _binomial_test_method_name(n: int) -> str:
    """Human-readable test name mirroring the branch in ``_binomial_two_sided_p``."""
    return (
        "binomial exact test" if n < _BINOMIAL_EXACT_CUTOFF
        else "binomial score test (normal approximation)"
    )


def _t_test_summary(
    mean: float, std: float, n: int,
) -> tuple[float, float, str]:
    """Compute t-stat, p-value, and significance marker in one call."""
    t = _calc_t_stat(mean, std, n)
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p)


def _significance_marker(p_value: float | None) -> str:
    """Map p-value to academic significance marker.

    | Marker | Condition   | Meaning              |
    |:------:|-------------|----------------------|
    | ``***``| p < 0.01    | Highly significant   |
    | ``**`` | p < 0.05    | Significant          |
    | ``*``  | p < 0.10    | Weakly significant   |
    |        | p >= 0.10   | Not significant      |

    Returns:
        One of ``"***"``, ``"**"``, ``"*"``, ``""``.
    """
    if p_value is None:
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def _resolve_nw_lags(
    n: int,
    lags: int | None,
    forward_periods: int | None,
) -> int:
    """Pick Bartlett-kernel bandwidth, honoring the overlap horizon.

    ``max(floor(T^(1/3)), forward_periods - 1)`` when ``forward_periods``
    is provided; the ``h - 1`` floor is required for consistency when
    input series carries an MA(h-1) structure from overlapping forward
    returns. Clipped to ``n - 1`` so the kernel stays inside the sample.
    """
    base = int(np.floor(n ** (1 / 3))) if lags is None else lags
    if forward_periods is not None:
        base = max(base, max(forward_periods - 1, 0))
    return min(base, n - 1)


def _newey_west_se(
    values: np.ndarray,
    lags: int | None = None,
    forward_periods: int | None = None,
) -> float:
    """Newey-West HAC standard error for the mean of a time series.

    Uses Bartlett kernel weights: w_j = 1 - j/(L+1).

    Args:
        values: 1-D array of time series observations.
        lags: Number of lags. Defaults to ``floor(T^(1/3))``.
        forward_periods: Overlap horizon of the input series. When set,
            enforces ``lags >= forward_periods - 1`` — the minimum
            consistent bandwidth for overlapping h-period returns
            (Hansen-Hodrick 1980 MA(h-1) structure).

    Returns:
        HAC-adjusted standard error of the mean.
    """
    n = len(values)
    if n < 2:
        return 0.0

    lags = _resolve_nw_lags(n, lags, forward_periods)

    mean = float(np.mean(values))
    demeaned = values - mean

    # γ_0 = Var
    gamma_0 = float(np.dot(demeaned, demeaned)) / n

    # Weighted autocovariances: γ_j with Bartlett kernel
    weighted_sum = gamma_0
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        weight = 1.0 - j / (lags + 1)
        weighted_sum += 2.0 * weight * gamma_j

    variance_of_mean = max(weighted_sum / n, 0.0)
    return float(np.sqrt(variance_of_mean))


# MacKinnon (1996) asymptotic critical values, constant-only ADF model.
# Used for a linear-interpolation p-value approximation — precise to
# ~±0.02 across the decision-relevant tail. For production-grade
# p-values, call statsmodels.tsa.stattools.adfuller instead.
_ADF_CRITS_CONSTANT: tuple[tuple[float, float], ...] = (
    (-3.96, 0.001),
    (-3.43, 0.01),
    (-2.86, 0.05),
    (-2.57, 0.10),
    (-1.62, 0.50),
    (-0.44, 0.90),
    (0.23, 0.95),
)


def _adf_pvalue_interp(tau: float) -> float:
    """Linear interpolation of ADF p-value from MacKinnon (1996) crits.

    Behaviour at the tails is driven by the outermost critical points
    in ``_ADF_CRITS_CONSTANT``: τ below the leftmost point clamps to
    0.001 (strongly reject unit root); τ above the rightmost clamps to
    0.95 — this is the rightmost MacKinnon value, **not** a hardcoded
    cap. Extending the right tail would require adding critical points
    beyond τ = 0.23.
    """
    if tau <= _ADF_CRITS_CONSTANT[0][0]:
        return _ADF_CRITS_CONSTANT[0][1]
    if tau >= _ADF_CRITS_CONSTANT[-1][0]:
        return _ADF_CRITS_CONSTANT[-1][1]
    for (t1, p1), (t2, p2) in zip(_ADF_CRITS_CONSTANT[:-1], _ADF_CRITS_CONSTANT[1:]):
        if t1 <= tau <= t2:
            return p1 + (p2 - p1) * (tau - t1) / (t2 - t1)
    return 0.5


def _adf(y: np.ndarray, lags: int = 0) -> tuple[float, float]:
    """Augmented Dickey-Fuller test with drift (constant, no trend).

    Estimates Δy_t = α + β·y_{t-1} + Σ γ_i·Δy_{t-i} + ε and returns
    (τ, p_approx) where τ = β̂ / SE(β̂) and p_approx comes from linear
    interpolation of MacKinnon (1996) asymptotic critical values for
    the constant-only specification. H0: unit root (β = 0); small τ
    rejects in favour of stationarity.

    Lean-dependency implementation: no statsmodels. Sufficient for
    flagging "likely persistent" factors before downstream regressions;
    not a substitute for a full unit-root toolkit.
    """
    y = np.asarray(y, dtype=np.float64)
    # Defence-in-depth for callers that didn't pre-filter: NaN / Inf
    # inputs feed straight into np.linalg.lstsq and trip LAPACK's
    # DLASCL "parameter had an illegal value" emission at process exit.
    # Return the same "can't reject unit root" shape the short sample
    # guard returns; this is the honest answer on a degenerate input.
    if not np.isfinite(y).all():
        return 0.0, 1.0
    n = len(y)
    if n < 10 + lags:
        return 0.0, 1.0

    dy = np.diff(y)
    y_lag1 = y[:-1]
    T = len(dy) - lags
    if T < 5:
        return 0.0, 1.0

    target = dy[lags:]
    X_cols = [np.ones(T), y_lag1[lags:]]
    for i in range(1, lags + 1):
        X_cols.append(dy[lags - i : len(dy) - i])
    X = np.column_stack(X_cols)

    try:
        beta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    resid = target - X @ beta
    dof = T - X.shape[1]
    if dof <= 0:
        return 0.0, 1.0
    sigma2 = float(np.dot(resid, resid)) / dof
    if sigma2 < EPSILON:
        return 0.0, 1.0
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    if se < EPSILON:
        return 0.0, 1.0
    tau = float(beta[1] / se)
    return tau, _adf_pvalue_interp(tau)


def _newey_west_t_test(
    values: np.ndarray,
    lags: int | None = None,
    forward_periods: int | None = None,
) -> tuple[float, float, str]:
    """Newey-West t-test for H₀: mean = 0.

    Args:
        values: 1-D array of time series observations.
        lags: Optional explicit Bartlett-kernel bandwidth. ``None`` uses
            the default ``floor(T^(1/3))`` rule-of-thumb.
        forward_periods: Overlap horizon of the series. When set,
            bandwidth is floored at ``forward_periods - 1`` to stay
            consistent under the MA(h-1) overlap structure.

    Returns:
        (t_stat, p_value, significance_marker)
    """
    from factrix._logging import get_metrics_logger

    n = len(values)
    if n < 3:
        return 0.0, 1.0, ""

    effective_lags = _resolve_nw_lags(n, lags, forward_periods)
    logger = get_metrics_logger()
    logger.debug("newey_west_t_test: n=%d lags=%d", n, effective_lags)
    # WARNING: NW kernel needs enough samples per lag to estimate
    # autocovariances; a crude but standard rule is T >= 5 * lags.
    if effective_lags > 0 and n < 5 * effective_lags:
        logger.warning(
            "newey_west_t_test: n=%d < 5 * lags=%d — HAC estimate may be "
            "poorly conditioned. Consider smaller lags or more data.",
            n, effective_lags,
        )

    mean = float(np.mean(values))
    se = _newey_west_se(values, lags, forward_periods=forward_periods)
    if se < EPSILON:
        return 0.0, 1.0, ""

    t = mean / se
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p)
