"""Heteroskedasticity- and autocorrelation-consistent (HAC) standard errors.

Newey-West (Bartlett kernel) and Hansen-Hodrick (rectangular kernel)
HAC SE / t-test for the mean of a (possibly overlapping) time series.
``_resolve_nw_lags`` is the shared bandwidth picker honouring the
forward-overlap horizon.
"""

from __future__ import annotations

import numpy as np

from factrix._stats.core import _p_value_from_t, _significance_marker
from factrix._types import EPSILON


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
            n,
            effective_lags,
        )

    mean = float(np.mean(values))
    se = _newey_west_se(values, lags, forward_periods=forward_periods)
    if se < EPSILON:
        return 0.0, 1.0, ""

    t = mean / se
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p)


def _hansen_hodrick_se(
    values: np.ndarray,
    forward_periods: int,
) -> tuple[float, bool]:
    """Hansen-Hodrick (1980) rectangular-kernel HAC SE for a sample mean.

    Closed-form variance under the textbook MA(h-1) overlap structure
    induced by h-period forward returns:

        Var(mean) = (γ₀ + 2 Σ_{j=1..h-1} γⱼ) / n,    h = forward_periods

    Unlike the Bartlett kernel used by ``_newey_west_se``, weights are
    flat (1.0) inside ``j ≤ h-1`` and zero beyond. The estimator carries
    no PSD guarantee (Andrews 1991 §3): on short / mildly anti-correlated
    samples the parenthesised sum can come out negative. Callers may map
    ``clamped=True`` to a degenerate-sample warning.

    Args:
        values: 1-D array of the overlapping series whose mean is tested.
        forward_periods: Overlap horizon ``h``. Must be ≥ 1; ``h = 1``
            collapses to the iid SE (no autocovariance terms).

    Returns:
        ``(se, clamped)`` — clamped variance √max(., 0); ``clamped`` is
        ``True`` iff the raw variance estimate was < 0.
    """
    n = len(values)
    if n < 2 or forward_periods < 1:
        return 0.0, False

    mean = float(np.mean(values))
    demeaned = values - mean

    gamma_0 = float(np.dot(demeaned, demeaned)) / n
    weighted_sum = gamma_0
    lags = min(forward_periods - 1, n - 1)
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        weighted_sum += 2.0 * gamma_j

    variance_of_mean = weighted_sum / n
    clamped = variance_of_mean < 0.0
    return float(np.sqrt(max(variance_of_mean, 0.0))), clamped


def _hansen_hodrick_t_test(
    values: np.ndarray,
    forward_periods: int,
) -> tuple[float, float, str, bool]:
    """Hansen-Hodrick t-test for ``H₀: mean = 0`` on an overlapping series.

    Returns ``(t, p, marker, clamped)``. The 4-tuple deviates from
    ``_newey_west_t_test``'s 3-tuple deliberately: rectangular-kernel
    variance has no PSD guarantee and callers must surface the clamp
    case as a warning rather than silently treat it as a non-rejection.
    SE → 0 (whether by near-zero raw variance or by clamping) returns
    ``(0.0, 1.0, "", clamped)`` — the conservative "cannot reject"
    direction.
    """
    n = len(values)
    if n < 3 or forward_periods < 1:
        return 0.0, 1.0, "", False

    mean = float(np.mean(values))
    se, clamped = _hansen_hodrick_se(values, forward_periods)
    if se < EPSILON:
        return 0.0, 1.0, "", clamped

    t = mean / se
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p), clamped
