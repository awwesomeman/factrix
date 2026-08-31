"""Heteroskedasticity- and autocorrelation-consistent (HAC) standard errors.

Newey-West (Bartlett kernel) and Hansen-Hodrick (rectangular kernel)
HAC SE / t-test for the mean of a (possibly overlapping) time series.
``_resolve_har_lags`` (scalar series-mean HAR t-test) and
``_resolve_nw_lags`` (multi-restriction Wald) are the two bandwidth
pickers, both honouring the forward-overlap horizon;
``_resolve_scalar_wald_hac`` applies the first one's full recipe —
bandwidth, variance scale and effective degrees of freedom — to a
single-restriction regression contrast.
"""

from __future__ import annotations

import numpy as np

from factrix._stats.constants import auto_bartlett, har_bandwidth
from factrix._stats.core import _p_value_from_t, _significance_marker
from factrix._types import EPSILON

# Shared "this sample admits no t-statistic" return for the HAC t-tests:
# a sample too short for the kernel, or one whose HAC SE collapses to zero.
# NaN rather than the former ``(0.0, 1.0)``: a zero-SE sample is degenerate
# in the maximum-evidence direction (or an undefined 0/0), never the null,
# so reporting ``p = 1`` inverted the meaning. Matches
# ``factrix._stats.core._calc_t_stat`` — see its docstring for the
# scipy / R reference behaviour and why ±inf is not used instead.
# Callers hold the metric- or inference-level context needed to label the
# cause and surface it as ``WarningCode.DEGENERATE_VARIANCE``. The two
# too-short-sample branches also return this sentinel, but that is a data
# shortage rather than degeneracy, and callers flag it as such.
_NOT_COMPUTABLE: tuple[float, float, str] = (float("nan"), float("nan"), "")


def _require_finite(values: np.ndarray, func_name: str) -> np.ndarray:
    """Coerce to a float array and reject non-finite entries.

    A NaN propagates through ``np.mean`` / ``np.dot`` and then slips past
    the ``se < EPSILON`` degeneracy guard (``max(nan, 0.0)`` is ``nan`` and
    every comparison with NaN is False), surfacing as a NaN t / p far from
    the cause. Callers must drop or impute upstream; the public
    ``stationary_bootstrap_resamples`` enforces the same contract.
    """
    values = np.asarray(values, dtype=float)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(f"{func_name}: values must be finite (no NaN / inf).")
    return values


def _resolve_nw_lags(
    n: int,
    lags: int | None,
    overlap_periods: int | None,
) -> int:
    """Pick Bartlett-kernel bandwidth for a *multivariate* HAC fit, honoring the overlap horizon.

    ``max(auto_bartlett(T), overlap_periods - 1)`` when ``overlap_periods``
    is provided; the Newey-West (1994) auto rule supplies the default
    Bartlett bandwidth and the ``h - 1`` floor is required for consistency
    when input series carries an MA(h-1) structure from overlapping forward
    returns. Clipped to ``n - 1`` so the kernel stays inside the sample.

    This is the bandwidth for the **multi-restriction** Wald consumers —
    today the slice cluster-mean tests (:func:`factrix.slice_joint_test`,
    :func:`factrix.slice_pairwise_test`). A K-restriction Wald statistic
    inverts a K x K HAC matrix and degrades under a bandwidth that helps a
    scalar statistic: the K=5 slice joint test moves from 8-9% to 21% at 50
    periods per slice under the HAR rule, so it keeps the narrow rule.

    Single-restriction regression consumers (``spanning_alpha``,
    ``common_quantile_spread``, ``common_asymmetry``) do **not** use this
    rule; they take :func:`_resolve_scalar_wald_hac`, which is the scalar
    HAR recipe. See that function for the measured split.
    """
    if n < 2:
        return 0
    base = auto_bartlett(n) if lags is None else lags
    if overlap_periods is not None:
        base = max(base, max(overlap_periods - 1, 0))
    return max(0, min(base, n - 1))


#: Estimability cap on the HAR bandwidth, as a fraction of the sample.
#: Beyond ``T / 3`` the kernel averages over so few effective blocks that the
#: long-run variance is noise; ``arch`` applies the same ``n / 3`` bound to the
#: block-bootstrap block length
#: (:func:`factrix._stats.bootstrap._politis_white_block_length`).
_MAX_BANDWIDTH_FRACTION = 3


def _resolve_har_lags(
    n: int,
    lags: int | None,
    overlap_periods: int | None,
) -> int:
    """Pick the Bartlett-kernel bandwidth for the scalar series-mean HAR t-test.

    ``max(har_bandwidth(T), 3 * (h - 1))``, capped at ``ceil(T / 3)``.

    Two departures from :func:`_resolve_nw_lags`, both driven by measured
    size (see :func:`_newey_west_t_test`'s Notes):

    - The base rule is [Lazarus-Lewis-Stock-Watson (2018)][llsw-2018]'s
      ``1.3*sqrt(T)`` HAR recommendation rather than the Newey-West (1994)
      plug-in. The plug-in's 4-5 lags at research sample sizes leave the
      Bartlett long-run variance badly downward-biased.
    - The overlap floor is ``3 * (h - 1)``, not ``h - 1``. ``h - 1`` is the
      *consistency* floor ([Hansen-Hodrick (1980)][hansen-hodrick-1980]
      MA(h-1) structure) and nothing more: the Bartlett weight
      ``w_j = 1 - j/(L+1)`` sends the lag-``(h-1)`` autocovariance to
      ``1/h`` (near zero) when ``L = h - 1``, so a bandwidth exactly at the
      consistency floor still discards most of the overlap covariance it is
      there to capture. At ``L = 3(h - 1)`` the mean weight across the MA
      band is about 0.83 instead of about 0.5, and the measured size at
      ``T=240, h=21`` falls from 10.1% to 6.0% (40000 replications, seed
      20260828; ``tests/stats/test_overlap_floor_size.py`` re-runs the
      contrast at a reduced replication count). That comparison moves the
      floor alone and holds the other two departures below at what factrix
      does today; the "before" column of the table in
      :func:`_newey_west_t_test`'s Notes is the wider contrast that swaps
      all three at once.

    The ``ceil(T / 3)`` cap keeps the kernel estimable; it binds only in the
    thin regime that :func:`_hac_bandwidth_ill_conditioned` flags. An explicit
    ``lags`` replaces the base rule but is still floored by the overlap
    horizon and subject to the cap.
    """
    if n < 2:
        return 0
    base = har_bandwidth(n) if lags is None else lags
    if overlap_periods is not None:
        base = max(base, 3 * max(overlap_periods - 1, 0))
    cap = min(n - 1, max(1, -(-n // _MAX_BANDWIDTH_FRACTION)))
    return max(0, min(base, cap))


def _har_dof(n: int, lags: int, overlap_periods: int | None) -> float:
    """Effective degrees of freedom for a Bartlett-kernel HAR t-test.

    ``min(1.5 · T / L - 1, T / h - 1)``, floored at 1.

    The first term rests on [Lazarus-Lewis-Stock-Watson (2018)][llsw-2018]:
    the fixed-``b`` limiting distribution of a Bartlett-kernel HAR t-statistic
    ([Kiefer-Vogelsang (2005)][kiefer-vogelsang-2005]) is well approximated by
    a Student ``t`` with about ``1.5·T/L`` degrees of freedom. Using
    ``t_{T-1}`` instead — the previous convention — treats a bandwidth-``L``
    HAC variance as if it were estimated from ``T`` independent observations.

    Two departures from LLSW, both factrix choices rather than published
    forms, stated here and on the ``statistical-methods`` docs page:

    - The ``- 1``. LLSW's own Stata implementation (``harreg.ado``) uses
      ``ceil(1.5 · T / S)`` with no subtraction. Subtracting one is a
      small-sample conservatism worth under one degree of freedom; it is
      kept because it never widens the test and costs no measurable power.
    - The ``T / h - 1`` cap. Not in LLSW or KV: an h-period overlapping
      series carries at most ``T / h`` independent observations no matter
      how the kernel is tuned, and without the cap the ``h = 21`` cells
      stay 2–3× oversized at moderate ``T`` (measured T=60, h=21:
      12.2% -> 4.3%).

    Consequence of the cap, disclosed rather than corrected: when
    ``overlap_periods`` is passed on a series with *less* dependence than
    ``h`` implies — an already non-overlapping or nearly iid series — the
    test is markedly conservative (measured size at h=21: 0.2% at T=60,
    2.1% at T=120, 3.5% at T=240; AR(0.6) h=21 T=60: 0.8%). Power stays
    high in those cells (0.82–1.0), so the cost is a wider interval rather
    than a blind test. Pass ``overlap_periods`` only for the horizon the
    series is actually built from.
    """
    dof = 1.5 * n / max(lags, 1) - 1.0
    if overlap_periods is not None and overlap_periods > 1:
        dof = min(dof, n / overlap_periods - 1.0)
    return max(dof, 1.0)


def _resolve_scalar_wald_hac(
    n: int,
    lags: int | None,
    overlap_periods: int | None,
) -> tuple[int, float, float]:
    """Bandwidth, variance scale and reference df for a **single-restriction** HAC test.

    Returns ``(lags, variance_scale, df_denom)``: the caller runs
    ``_ols_nw_multivariate`` at ``lags``, multiplies the HAC covariance by
    ``variance_scale``, and reads the resulting ``t`` / Wald against
    ``df_denom`` degrees of freedom.

    This is the scalar HAR recipe of :func:`_resolve_har_lags` /
    :func:`_har_dof` — the ``1.3*sqrt(T)`` base, the ``3(h - 1)`` overlap
    floor, the ``ceil(T / 3)`` cap, the ``T / (T - L - 1)`` finite-sample
    variance scale and the fixed-``b`` effective degrees of freedom —
    applied to a regression contrast rather than to a series mean. A
    contrast ``R beta`` with ``R`` of rank one is a scalar statistic, so it
    inherits the calibration that recipe was measured on; only the
    ``K >= 2`` Wald statistics of :func:`_resolve_nw_lags` degrade under it.

    **The split, measured.** Empirical size at a nominal 5% on the
    common-factor null (one AR(phi) factor broadcast to 50 assets,
    independent of the returns; 300 replications per cell, seed
    ``20260830 + rep``, Monte-Carlo standard error about 1.3pp), for the
    two single-restriction consumers of this rule:

    | metric | phi | T, h | narrow rule | this rule |
    |---|---|---|---|---|
    | `common_asymmetry` | 0.0 | 60, 5 | 15.3% | 8.0% |
    | `common_asymmetry` | 0.0 | 60, 21 | 34.0% | 5.7% |
    | `common_asymmetry` | 0.0 | 240, 5 | 10.0% | 5.7% |
    | `common_quantile_spread` | 0.0 | 60, 5 | 9.7% | 5.7% |
    | `common_quantile_spread` | 0.0 | 60, 21 | 16.3% | 0.0% |
    | `common_quantile_spread` | 0.0 | 240, 5 | 6.3% | 4.0% |

    Widening only the overlap floor to ``3(h - 1)`` while keeping the
    Newey-West (1994) base, the ``n - 1`` clip and the ``T - k`` reference
    — the narrow rule's other three pieces — measures *worse*, not better
    (`common_quantile_spread` at ``phi=0, T=60, h=5``: 9.7% -> 16.0%; at
    ``h=21``: 16.3% -> 35.7%). A wide Bartlett kernel read against ``T - k``
    degrees of freedom is the case fixed-``b`` theory says needs the
    Kiefer-Vogelsang / LLSW reference; the bandwidth and the reference have
    to move together, which is why this returns all three pieces rather
    than a lag count.

    What it does not fix: a per-period factor that stays persistent *beyond*
    the overlap horizon. At ``phi = 0.9`` the two metrics still measure
    13.0% / 16.3% at ``T = 60, h = 5``, converging to 6.0% / 7.7% by
    ``T = 240``. That regime is flagged
    (:attr:`~factrix._codes.WarningCode.SERIAL_CORRELATION_DETECTED`)
    rather than corrected — see ``reference/inference-calibration``.
    """
    resolved = _resolve_har_lags(n, lags, overlap_periods)
    remaining = n - resolved - 1
    scale = n / remaining if remaining > 0 else 1.0
    return resolved, scale, _har_dof(n, resolved, overlap_periods)


#: The HAC sum needs enough lag products per autocovariance to be stable; the
#: standard crude rule is ``T >= 5 * L``. Below it the long-run variance is
#: dominated by estimation noise and callers raise
#: ``WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED``.
_MIN_PERIODS_PER_LAG = 5


def _hac_bandwidth_ill_conditioned(n: int, lags: int) -> bool:
    """True when ``n_periods < 5 * lags`` — the HAC estimate is poorly conditioned.

    Callers own the ``MetricResult`` / ``InferenceResult`` and map this to
    :attr:`factrix._codes.WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED`.
    """
    return lags > 0 and n < _MIN_PERIODS_PER_LAG * lags


# Andrews-Monahan clip on the AR(1) prewhitening coefficient: at |phi| -> 1 the
# recolouring factor 1 / (1 - phi)^2 explodes, so the fit is bounded away from
# the unit root exactly as Andrews & Monahan (1992) recommend.
_PREWHITEN_PHI_CLIP = 0.97


def _ar1_phi(demeaned: np.ndarray) -> float:
    """Clipped least-squares AR(1) coefficient of an already-demeaned series."""
    denom = float(np.dot(demeaned[:-1], demeaned[:-1]))
    if denom < EPSILON:
        return 0.0
    phi = float(np.dot(demeaned[1:], demeaned[:-1]) / denom)
    return float(np.clip(phi, -_PREWHITEN_PHI_CLIP, _PREWHITEN_PHI_CLIP))


def _bartlett_long_run_variance(demeaned: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance ``γ_0 + 2 Σ w_j γ_j`` of a demeaned series."""
    n = len(demeaned)
    weighted_sum = float(np.dot(demeaned, demeaned)) / n
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        weighted_sum += 2.0 * (1.0 - j / (lags + 1)) * gamma_j
    return weighted_sum


def _newey_west_se(
    values: np.ndarray,
    lags: int | None = None,
    overlap_periods: int | None = None,
    *,
    prewhiten: bool = False,
) -> float:
    """Newey-West standard error of a series mean.

    What it does for a factor test: a per-period IC (or spread, or beta)
    series that trends or moves in regimes has fewer independent
    observations than its length suggests, so a naive ``mean / (sd / √n)``
    overstates the evidence. This SE widens with the series' serial
    correlation so the resulting t / p reflect the information actually
    in the sample. Bartlett kernel weights ``w_j = 1 - j/(L+1)``, the
    convention factor-research papers report against.

    Known limit — measured, disclosed, not corrected by default: the
    Bartlett estimate still understates the long-run variance of a
    *strongly* persistent series. With the LLSW bandwidth and effective
    df now in force (see :func:`_resolve_har_lags` / :func:`_har_dof`) the
    AR(0.6) mean test sits at 5.4–8.1% against a nominal 5% (8.1% at
    T=60, 5.4% at T=1000; was 9–17%), but
    AR(0.9) is 10–18% and no bandwidth rule fixes that. Above lag-1
    autocorrelation 0.3 the tested series is flagged
    ``SERIAL_CORRELATION_DETECTED`` so the regime is never silent (see
    ``PERSISTENT_SERIES_AUTOCORR``).

    The returned SE carries the ``T / (T - L - 1)`` finite-sample scale
    (see the inline note below), so it is *not* the textbook
    ``√(LRV / T)``. Squaring it and comparing against a hand-computed
    Bartlett sum requires dividing the scale back out.

    ``prewhiten=True`` applies [Andrews-Monahan (1992)][andrews-monahan-1992]
    AR(1) prewhitening — fit ``x_t = φ x_{t-1} + e_t`` on the demeaned
    series, run the Bartlett sum on ``e``, recolour by ``1 / (1 - φ̂)²``
    with ``φ̂`` clipped to ±0.97. It recovers 93–97% of the AR(0.6)
    long-run variance and brings the pure-AR(1) mean test back to its iid
    baseline, at no cost on iid or real overlapping input. It is *not* the
    default: factor-research convention is plain Newey-West, matching
    published numbers is a core use of this library, and R's
    ``sandwich::NeweyWest`` is the only mainstream tool that defaults it
    on (statsmodels and Stata do not). The flag exists so the
    characterisation tests and ``reference/inference-calibration`` can pin
    what prewhitening would and would not buy; every library path uses
    the default.

    Args:
        values: 1-D array of time series observations.
        lags: Number of lags. Defaults to ``har_bandwidth(T)`` via
            :func:`_resolve_har_lags`.
        overlap_periods: Overlap horizon of the input series. When set,
            enforces ``lags >= overlap_periods - 1`` — the minimum
            consistent bandwidth for overlapping h-period returns
            ([Hansen-Hodrick (1980)][hansen-hodrick-1980] MA(h-1) structure).
        prewhiten: Andrews-Monahan AR(1) prewhitening. Off by default;
            see above.

    Returns:
        HAC-adjusted standard error of the mean. ``0.0`` for ``n < 2``; a
        series too short to fit the AR(1) (``n < 4``) uses plain Bartlett
        even when ``prewhiten=True``.
    """
    values = _require_finite(values, "_newey_west_se")
    n = len(values)
    if n < 2:
        return 0.0

    lags = _resolve_har_lags(n, lags, overlap_periods)
    demeaned = values - float(np.mean(values))

    if prewhiten and n >= 4:
        phi = _ar1_phi(demeaned)
        resid = demeaned[1:] - phi * demeaned[:-1]
        lrv = (
            _bartlett_long_run_variance(resid, min(lags, len(resid) - 1))
            / (1.0 - phi) ** 2
        )
    else:
        lrv = _bartlett_long_run_variance(demeaned, lags)

    # Small-sample scale ``T / (T - L - 1)``: a factrix choice, not a textbook
    # or Stata one (Stata's ``newey`` scales by ``T/(T-k)`` with ``k`` the
    # number of regressors — 1 for a mean — and never by the bandwidth). Its
    # derivation: the Bartlett sum is built from autocovariances of a series
    # demeaned *in sample*, so each ``γ̂_j`` carries a bias of about
    # ``-γ_0/T``. For white noise that gives
    # ``E[LRV̂] ≈ γ_0 (1 - (1 + 2 Σ_j w_j) / T) = γ_0 (1 - (L + 1) / T)``,
    # which ``T / (T - L - 1)`` undoes exactly. Without it the SE is 5-15%
    # too small at research sample sizes (h=5 size 8.2-9.6% vs 6.2-6.7%).
    # It partly double-counts with the fixed-b reference distribution of
    # :func:`_har_dof`, whose Kiefer-Vogelsang limit already embeds the
    # demeaning; that is why the h=1 iid cells come out slightly *under*-sized
    # (measured 4.3-4.7% at T <= 120). The trade is deliberate: the h>1
    # overlap cells, which are the ones this library actually serves, need it.
    variance_of_mean = max(lrv / n, 0.0) * (n / max(n - lags - 1, 1))
    return float(np.sqrt(variance_of_mean))


def _newey_west_t_test(
    values: np.ndarray,
    lags: int | None = None,
    overlap_periods: int | None = None,
) -> tuple[float, float, str]:
    """Newey-West HAR t-test for H₀: mean = 0.

    Three pieces, all needed together for the test to be calibrated on
    overlapping data:

    1. Bandwidth ``max(1.3·√T, 3(h - 1))`` capped at ``T/3`` —
       :func:`_resolve_har_lags`.
    2. The ``T / (T - L - 1)`` finite-sample scale on the SE —
       :func:`_newey_west_se`.
    3. Effective degrees of freedom ``min(1.5·T/L - 1, T/h - 1)`` rather
       than ``T - 1`` — :func:`_har_dof`.

    Args:
        values: 1-D array of time series observations.
        lags: Optional explicit Bartlett-kernel bandwidth. ``None`` uses
            the [LLSW (2018)][llsw-2018] ``har_bandwidth(T)`` default.
        overlap_periods: Overlap horizon ``h`` of the series. Floors the
            bandwidth at ``3(h - 1)`` and caps the effective df at
            ``T/h - 1``. Pass it whenever the series is built from
            overlapping h-period forward returns — omitting it leaves the
            test oversized by 2–7×.

    Returns:
        ``(t_stat, p_value, significance_marker)``. A sample too short to
        run the kernel (``n < 3``) or one whose HAC SE collapses to zero
        returns ``(nan, nan, "")`` — see the degeneracy note below.

    Notes:
        Measured rejection frequency at a nominal 5%, 4000 replications
        per cell, on a null MA(h-1) series built from overlapping
        h-period sums of iid normals (``tests/stats/test_hac_overlap_size.py``
        re-runs a small grid of these). "before" is the
        ``max(auto_bartlett(T), h-1)`` bandwidth with ``t_{T-1}``:

        | T   | h  | before | after |
        |-----|----|--------|-------|
        | 60  | 1  | 0.065  | 0.039 |
        | 120 | 1  | 0.064  | 0.048 |
        | 240 | 1  | 0.056  | 0.050 |
        | 500 | 1  | 0.049  | 0.049 |
        | 60  | 5  | 0.148  | 0.068 |
        | 120 | 5  | 0.123  | 0.068 |
        | 240 | 5  | 0.115  | 0.063 |
        | 500 | 5  | 0.102  | 0.056 |
        | 60  | 21 | 0.342  | 0.049 |
        | 120 | 21 | 0.228  | 0.073 |
        | 240 | 21 | 0.180  | 0.060 |
        | 500 | 21 | 0.131  | 0.068 |

        The cost is power, not size: against a mean of
        ``2.5 / √(T/h)`` the same grid rejects 42–70%. ``NonOverlapping``
        remains the more conservative overlap-aware path (it is
        calibrated in every cell but throws away ``h-1`` of every ``h``
        observations); ``NeweyWest`` is now competitive on size rather
        than a strict trade of size for power.

    References:
        - [Lazarus, Lewis, Stock & Watson (2018)][llsw-2018]. "HAR
          Inference: Recommendations for Practice." Journal of Business &
          Economic Statistics, 36(4), 541–559.
        - [Kiefer & Vogelsang (2005)][kiefer-vogelsang-2005]. "A New
          Asymptotic Theory for Heteroskedasticity-Autocorrelation Robust
          Tests." Econometric Theory, 21(6), 1130–1164.
    """
    from factrix._logging import get_metrics_logger

    values = _require_finite(values, "_newey_west_t_test")
    n = len(values)
    if n < 3:
        return _NOT_COMPUTABLE

    effective_lags = _resolve_har_lags(n, lags, overlap_periods)
    logger = get_metrics_logger()
    logger.debug("newey_west_t_test: n=%d lags=%d", n, effective_lags)
    # NW kernel needs enough samples per lag to estimate autocovariances.
    # Callers surface this structurally via ``_hac_bandwidth_ill_conditioned``;
    # the log line stays for paths that hold no result object.
    if _hac_bandwidth_ill_conditioned(n, effective_lags):
        logger.warning(
            "newey_west_t_test: n=%d < 5 * lags=%d — HAC estimate may be "
            "poorly conditioned. Consider smaller lags or more data.",
            n,
            effective_lags,
        )

    mean = float(np.mean(values))
    se = _newey_west_se(values, lags, overlap_periods=overlap_periods)
    if se < EPSILON:
        return _NOT_COMPUTABLE

    t = mean / se
    p = _p_value_from_t(t, n, dof=_har_dof(n, effective_lags, overlap_periods))
    return t, p, _significance_marker(p)


def _hansen_hodrick_se(
    values: np.ndarray,
    overlap_periods: int,
) -> tuple[float, bool]:
    """[Hansen-Hodrick (1980)][hansen-hodrick-1980] rectangular-kernel HAC SE for a sample mean.

    Closed-form variance under the textbook MA(h-1) overlap structure
    induced by h-period forward returns:

        Var(mean) = (γ₀ + 2 Σ_{j=1..h-1} γⱼ) / n,    h = overlap_periods

    Unlike the Bartlett kernel used by ``_newey_west_se``, weights are
    flat (1.0) inside ``j ≤ h-1`` and zero beyond. The estimator carries
    no PSD guarantee ([Andrews (1991)][andrews-1991] §3): on short / mildly anti-correlated
    samples the parenthesised sum can come out negative. Callers may map
    ``clamped=True`` to a degenerate-sample warning.

    Args:
        values: 1-D array of the overlapping series whose mean is tested.
        overlap_periods: Overlap horizon ``h``. Must be ≥ 1; ``h = 1``
            collapses to the iid SE (no autocovariance terms).

    Returns:
        ``(se, clamped)`` — clamped variance √max(., 0); ``clamped`` is
        ``True`` iff the raw variance estimate was < 0.
    """
    values = _require_finite(values, "_hansen_hodrick_se")
    n = len(values)
    if n < 2 or overlap_periods < 1:
        return 0.0, False

    mean = float(np.mean(values))
    demeaned = values - mean

    gamma_0 = float(np.dot(demeaned, demeaned)) / n
    weighted_sum = gamma_0
    lags = min(overlap_periods - 1, n - 1)
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        weighted_sum += 2.0 * gamma_j

    variance_of_mean = weighted_sum / n
    clamped = variance_of_mean < 0.0
    return float(np.sqrt(max(variance_of_mean, 0.0))), clamped


def _bartlett_lrcov(scores_per_period: np.ndarray, lags: int) -> np.ndarray:
    r"""Newey-West (Bartlett-kernel) long-run covariance of a ``(T, K)`` vector sequence.

    For a chronologically ordered sequence $h_1, \dots, h_T$ of
    $K$-vectors:

        $$
        S = \Omega_0 + \sum_{j=1}^{L}\Bigl(1 - \tfrac{j}{L+1}\Bigr)
            (\Omega_j + \Omega_j'),\qquad
        \Omega_j = \sum_{t=j+1}^{T} h_t\,h_{t-j}'.
        $$

    Returns the $K\times K$ matrix $S$ as a *sum* (not time-averaged), so
    a sandwich $(X'X)^{-1} S (X'X)^{-1}$ stays correctly scaled when
    $X'X$ is itself a sum over the same observations — the
    Driscoll-Kraay use below. The Bartlett weight keeps $S$ positive
    semi-definite ([Newey-West (1987)][newey-west-1987]).

    Args:
        scores_per_period: ``(T, K)`` array; row ``t`` is the period-``t``
            vector $h_t$. Rows must already be in time order.
        lags: Bartlett bandwidth $L$. ``0`` collapses to $\Omega_0$
            (the no-autocorrelation White form). Lags beyond ``T - 1``
            contribute nothing and are skipped.
    """
    H = np.atleast_2d(scores_per_period)
    T = H.shape[0]
    cov = H.T @ H  # Ω_0
    max_lag = min(lags, T - 1)
    for j in range(1, max_lag + 1):
        omega_j = H[j:].T @ H[:-j]
        weight = 1.0 - j / (lags + 1)
        cov = cov + weight * (omega_j + omega_j.T)
    return cov


def _driscoll_kraay_cov(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    lags: int | None = None,
) -> tuple[np.ndarray, int, int]:
    r"""[Driscoll & Kraay (1998)][driscoll-kraay-1998] cross-section-robust HAC covariance for a pooled OLS fit.

    Aggregates the per-observation OLS scores
    $u_{it} = x_{it}\,\hat e_{it}$ cross-sectionally within each period to
    $h_t = \sum_{i} u_{it}$, runs a Bartlett-kernel HAC
    (:func:`_bartlett_lrcov`) on the $T$-length sequence of $K$-vectors
    $h_t$, and sandwiches with $(X'X)^{-1}$:

        $$
        V = (X'X)^{-1}\,\hat S_T\,(X'X)^{-1},\qquad
        \hat S_T = \hat\Omega_0 + \sum_{j=1}^{m}\Bigl(1-\tfrac{j}{m+1}\Bigr)
                   (\hat\Omega_j + \hat\Omega_j').
        $$

    Robust to **arbitrary contemporaneous cross-sectional correlation**
    (and serial correlation up to lag $m$): collapsing each period's
    cross-section into the single sum $h_t$ folds the within-period
    dependence into one $K$-vector per period, so the SE only needs the
    time-series HAC of that sequence. This is the gap a one-way
    cluster-on-date SE leaves open — clustering on date treats periods as
    independent and so understates SE when shocks persist across periods,
    while DK is robust to both axes at once.

    Args:
        X: ``(N, K)`` pooled design matrix.
        resid: ``(N,)`` OLS residuals $\hat e_{it}$.
        time_ids: ``(N,)`` period label per row. Cross-sectional sums are
            taken within each distinct label; ``np.unique`` ordering
            (sorted) sets the chronological order of the HAC sequence, so
            sortable date labels keep the lag structure honest.
        lags: Bartlett bandwidth $m$. ``None`` → [Newey-West
            (1994)][newey-west-1994] auto-bandwidth ``auto_bartlett(T)``
            on the *period* count $T$ (not the row count $N$). Clipped to
            ``[0, T - 1]``.

    Returns:
        ``(cov, n_periods, lags_used)`` — ``cov`` is the $K\times K$
        covariance $V$; ``n_periods`` is the number of distinct
        ``time_ids``; ``lags_used`` is the resolved bandwidth $m$.

    Raises:
        numpy.linalg.LinAlgError: ``X'X`` is singular.

    References:
        - [Driscoll & Kraay (1998)][driscoll-kraay-1998]. "Consistent
          Covariance Matrix Estimation with Spatially Dependent Panel
          Data." Review of Economics and Statistics, 80(4), 549–560.
    """
    from factrix._stats.constants import auto_bartlett

    scores = X * resid[:, None]  # (N, K) per-obs score u_it
    # Sum scores within each period → H (T, K). Sorted unique labels give
    # the chronological order the Bartlett lags assume.
    uniq, inverse = np.unique(time_ids, return_inverse=True)
    n_periods = len(uniq)
    cross_section_sums = np.zeros((n_periods, X.shape[1]))
    np.add.at(cross_section_sums, inverse.ravel(), scores)

    lags_used = auto_bartlett(n_periods) if lags is None else lags
    lags_used = max(0, min(lags_used, n_periods - 1))

    # numpy < 2.4 on Apple Accelerate raises spurious FP flags from small
    # dense matmuls on finite input; singular designs are caught via
    # ``LinAlgError`` and the degenerate-SE checks downstream.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        long_run_cov = _bartlett_lrcov(cross_section_sums, lags_used)
        xtx_inv = np.linalg.inv(X.T @ X)
        cov = xtx_inv @ long_run_cov @ xtx_inv
    return cov, n_periods, lags_used


def _hansen_hodrick_t_test(
    values: np.ndarray,
    overlap_periods: int,
) -> tuple[float, float, str, bool]:
    """Hansen-Hodrick t-test for ``H₀: mean = 0`` on an overlapping series.

    Returns ``(t, p, marker, clamped)``. The 4-tuple deviates from
    ``_newey_west_t_test``'s 3-tuple deliberately: rectangular-kernel
    variance has no PSD guarantee and callers must surface the clamp
    case as a warning rather than silently treat it as a non-rejection.
    SE → 0 (whether by near-zero raw variance or by clamping) returns
    ``(nan, nan, "", clamped)`` — see the degeneracy note below.
    """
    values = _require_finite(values, "_hansen_hodrick_t_test")
    n = len(values)
    if n < 3 or overlap_periods < 1:
        return (*_NOT_COMPUTABLE, False)

    mean = float(np.mean(values))
    se, clamped = _hansen_hodrick_se(values, overlap_periods)
    if se < EPSILON:
        return (*_NOT_COMPUTABLE, clamped)

    t = mean / se
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p), clamped
