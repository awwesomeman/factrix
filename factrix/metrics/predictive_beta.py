"""Single-asset predictive regression beta.

Notes:
    **Pipeline.** One asset, dense factor, time-series OLS with Newey-West
    heteroskedasticity-and-autocorrelation-consistent (HAC) standard error on
    the slope.

    **Input.** Long panel with ``date, asset_id, factor, forward_return`` where
    ``asset_id`` has one unique value.

    **Output.** ``MetricResult.value`` is the predictive slope ``beta`` and
    ``p_value`` tests ``H0: beta = 0``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from factrix._axis import Aggregation, DataStructure, FactorDensity, InputShape
from factrix._codes import WarningCode, _emit_warning
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import (
    _adf,
    _lag1_autocorr,
    _ols_nw_slope_t,
    _resolve_har_lags,
    _resolve_nw_lags,
)
from factrix._stats.constants import (
    MIN_PERIODS_HARD,
    MIN_PERIODS_WARN,
    PERSISTENT_SERIES_AUTOCORR,
)
from factrix._stats.ols import _amihud_hurvich_beta
from factrix._types import (
    DDOF,
    DEFAULT_FORWARD_PERIODS,
    EPSILON,
)
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _degenerate_test_fields,
    _enforce_min_floor,
    _finite_expr,
    _short_circuit_output,
    _validate_adf_threshold,
    _warn_below_floor,
)

__all__ = ["predictive_beta"]

#: ``|rho_hat * phi_corrected|`` above which the Stambaugh channel is strong
#: enough that even the bias-corrected test is no longer well sized. Re-derived
#: from a size sweep over ``T`` in {60, 120, 240} x ``phi`` in {0.5, 0.9, 0.95,
#: 0.99} x ``rho`` in {-0.5, -0.9} at ``h = 1``, 1500 draws per cell: cells with
#: a measured channel at or below 0.5 reject 4.1-6.6% at a nominal 5%, cells
#: above 0.8 reject 5.7-9.1%. The previous 0.3 fired on the whole
#: ``rho = -0.5`` column, which is calibrated - it read "the corrected test is
#: oversized here" where it is not.
_STAMBAUGH_CHANNEL_WARN: float = 0.7


def _validate_predictive_beta(m: MetricBase) -> None:
    """``adf_threshold`` is a probability, or ``None`` to skip the ADF gate."""
    _validate_adf_threshold(
        m.adf_threshold,  # type: ignore[attr-defined]
        func_name="predictive_beta",
        docs_path="api/metrics/predictive_beta",
    )


@metric(
    cell=cell(None, FactorDensity.DENSE, structure=DataStructure.TIMESERIES),
    aggregation=Aggregation.TS_ONLY,
    slice_boundary_sensitive=True,
    input_shape=InputShape.PANEL,
    sample_threshold=SampleThreshold(
        min_periods=MIN_PERIODS_HARD,
        warn_periods=MIN_PERIODS_WARN,
    ),
    validate=_validate_predictive_beta,
)
def predictive_beta(
    data: pl.DataFrame,
    *,
    newey_west_lags: int | None = None,
    overlap_periods: int = DEFAULT_FORWARD_PERIODS,
    adf_threshold: float | None = 0.10,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Predictive beta for a single-asset dense factor.

    Fits the direct predictive regression
    $R_{t+h} = \alpha + \beta F_t + \varepsilon_t$ on one asset and tests
    ``H0: beta = 0``. The headline bias-corrected slope test uses the
    project's scalar HAR bandwidth; the raw OLS reference retained in
    metadata uses the narrower Newey-West rule.

    ``value`` is the **Stambaugh-bias-corrected** slope, not the raw OLS
    one. [Stambaugh (1999)][stambaugh-1999] showed OLS here is biased by
    $(\sigma_{ev}/\sigma_v^2)(1+3\phi)/T$ whenever the predictor is
    persistent *and* its AR(1) innovation correlates with the return
    innovation — the classic dividend-yield artefact. The bias is the
    **product** of persistence $\phi$ and innovation correlation $\rho$,
    so an ADF screen on the regressor cannot detect it: ADF proxies
    $\phi$ only and has low power at exactly the $\phi \approx 0.95$
    values where the bias bites. Against a true $\beta = 0$ at
    $T=60,\ \phi=0.95,\ \rho=-0.9$, plain OLS averaged $+0.076$ and
    rejected 20.6% at a nominal 5%.

    The correction is the [Amihud-Hurvich (2004)][amihud-hurvich-2004]
    augmented regression — see
    ``factrix._stats.ols._amihud_hurvich_beta`` for the construction, the
    generated-regressor standard error, the two factrix departures from AH
    and the measured bias / size table. The raw OLS slope stays in
    ``metadata["beta_ols_uncorrected"]`` and the correction applied in
    ``metadata["stambaugh_bias_estimate"]``.

    **What the correction does and does not fix.** At
    ``overlap_periods = 1`` the corrected test is calibrated: 4.3–5.5% at a
    nominal 5% when $\rho = 0$ and 6.2–8.3% in the strongest Stambaugh
    cells, against 8.4–18.3% for plain OLS. At ``overlap_periods > 1`` it
    is not — 7.5–14.5% measured, and the excess is there at $\rho = 0$ for
    every $\phi$, so it is neither the Stambaugh channel nor the
    near-unit-root regime. It is the overlapping-regression HAC problem
    (plain OLS-NW carries the same excess), only partly repaired by the HAR
    bandwidth and fixed-$b$ effective df this test now uses. Read an
    ``h > 1`` predictive $p$ against a raised hurdle regardless of the
    correction. ``WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE`` makes that
    known regime explicit on every such result.

    **The correction costs power** where OLS's apparent power was partly
    its own bias: at $T=60,\ \phi=0.95,\ \rho=-0.9$ the corrected test
    rejects 28.8% of a true alternative against OLS's 88.6%. At $\rho = 0$,
    where OLS is unbiased, the gap is small (63.2% against 70.5%). A metric
    that stops being significant after the correction was not necessarily
    significant before it.

    Args:
        data: Single-asset long panel with ``date``, ``asset_id``,
            ``factor_col`` and ``return_col``.
        newey_west_lags: Optional explicit Bartlett lag. ``None`` uses the
            project default bandwidth.
        overlap_periods: Forward-return horizon injected by ``evaluate`` from
            the panel metadata; standalone calls may pass it directly.
        adf_threshold: Augmented Dickey-Fuller p-value above which the
            factor is flagged as a *unit-root suspect*. ``None`` disables
            the check. This is a persistence screen on the regressor, not a
            Stambaugh-bias screen — the bias itself is corrected
            unconditionally, and ``WarningCode.PERSISTENT_REGRESSOR`` also
            fires off the measured bias channel
            ``|rho_hat * phi_corrected|`` above
            ``_STAMBAUGH_CHANNEL_WARN``, which is the trigger that tracks
            the problem.
        factor_col: Predictor column.
        return_col: Forward-return column.

    Returns:
        ``MetricResult`` with ``value`` = the bias-corrected beta, ``stat``
        = its ``t`` statistic, and ``p_value`` for ``H0: beta = 0``. A
        perfect fit or a degenerate design withholds the test
        (``stat`` / ``p_value`` are ``None`` under
        ``WarningCode.DEGENERATE_VARIANCE``) while keeping ``value``.

        Every metadata key names the fit it belongs to.
        ``metadata["alpha"]``, ``metadata["residual_lag1_autocorr"]``,
        ``n_obs``, ``metadata["n_periods"]`` and
        ``metadata["n_periods_effective"]`` describe the **corrected** fit —
        the model ``value`` came from, on the rows it was estimated on.
        ``metadata["beta_ols_uncorrected"]`` and
        ``metadata["r_squared_ols_uncorrected"]`` are the pre-correction OLS
        reference, on the full finite-pair sample
        (``metadata["n_periods_finite"]``). There is deliberately no
        corrected $R^2$: the Amihud-Hurvich slope does not minimise the sum
        of squares, so ``1 - SSR/SST`` off its residual can go negative and
        has no standing in this literature, which reports $R^2$ for the OLS
        regression.

    Notes:
        This is **not** a ``common_beta`` fallback. ``common_beta`` tests the
        cross-asset mean of per-asset betas and therefore remains a PANEL
        metric. ``predictive_beta`` is the explicit TIMESERIES dense metric
        for a single asset.

        Sample definition: the regression runs on the **finite** ``(factor,
        return)`` pairs — non-null and neither ``NaN`` nor ``±inf``. polars'
        ``drop_nulls`` keeps float ``NaN`` (unlike pandas, where ``NaN`` *is*
        the missing marker), and a single ``NaN`` reaching the Newey-West
        slope routine either raises or yields a ``NaN`` beta that
        ``MetricResult`` rejects — one bad cell would fail the whole series.
        ``metadata["n_periods_finite"]`` counts those pairs.

        ``n_obs`` counts the rows the **headline** test ran on, which is
        ``n_periods_finite - overlap_periods`` whenever the correction
        applies: the augmented design spends the first observation on the
        AR(1) lag and, at ``overlap_periods > 1``, the last
        ``overlap_periods - 1`` windows on the horizon-summed innovation
        proxy (a zero-padded truncated sum is a different regressor from the
        one every other row carries). When the sample is too short for the
        augmented design and ``value`` falls back to the plain OLS slope,
        ``n_obs`` is the finite-pair count again — the reported model is then
        the OLS one.

        **What the persistence flags do and do not say.**
        ``PERSISTENT_REGRESSOR`` is an ADF verdict on the *regressor*: the
        unit-root null was not rejected at ``adf_threshold``. It is not a
        statement about the size of this test. Under the classic Stambaugh
        setup (AR(1) $\phi = 0.99$, $\mathrm{corr}(u_x, \varepsilon_r) = -0.9$,
        true $\beta = 0$, 300 seeds) the flag and the distortion move in
        opposite directions with the sample: at $T = 500$ it fires on 74-75%
        of draws while the test rejects 13% (h=1) / 35% (h=21) at a nominal
        5%; at $T = 2500$ the ADF test has the power to reject the unit root,
        the flag falls silent (0.0-0.3% of draws) and the test still rejects
        9% / 14%. Read it as "this slope carries persistent-regressor risk",
        never as "no bias here" when it is absent. The Stambaugh bias itself
        is corrected unconditionally — this flag is about the *regressor*,
        not about whether ``value`` still carries the bias.

        ``SERIAL_CORRELATION_DETECTED`` is the complementary screen, on the
        residuals of the **reported** model — ``y - alpha - value * factor``
        over the ``n_obs`` rows, so the corrected fit's residuals whenever
        the correction applies and the plain OLS ones when it falls back —
        read at stride ``overlap_periods``: above
        ``PERSISTENT_SERIES_AUTOCORR`` no HAC or
        bootstrap path in the library is calibrated, which is the same rule
        ``fm_beta`` and the series-mean inference members apply to their own
        tested series. The stride matters — overlapping forward returns give
        the raw residuals an MA($h-1$) structure by construction, which the
        $h - 1$ Bartlett floor already absorbs, so an unstrided screen would
        fire on every $h > 1$ run and carry no information;
        :class:`~factrix.inference.NonOverlapping` strides its own screen for
        the same reason.

        ``UNRELIABLE_SE_SHORT_PERIODS`` reads the **effective** sample
        ``n_obs // overlap_periods``, not the raw pair count. Overlapping
        forward returns mean ``n`` rows carry about ``n / h`` independent
        observations, and the HAC lag floor rises with ``h`` at the same time,
        so a raw count well clear of the floor can still be a handful of
        independent draws: at $T = 120$, $h = 21$ the regression runs on 98
        rows with a Bartlett lag of 20 and rejects 17.5% of null draws at a
        nominal 5%, while the raw-``n`` gate stayed silent. At ``h = 1`` the
        effective and raw counts coincide, so nothing changes there.
    """
    if factor_col not in data.columns:
        return _short_circuit_output(
            "predictive_beta",
            "no_factor_column",
            missing_column=factor_col,
        )
    if return_col not in data.columns:
        return _short_circuit_output(
            "predictive_beta",
            "no_return_column",
            missing_column=return_col,
        )

    # Finite (not merely non-null) pairs: polars' ``drop_nulls`` keeps float
    # NaN, which would flow into ``_ols_nw_slope_t`` and either crash the
    # Newey-West path or return a NaN beta that ``MetricResult`` rejects.
    paired = (
        data.select("date", factor_col, return_col)
        .filter(_finite_expr(factor_col) & _finite_expr(return_col))
        .sort("date")
    )
    n = paired.height
    sc = _enforce_min_floor(
        predictive_beta,
        "predictive_beta",
        n,
        "insufficient_predictive_periods",
    )
    if sc is not None:
        return sc

    x = paired[factor_col].to_numpy().astype(np.float64)
    y = paired[return_col].to_numpy().astype(np.float64)
    x_std = float(np.std(x, ddof=DDOF))
    if x_std < EPSILON:
        return _short_circuit_output(
            "predictive_beta",
            "degenerate_factor_variance",
            n_obs=n,
            n_obs_axis="periods",
            factor_std=x_std,
        )

    lags = _resolve_nw_lags(n, newey_west_lags, overlap_periods)
    beta_ols, t_ols, p_ols, resid = _ols_nw_slope_t(y, x, lags=lags)
    # The headline test is a SINGLE-restriction slope t, so it takes the HAR
    # bandwidth and the fixed-b effective df that the scalar series-mean path
    # uses - the K x K Wald degradation that keeps the multivariate paths on
    # the narrow rule does not apply to one restriction. The uncorrected OLS
    # slope above stays on the narrow rule so it remains the pre-correction
    # reference it is reported as.
    har_lags = _resolve_har_lags(n, newey_west_lags, overlap_periods)
    # Stambaugh (1999) bias correction via the Amihud-Hurvich (2004)
    # augmented regression. Reported as the headline beta: the uncorrected
    # OLS slope is biased by (sigma_ev/sigma_v^2)(1+3phi)/T whenever the
    # predictor is persistent AND its innovation correlates with the return
    # innovation - the classic dividend-yield artefact, +0.076 against a true
    # 0 at T=60, phi=0.95, rho=-0.9, with a 20.6% rejection rate. The raw OLS
    # slope stays in metadata.
    fit = _amihud_hurvich_beta(y, x, lags=har_lags, overlap_periods=overlap_periods)
    if math.isnan(fit.beta):
        # Too short / degenerate for the augmented design; fall back to the
        # plain slope so the metric still reports what it can. The reported
        # model IS the OLS one here, so every diagnostic below stays on its
        # residuals and on the full finite-pair count.
        beta, t_stat, p_value = beta_ols, t_ols, p_ols
        stambaugh_applied = False
        alpha = float(np.mean(y) - beta * np.mean(x))
        model_resid = resid
        n_used = n
    else:
        beta, t_stat, p_value = fit.beta, fit.t_stat, fit.p_value
        stambaugh_applied = True
        # Diagnostics describe the model that is REPORTED, on the rows that
        # produced it. The augmented design spends the first observation on
        # the AR(1) lag and, at h > 1, the last h - 1 windows on the
        # horizon-summed innovation proxy, so ``alpha``, the residual screen
        # and every sample count read ``fit``'s own rows - not the n finite
        # pairs the uncorrected OLS reference was fitted on. Reporting a
        # corrected slope next to an intercept and a residual autocorrelation
        # taken off the uncorrected fit described a model no caller was shown.
        alpha = fit.alpha
        model_resid = fit.resid
        n_used = fit.n_used
    # R-squared stays an OLS quantity and is named for it. A "corrected R²"
    # is not a least-squares object: the AH slope does not minimise the sum
    # of squares on these rows, so 1 - SSR/SST off the corrected residual can
    # go negative and has no standing in the Stambaugh / AH literature, which
    # reports R² for the OLS regression. Labelling beats inventing.
    ss_res = float(np.dot(resid, resid))
    y_c = y - float(np.mean(y))
    ss_tot = float(np.dot(y_c, y_c))
    r_squared_ols = 0.0 if ss_tot < EPSILON else max(0.0, 1.0 - ss_res / ss_tot)

    adf_metadata: dict[str, float | bool] = {}
    unit_root_suspected = False
    if adf_threshold is not None:
        adf_stat, adf_p = _adf(x)
        unit_root_suspected = adf_p > adf_threshold
        adf_metadata = {
            "adf_stat": adf_stat,
            "adf_p": adf_p,
            "adf_threshold": adf_threshold,
            "unit_root_suspected": unit_root_suspected,
        }

    warning_codes: list[str] = []
    if overlap_periods > 1:
        _emit_warning(
            WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE,
            f"overlap_periods={overlap_periods} puts the predictive slope "
            "test in a known-oversized HAC regime (7.5-14.5% measured null "
            "rejection at a nominal 5%). The estimate and p-value are "
            "returned; use a raised hurdle and compare h=1 or a genuinely "
            "non-overlapping design.",
            label="predictive_beta",
            expected_warnings=expected_warnings,
            warning_codes=warning_codes,
            stacklevel=2,
        )
    # The regime flag fires on the ACTUAL bias channel - the product of
    # persistence and innovation correlation - not on the ADF screen alone.
    # ADF proxies phi only, carries no information about rho, and has low
    # power at exactly the phi ~ 0.95 values where the bias bites: it fired
    # on 1% of runs at phi=0.5 where the bias is already present, and stayed
    # silent on 62% of biased runs at T=240. The ADF result is still
    # reported; it just no longer decides the flag on its own.
    bias_channel = (
        abs(fit.innovation_corr * fit.phi_corrected)
        if stambaugh_applied and not math.isnan(fit.innovation_corr)
        else 0.0
    )
    # The bias-corrected AR(1) coefficient is not clamped (clamping it re-opens
    # the bias the correction exists to close - see _amihud_hurvich_beta), so a
    # phi_c at or above one is a real regime the caller has to see: the
    # innovation proxy is then a non-stationary filter of the predictor.
    phi_corrected_explosive = stambaugh_applied and fit.phi_corrected >= 1.0
    if (
        unit_root_suspected
        or phi_corrected_explosive
        or bias_channel > _STAMBAUGH_CHANNEL_WARN
    ):
        warning_codes.append(WarningCode.PERSISTENT_REGRESSOR.value)
    # Persistence screen on this regression's own residuals, taken at stride
    # overlap_periods — exactly what inference.NonOverlapping does to its
    # tested series, and for the same reason. Overlapping forward returns give
    # the raw residuals an MA(h-1) structure by construction, which the HAC lag
    # floor (h - 1) already absorbs; screening the raw series would therefore
    # fire on every h > 1 run and say nothing. Striding at h removes the
    # mechanical overlap (an AR(phi) series strided at h sits at phi^h) and
    # leaves the genuine persistence the code is about: above
    # PERSISTENT_SERIES_AUTOCORR no HAC or bootstrap path here is calibrated,
    # so the response is a raised hurdle or a longer sample, not a different
    # estimator.
    resid_autocorr = _lag1_autocorr(model_resid[:: max(overlap_periods, 1)])
    if resid_autocorr > PERSISTENT_SERIES_AUTOCORR:
        warning_codes.append(WarningCode.SERIAL_CORRELATION_DETECTED.value)
    # Effective sample, not raw rows: overlapping forward returns leave about
    # n / h independent observations while the HAC lag floor grows with h, so
    # the short-sample gate has to read the same axis the standard error does.
    # h = 1 leaves this identical to the raw count.
    n_effective = n_used // max(overlap_periods, 1)
    warn_code = _warn_below_floor(
        predictive_beta,
        n_effective,
        f"n_periods={n_used} at "
        f"overlap_periods={overlap_periods} "
        f"leaves an effective sample of {n_effective} non-overlapping "
        f"observations, below MIN_PERIODS_WARN={MIN_PERIODS_WARN}; Newey-West "
        f"HAC inference is not calibrated there (measured 17.5% rejection at a "
        f"nominal 5% for n=98, h=21). t-stat is returned but read p-values "
        f"cautiously.",
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
        label="predictive_beta",
        expected_warnings=expected_warnings,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    metadata: dict[str, object] = {
        "stat_type": "t",
        "h0": "beta=0",
        "method": "single-asset predictive regression + Newey-West",
        "n_periods": n_used,
        "n_periods_effective": n_effective,
        # The finite pairs available before the augmented design trimmed its
        # rows: the two counts differ by ``overlap_periods`` whenever the
        # correction applies, and both have to stay auditable.
        "n_periods_finite": n,
        "residual_lag1_autocorr": resid_autocorr,
        "newey_west_lags": lags,
        "har_lags": har_lags,
        "overlap_periods": overlap_periods,
        "alpha": alpha,
        "r_squared_ols_uncorrected": r_squared_ols,
        "factor_std": x_std,
        "stambaugh_adjusted": stambaugh_applied,
        "beta_ols_uncorrected": beta_ols,
        "stambaugh_bias_estimate": (beta_ols - beta) if stambaugh_applied else 0.0,
        "ar1_phi": fit.phi,
        "ar1_phi_corrected": fit.phi_corrected,
        "innovation_corr": fit.innovation_corr,
        "stambaugh_bias_channel": bias_channel,
        "ar1_phi_corrected_explosive": phi_corrected_explosive,
        **adf_metadata,
    }
    # A perfect fit (zero residuals) leaves se_beta ~ 0 and _ols_nw_slope_t
    # returns NaN: degeneracy in the MAXIMUM-evidence direction, not the null.
    # The former (t=0, p=1.0) flowed straight into MetricResult with no flag.
    stat, p_out, alternative = _degenerate_test_fields(
        t_stat, p_value, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        value=beta,
        p_value=p_out,
        alternative=alternative,
        n_obs=n_used,
        n_obs_axis="periods",
        stat=stat,
        warning_codes=tuple(warning_codes),
        metadata=metadata,
    )
