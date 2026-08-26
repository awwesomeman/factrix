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

import numpy as np
import polars as pl

from factrix._axis import Aggregation, DataStructure, FactorDensity, InputShape
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import _adf, _lag1_autocorr, _ols_nw_slope_t, _resolve_nw_lags
from factrix._stats.constants import (
    MIN_PERIODS_HARD,
    MIN_PERIODS_WARN,
    PERSISTENT_SERIES_AUTOCORR,
)
from factrix._types import DDOF, EPSILON
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _degenerate_test_fields,
    _enforce_min_floor,
    _finite_expr,
    _short_circuit_output,
    _warn_below_floor,
)

__all__ = ["predictive_beta"]


@metric(
    cell=cell(None, FactorDensity.DENSE, structure=DataStructure.TIMESERIES),
    aggregation=Aggregation.TS_ONLY,
    slice_boundary_sensitive=True,
    input_shape=InputShape.PANEL,
    sample_threshold=SampleThreshold(
        min_periods=MIN_PERIODS_HARD,
        warn_periods=MIN_PERIODS_WARN,
    ),
)
def predictive_beta(
    data: pl.DataFrame,
    *,
    newey_west_lags: int | None = None,
    forward_periods: int = 5,
    adf_threshold: float | None = 0.10,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Predictive beta for a single-asset dense factor.

    Fits the direct predictive regression
    $R_{t+h} = \alpha + \beta F_t + \varepsilon_t$ on one asset and tests
    ``H0: beta = 0`` with a Newey-West HAC standard error. The Bartlett lag
    defaults to the Newey-West (1994) automatic rule, floored at
    ``forward_periods - 1`` so overlapping forward-return windows do not
    understate the standard error.

    Args:
        data: Single-asset long panel with ``date``, ``asset_id``,
            ``factor_col`` and ``return_col``.
        newey_west_lags: Optional explicit Bartlett lag. ``None`` uses the
            project default bandwidth.
        forward_periods: Forward-return horizon injected by ``evaluate`` from
            the panel metadata; standalone calls may pass it directly.
        adf_threshold: Augmented Dickey-Fuller p-value above which the
            factor is flagged as persistent. ``None`` disables the check.
        factor_col: Predictor column.
        return_col: Forward-return column.

    Returns:
        ``MetricResult`` with ``value`` = beta, ``stat`` = HAC ``t`` statistic,
        and ``p_value`` for ``H0: beta = 0``.

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
        ``n_obs`` counts the finite pairs actually regressed.

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
        never as "no bias here" when it is absent, and note that factrix
        applies no Stambaugh correction.

        ``SERIAL_CORRELATION_DETECTED`` is the complementary screen, on the
        residuals this regression actually produced, read at stride
        ``forward_periods``: above ``PERSISTENT_SERIES_AUTOCORR`` no HAC or
        bootstrap path in the library is calibrated, which is the same rule
        ``fm_beta`` and the series-mean inference members apply to their own
        tested series. The stride matters — overlapping forward returns give
        the raw residuals an MA($h-1$) structure by construction, which the
        $h - 1$ Bartlett floor already absorbs, so an unstrided screen would
        fire on every $h > 1$ run and carry no information;
        :class:`~factrix.inference.NonOverlapping` strides its own screen for
        the same reason.

        ``UNRELIABLE_SE_SHORT_PERIODS`` reads the **effective** sample
        ``n_obs // forward_periods``, not the raw pair count. Overlapping
        forward returns mean ``n`` rows carry about ``n / h`` independent
        observations, and the HAC lag floor rises with ``h`` at the same time,
        so a raw count well clear of the floor can still be a handful of
        independent draws: at $T = 120$, $h = 21$ the regression runs on 98
        rows with a Bartlett lag of 20 and rejects 17.5% of null draws at a
        nominal 5%, while the raw-``n`` gate stayed silent. At ``h = 1`` the
        effective and raw counts coincide, so nothing changes there.
    """
    if adf_threshold is not None and not (0.0 < adf_threshold < 1.0):
        raise ValueError(
            f"adf_threshold must be a probability in (0, 1) or None, "
            f"got {adf_threshold!r}"
        )
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

    lags = _resolve_nw_lags(n, newey_west_lags, forward_periods)
    beta, t_stat, p_value, resid = _ols_nw_slope_t(y, x, lags=lags)
    alpha = float(np.mean(y) - beta * np.mean(x))
    ss_res = float(np.dot(resid, resid))
    y_c = y - float(np.mean(y))
    ss_tot = float(np.dot(y_c, y_c))
    r_squared = 0.0 if ss_tot < EPSILON else max(0.0, 1.0 - ss_res / ss_tot)

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
    if unit_root_suspected:
        warning_codes.append(WarningCode.PERSISTENT_REGRESSOR.value)
    # Persistence screen on this regression's own residuals, taken at stride
    # forward_periods — exactly what inference.NonOverlapping does to its
    # tested series, and for the same reason. Overlapping forward returns give
    # the raw residuals an MA(h-1) structure by construction, which the HAC lag
    # floor (h - 1) already absorbs; screening the raw series would therefore
    # fire on every h > 1 run and say nothing. Striding at h removes the
    # mechanical overlap (an AR(phi) series strided at h sits at phi^h) and
    # leaves the genuine persistence the code is about: above
    # PERSISTENT_SERIES_AUTOCORR no HAC or bootstrap path here is calibrated,
    # so the response is a raised hurdle or a longer sample, not a different
    # estimator.
    resid_autocorr = _lag1_autocorr(resid[:: max(forward_periods, 1)])
    if resid_autocorr > PERSISTENT_SERIES_AUTOCORR:
        warning_codes.append(WarningCode.SERIAL_CORRELATION_DETECTED.value)
    # Effective sample, not raw rows: overlapping forward returns leave about
    # n / h independent observations while the HAC lag floor grows with h, so
    # the short-sample gate has to read the same axis the standard error does.
    # h = 1 leaves this identical to the raw count.
    n_effective = n // max(forward_periods, 1)
    warn_code = _warn_below_floor(
        predictive_beta,
        n_effective,
        f"predictive_beta: n_periods={n} at forward_periods={forward_periods} "
        f"leaves an effective sample of {n_effective} non-overlapping "
        f"observations, below MIN_PERIODS_WARN={MIN_PERIODS_WARN}; Newey-West "
        f"HAC inference is not calibrated there (measured 17.5% rejection at a "
        f"nominal 5% for n=98, h=21). t-stat is returned but read p-values "
        f"cautiously.",
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
        expected_warnings=expected_warnings,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    metadata: dict[str, object] = {
        "stat_type": "t",
        "h0": "beta=0",
        "method": "single-asset predictive regression + Newey-West",
        "n_periods": n,
        "n_periods_effective": n_effective,
        "residual_lag1_autocorr": resid_autocorr,
        "newey_west_lags": lags,
        "forward_periods": forward_periods,
        "alpha": alpha,
        "r_squared": r_squared,
        "factor_std": x_std,
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
        n_obs=n,
        n_obs_axis="periods",
        stat=stat,
        warning_codes=tuple(warning_codes),
        metadata=metadata,
    )
