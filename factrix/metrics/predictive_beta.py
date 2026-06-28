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
from factrix._stats import _ols_nw_slope_t, _resolve_nw_lags
from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_WARN
from factrix._types import DDOF, EPSILON
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _enforce_min_floor,
    _short_circuit_output,
    _warn_below_floor,
)

__all__ = ["predictive_beta"]


@metric(
    cell=cell(None, FactorDensity.DENSE, structure=DataStructure.TIMESERIES),
    aggregation=Aggregation.TS_ONLY,
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
    factor_col: str = "factor",
    return_col: str = "forward_return",
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
        factor_col: Predictor column.
        return_col: Forward-return column.

    Returns:
        ``MetricResult`` with ``value`` = beta, ``stat`` = HAC ``t`` statistic,
        and ``p_value`` for ``H0: beta = 0``.

    Notes:
        This is **not** a ``ts_beta`` fallback. ``ts_beta`` tests the
        cross-asset mean of per-asset betas and therefore remains a PANEL
        metric. ``predictive_beta`` is the explicit TIMESERIES dense metric
        for a single asset.
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

    paired = (
        data.select("date", factor_col, return_col)
        .drop_nulls([factor_col, return_col])
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

    warning_codes: list[str] = []
    warn_code = _warn_below_floor(
        predictive_beta,
        n,
        f"predictive_beta: n_periods={n} below MIN_PERIODS_WARN="
        f"{MIN_PERIODS_WARN}; Newey-West HAC inference on a short "
        f"single-asset series is power-thin. t-stat is returned but read "
        f"p-values cautiously.",
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    return MetricResult(
        value=beta,
        p_value=p_value,
        n_obs=n,
        n_obs_axis="periods",
        stat=t_stat,
        warning_codes=tuple(warning_codes),
        metadata={
            "stat_type": "t",
            "h0": "beta=0",
            "method": "single-asset predictive regression + Newey-West",
            "n_periods": n,
            "newey_west_lags": lags,
            "forward_periods": forward_periods,
            "alpha": alpha,
            "r_squared": r_squared,
            "factor_std": x_std,
        },
    )
