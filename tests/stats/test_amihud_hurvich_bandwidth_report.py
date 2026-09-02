"""``predictive_beta`` must report the bandwidth its kernel actually used.

``_resolve_har_lags`` resolves against the full finite-pair count ``n``, but
the augmented Amihud-Hurvich design fits on ``m = n - h`` rows. At a long
horizon ``m`` can fall below the resolved bandwidth, and the Bartlett kernel
inside :func:`factrix._stats.ols._ols_nw_multivariate` clips to ``m - 1``.
The metadata has to follow the kernel, the way ``_ols_scalar_wald_hac`` and
``_driscoll_kraay_cov`` already report the bandwidth they ran at.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix._stats.ols import _amihud_hurvich_beta
from factrix.metrics.predictive_beta import predictive_beta


def _ts_panel(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    n = len(x)
    return pl.DataFrame(
        {
            "date": np.arange(n),
            "asset_id": ["A"] * n,
            "factor": x,
            "forward_return": y,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.mark.parametrize(("n", "overlap"), [(60, 42), (75, 63), (90, 63)])
def test_reported_har_lags_never_exceeds_the_fitted_design(
    n: int, overlap: int
) -> None:
    """A bandwidth wider than the design it ran on was never applied."""
    rng = np.random.default_rng(n * 100 + overlap)
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=overlap)

    assert not np.isnan(result.value), "guard the real-test path, not a short circuit"
    har_lags = result.metadata["har_lags"]
    n_periods = result.metadata["n_periods"]
    assert isinstance(har_lags, int)
    # The Bartlett kernel cannot use a lag it has no observation pair for.
    assert har_lags <= n_periods - 1


def test_fit_reports_the_bandwidth_the_kernel_clipped_to() -> None:
    """``lags_used`` is the applied bandwidth, not the requested one."""
    rng = np.random.default_rng(1041)
    n, h = 90, 63
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    requested = 30
    fit = _amihud_hurvich_beta(y, x, lags=requested, overlap_periods=h)

    assert fit.n_used > 0
    assert fit.lags_used == min(requested, fit.n_used - 1)
    assert fit.lags_used < requested


def test_ill_conditioned_warning_names_the_bandwidth_that_ran() -> None:
    """Warning text and ``metadata["har_lags"]`` must not contradict."""
    rng = np.random.default_rng(9063)
    n, h = 90, 63
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=h)

    har_lags = result.metadata["har_lags"]
    bandwidth_msgs = [
        str(w.message)
        for w in caught
        if "hac_bandwidth_ill_conditioned" in str(w.message)
    ]
    assert bandwidth_msgs, "the ill-conditioned bandwidth screen should fire here"
    message = bandwidth_msgs[0]
    assert f"the kernel ran at L={har_lags}" in message


def test_short_horizon_reports_the_resolved_bandwidth_unchanged() -> None:
    """Where the design is long enough, the requested bandwidth is the used one."""
    rng = np.random.default_rng(7)
    n, h = 240, 5
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    fit = _amihud_hurvich_beta(y, x, lags=12, overlap_periods=h)

    assert fit.lags_used == 12

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=h, newey_west_lags=12)
    assert result.metadata["har_lags"] == 12
