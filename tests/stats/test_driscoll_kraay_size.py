"""Reduced-replication size check for pooled Driscoll-Kraay inference.

The panel null combines a persistent regressor, cross-sectional common shocks,
and overlapping forward returns. It pins the regression-specific calibration
behind the public ``pooled_beta`` path without turning the test suite into the
full 1,000-replication research run documented in inference-calibration.md.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from factrix._stats import _driscoll_kraay_cov, _p_value_from_t
from factrix._stats.constants import auto_bartlett
from factrix.metrics.fm_beta import pooled_beta

T = 120
N_ASSETS = 20
H = 21
N_REPS = 80


def _null_panel(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    common_x = np.zeros(T)
    idio_x = np.zeros((T, N_ASSETS))
    for t in range(1, T):
        common_x[t] = 0.8 * common_x[t - 1] + rng.normal()
        idio_x[t] = 0.8 * idio_x[t - 1] + rng.normal(size=N_ASSETS)
    factor = np.sqrt(0.3) * common_x[:, None] + np.sqrt(0.7) * idio_x

    common_e = rng.normal(size=T + H)
    idio_e = rng.normal(size=(T + H, N_ASSETS))
    shocks = np.sqrt(0.5) * common_e[:, None] + np.sqrt(0.5) * idio_e
    forward_return = np.stack(
        [shocks[t + 1 : t + H + 1].sum(axis=0) / np.sqrt(H) for t in range(T)]
    )

    return pl.DataFrame(
        {
            "date": np.repeat(np.arange(T), N_ASSETS),
            "asset_id": np.tile(np.arange(N_ASSETS), T),
            "factor": factor.ravel(),
            "forward_return": forward_return.ravel(),
        }
    )


def _legacy_auto_p_value(panel: pl.DataFrame) -> float:
    x = panel["factor"].to_numpy()
    y = panel["forward_return"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov, n_periods, _ = _driscoll_kraay_cov(
        X,
        resid,
        panel["date"].to_numpy(),
        lags=auto_bartlett(T),
    )
    t_stat = float(beta[1] / np.sqrt(cov[1, 1]))
    return _p_value_from_t(t_stat, n_periods)


def test_overlap_calibration_reduces_null_over_rejection() -> None:
    corrected_rejections = 0
    legacy_rejections = 0
    for rep in range(N_REPS):
        panel = _null_panel(20260831 + rep)
        corrected = pooled_beta(
            panel,
            driscoll_kraay=True,
            overlap_periods=H,
            expected_warnings=("hac_bandwidth_ill_conditioned",),
        )
        corrected_rejections += corrected.p_value < 0.05
        legacy_rejections += _legacy_auto_p_value(panel) < 0.05

    corrected_rate = corrected_rejections / N_REPS
    legacy_rate = legacy_rejections / N_REPS
    assert corrected_rate <= 0.12
    assert legacy_rate >= 0.13
    assert corrected_rate < legacy_rate
