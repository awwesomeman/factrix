"""Reproducible size grid for pooled Driscoll-Kraay inference.

The panel null combines an AR(1) regressor, cross-sectional common shocks, and
overlapping forward returns. The ordinary test runs one reduced-replication
cell; executing this module runs the complete 1,000-replication research grid
behind ``inference-calibration.md`` and prints a Markdown table::

    python tests/stats/test_driscoll_kraay_size.py

Every cell uses 20 assets and seed ``20260831 + rep``. AR recursion always
runs, including ``phi=0``; zero common share multiplies an independently drawn
common shock by zero rather than changing the RNG draw order. Those details
are explicit because reconstructing them differently moved null size by
several percentage points in small cells.

The reproducible 1,000-replication reference is:

| T | h | phi_x | return / factor common share | former | calibrated |
|---:|---:|---:|---:|---:|---:|
| 60 | 1 | 0.0 | 0.0 / 0.0 | 8.4% | 5.7% |
| 60 | 5 | 0.8 | 0.5 / 0.3 | 16.4% | 8.7% |
| 120 | 5 | 0.8 | 0.5 / 0.3 | 12.2% | 7.5% |
| 120 | 21 | 0.0 | 0.5 / 0.3 | 6.3% | 2.6% |
| 120 | 21 | 0.8 | 0.5 / 0.3 | 22.5% | 4.9% |
| 240 | 21 | 0.8 | 0.5 / 0.3 | 19.0% | 4.4% |
| 120 | 5 | 0.8 | 0.8 / 0.8 | 12.7% | 7.4% |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from factrix._stats import _driscoll_kraay_cov, _p_value_from_t
from factrix._stats.constants import auto_bartlett
from factrix.metrics.fm_beta import pooled_beta

N_ASSETS = 20
N_REPS = 80
REFERENCE_REPS = 1_000
SEED = 20260831


@dataclass(frozen=True, slots=True)
class CalibrationCell:
    """One row of the public DK calibration grid."""

    n_periods: int
    horizon: int
    factor_phi: float
    return_common_share: float
    factor_common_share: float


CALIBRATION_GRID = (
    CalibrationCell(60, 1, 0.0, 0.0, 0.0),
    CalibrationCell(60, 5, 0.8, 0.5, 0.3),
    CalibrationCell(120, 5, 0.8, 0.5, 0.3),
    CalibrationCell(120, 21, 0.0, 0.5, 0.3),
    CalibrationCell(120, 21, 0.8, 0.5, 0.3),
    CalibrationCell(240, 21, 0.8, 0.5, 0.3),
    CalibrationCell(120, 5, 0.8, 0.8, 0.8),
)
SMOKE_CELL = CALIBRATION_GRID[4]


def _null_panel(
    seed: int,
    cell: CalibrationCell = SMOKE_CELL,
    *,
    n_assets: int = N_ASSETS,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    T = cell.n_periods
    H = cell.horizon

    common_x = np.zeros(T)
    idio_x = np.zeros((T, n_assets))
    for t in range(1, T):
        common_x[t] = cell.factor_phi * common_x[t - 1] + rng.normal()
        idio_x[t] = cell.factor_phi * idio_x[t - 1] + rng.normal(size=n_assets)
    factor = (
        np.sqrt(cell.factor_common_share) * common_x[:, None]
        + np.sqrt(1.0 - cell.factor_common_share) * idio_x
    )

    common_e = rng.normal(size=T + H)
    idio_e = rng.normal(size=(T + H, n_assets))
    shocks = (
        np.sqrt(cell.return_common_share) * common_e[:, None]
        + np.sqrt(1.0 - cell.return_common_share) * idio_e
    )
    forward_return = np.stack(
        [shocks[t + 1 : t + H + 1].sum(axis=0) / np.sqrt(H) for t in range(T)]
    )

    return pl.DataFrame(
        {
            "date": np.repeat(np.arange(T), n_assets),
            "asset_id": np.tile(np.arange(n_assets), T),
            "factor": factor.ravel(),
            "forward_return": forward_return.ravel(),
        }
    )


def _legacy_auto_p_value(panel: pl.DataFrame, cell: CalibrationCell) -> float:
    x = panel["factor"].to_numpy()
    y = panel["forward_return"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov, n_periods, _ = _driscoll_kraay_cov(
        X,
        resid,
        panel["date"].to_numpy(),
        lags=auto_bartlett(cell.n_periods),
    )
    t_stat = float(beta[1] / np.sqrt(cov[1, 1]))
    return _p_value_from_t(t_stat, n_periods)


def _rejection_rates(
    cell: CalibrationCell,
    n_reps: int,
    *,
    n_assets: int = N_ASSETS,
) -> tuple[float, float]:
    corrected_rejections = 0
    legacy_rejections = 0
    for rep in range(n_reps):
        panel = _null_panel(SEED + rep, cell, n_assets=n_assets)
        corrected = pooled_beta(
            panel,
            driscoll_kraay=True,
            overlap_periods=cell.horizon,
            expected_warnings=(
                "hac_bandwidth_ill_conditioned",
                "unreliable_se_short_periods",
            ),
        )
        corrected_rejections += corrected.p_value < 0.05
        legacy_rejections += _legacy_auto_p_value(panel, cell) < 0.05

    return corrected_rejections / n_reps, legacy_rejections / n_reps


def test_overlap_calibration_reduces_null_over_rejection() -> None:
    corrected_rate, legacy_rate = _rejection_rates(SMOKE_CELL, N_REPS)

    # The 1,000-replication reference cell is approximately 5% corrected
    # versus 23% under the former path. These reduced-replication bounds are
    # deliberately wider than three Monte-Carlo standard errors.
    assert corrected_rate <= 0.10
    assert legacy_rate >= 0.15
    assert corrected_rate < legacy_rate


def _print_reference_grid() -> None:
    """Run and print every documented cell at the research replication count."""
    print("| T | h | phi_x | return / factor common share | former | calibrated |")
    print("|---:|---:|---:|---:|---:|---:|")
    for cell in CALIBRATION_GRID:
        corrected, former = _rejection_rates(cell, REFERENCE_REPS)
        print(
            f"| {cell.n_periods} | {cell.horizon} | {cell.factor_phi:.1f} | "
            f"{cell.return_common_share:.1f} / {cell.factor_common_share:.1f} | "
            f"{former:.1%} | {corrected:.1%} |"
        )


if __name__ == "__main__":
    _print_reference_grid()
