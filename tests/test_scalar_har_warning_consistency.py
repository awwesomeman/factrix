"""Warning parity across consumers of the scalar HAR bandwidth policy."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.inference.series_mean import NeweyWest
from factrix.metrics.common_asymmetry import common_asymmetry
from factrix.metrics.common_quantile import common_quantile_spread
from factrix.metrics.fm_beta import pooled_beta
from factrix.metrics.predictive_beta import predictive_beta
from factrix.metrics.spanning import spanning_alpha


def _series_panel(factor: np.ndarray, returns: np.ndarray) -> pl.DataFrame:
    n = len(factor)
    return pl.DataFrame(
        {
            "date": np.arange(n),
            "asset_id": np.zeros(n, dtype=int),
            "factor": factor,
            "forward_return": returns,
        }
    )


def _pooled_panel(n: int, rng: np.random.Generator) -> pl.DataFrame:
    n_assets = 8
    common = rng.standard_normal(n)
    factor = np.repeat(common, n_assets) + rng.standard_normal(n * n_assets)
    returns = (
        0.2 * factor + np.repeat(common, n_assets) + rng.standard_normal(n * n_assets)
    )
    return pl.DataFrame(
        {
            "date": np.repeat(np.arange(n), n_assets),
            "asset_id": np.tile(np.arange(n_assets), n),
            "factor": factor,
            "forward_return": returns,
        }
    )


def _has_bandwidth_warning(result: object) -> bool:
    warning_codes = getattr(result, "warning_codes", ())
    return WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED.value in warning_codes


@pytest.mark.parametrize(
    ("n", "overlap_periods", "expected"),
    [(120, 21, True), (240, 5, False)],
)
def test_scalar_har_consumers_share_the_bandwidth_warning_policy(
    n: int, overlap_periods: int, expected: bool
) -> None:
    rng = np.random.default_rng(20260831 + n)
    factor = rng.standard_normal(n)
    returns = 0.2 * factor + rng.standard_normal(n)
    panel = _series_panel(factor, returns)
    expected_warnings = (
        WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED.value,
        WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE.value,
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value,
    )

    base = rng.standard_normal(n)
    candidate = 0.3 * base + rng.standard_normal(n)
    spread_dates = np.arange(n)
    spanning = spanning_alpha(
        pl.DataFrame({"date": spread_dates, "spread": candidate}),
        base_spreads={"base": pl.DataFrame({"date": spread_dates, "spread": base})},
        overlap_periods=overlap_periods,
        expected_warnings=expected_warnings,
    )
    results = (
        common_asymmetry(
            panel,
            overlap_periods=overlap_periods,
            expected_warnings=expected_warnings,
        ),
        common_quantile_spread(
            panel,
            overlap_periods=overlap_periods,
            expected_warnings=expected_warnings,
        ),
        predictive_beta(
            panel,
            overlap_periods=overlap_periods,
            expected_warnings=expected_warnings,
        ),
        pooled_beta(
            _pooled_panel(n, rng),
            driscoll_kraay=True,
            overlap_periods=overlap_periods,
            expected_warnings=expected_warnings,
        ),
        spanning,
    )

    assert all(_has_bandwidth_warning(result) is expected for result in results)

    series_result = NeweyWest().compute(
        pl.DataFrame({"date": np.arange(n), "value": returns}),
        value_col="value",
        overlap_periods=overlap_periods,
    )
    assert (
        WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED in series_result.warnings
    ) is expected
