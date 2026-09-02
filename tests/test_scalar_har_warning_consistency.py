"""Warning parity across consumers of the scalar HAR bandwidth policy."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._stats import _MIN_PERIODS_PER_LAG
from factrix.inference.series_mean import NeweyWest
from factrix.metrics._helpers import (
    _INFERENCE_CODE_MESSAGE,
    _emit_scalar_har_warnings,
)
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


def _bandwidth_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    code = WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED.value
    return [str(item.message) for item in caught if code in str(item.message)]


def test_bandwidth_warning_text_uses_the_policy_constant() -> None:
    token = f"n_periods / {_MIN_PERIODS_PER_LAG}"
    template = _INFERENCE_CODE_MESSAGE[WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED]
    rendered = template.format(
        n=120,
        warn=30,
        autocorr=0.3,
        periods_per_lag=_MIN_PERIODS_PER_LAG,
    )
    assert token in rendered

    with warnings.catch_warnings(record=True) as scalar_caught:
        warnings.simplefilter("always")
        _emit_scalar_har_warnings(
            metric_name="test_metric",
            subject="the test series",
            n_periods=120,
            overlap_periods=21,
            persistent=False,
            bandwidth_ill_conditioned=True,
        )
    scalar_messages = _bandwidth_messages(scalar_caught)
    assert scalar_messages
    assert all(token in message for message in scalar_messages)

    rng = np.random.default_rng(1026)
    panel = _series_panel(rng.standard_normal(120), rng.standard_normal(120))
    with warnings.catch_warnings(record=True) as predictive_caught:
        warnings.simplefilter("always")
        predictive_beta(panel, overlap_periods=21)
    predictive_messages = _bandwidth_messages(predictive_caught)
    assert predictive_messages
    # ``predictive_beta`` names ``n_periods_finite``, not ``n_periods``: it is
    # the one metric whose ``metadata["n_periods"]`` is the truncated
    # augmented-design row count rather than the count this screen reads, so
    # the shared token would send a reader to the wrong key (see the metadata
    # contract tests in ``tests/stats``). What has to hold everywhere is that
    # the policy constant is interpolated rather than written out by hand.
    policy_token = f"/ {_MIN_PERIODS_PER_LAG}"
    assert all(policy_token in message for message in predictive_messages)
    assert all("n_periods_finite /" in message for message in predictive_messages)


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

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
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


@pytest.mark.parametrize(("n", "expected"), [(100, True), (180, False)])
def test_scalar_har_contrasts_share_the_effective_sample_warning(
    n: int, expected: bool
) -> None:
    rng = np.random.default_rng(1012 + n)
    factor = rng.standard_normal(n)
    returns = 0.2 * factor + rng.standard_normal(n)
    panel = _series_panel(factor, returns)
    declared = tuple(code.value for code in WarningCode)
    base = rng.standard_normal(n)
    candidate = 0.3 * base + rng.standard_normal(n)
    dates = np.arange(n)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        results = (
            common_asymmetry(panel, overlap_periods=5, expected_warnings=declared),
            common_quantile_spread(
                panel, overlap_periods=5, expected_warnings=declared
            ),
            predictive_beta(panel, overlap_periods=5, expected_warnings=declared),
            spanning_alpha(
                pl.DataFrame({"date": dates, "spread": candidate}),
                base_spreads={"base": pl.DataFrame({"date": dates, "spread": base})},
                overlap_periods=5,
                expected_warnings=declared,
            ),
        )

    code = WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value
    assert all((code in result.warning_codes) is expected for result in results)


def test_scalar_har_persistence_echoes_are_declarable_across_metrics() -> None:
    rng = np.random.default_rng(1012)
    n = 180
    factor = np.empty(n)
    factor[0] = rng.standard_normal()
    for idx in range(1, n):
        factor[idx] = 0.95 * factor[idx - 1] + rng.standard_normal()
    returns = 0.2 * factor + rng.standard_normal(n)
    panel = _series_panel(factor, returns)
    dates = np.arange(n)
    base = rng.standard_normal(n)
    declared = (WarningCode.SERIAL_CORRELATION_DETECTED.value,)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        results = (
            common_asymmetry(panel, overlap_periods=5, expected_warnings=declared),
            common_quantile_spread(
                panel, overlap_periods=5, expected_warnings=declared
            ),
            spanning_alpha(
                pl.DataFrame({"date": dates, "spread": factor}),
                base_spreads={"base": pl.DataFrame({"date": dates, "spread": base})},
                overlap_periods=5,
                expected_warnings=declared,
            ),
        )

    assert all(declared[0] in result.warning_codes for result in results)
