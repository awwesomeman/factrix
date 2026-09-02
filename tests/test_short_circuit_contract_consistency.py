"""Cross-metric contracts for descriptive and sample-shortfall results."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime

import factrix as fx
import polars as pl
import pytest
from factrix._errors import InsufficientSampleError, UserInputError
from factrix._results import MetricResult
from factrix.metrics.clustering_hhi import clustering_hhi
from factrix.metrics.common_beta import (
    common_beta_profile,
    common_beta_r_squared,
    common_beta_sign_consistency,
)
from factrix.metrics.event_quality import event_skewness, profit_factor, signal_density
from factrix.metrics.fm_beta import fm_beta_sign_consistency
from factrix.metrics.mfe_mae import mfe_mae
from factrix.metrics.tradability import notional_turnover, rank_turnover


def _panel(*, event: bool = False, n_assets: int = 3, n_dates: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, day + 1)
                for day in range(n_dates)
                for _ in range(n_assets)
            ],
            "asset_id": [f"A{i}" for _ in range(n_dates) for i in range(n_assets)],
            "factor": [1.0 if event else 0.0] * n_assets * n_dates,
            "forward_return": [0.01] * n_assets * n_dates,
        }
    )


def _empty_frame(**schema: pl.DataType) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


CaseFactory = Callable[[], MetricResult]


@pytest.mark.parametrize(
    ("factory", "reason", "error_type"),
    [
        pytest.param(
            lambda: clustering_hhi(_panel()),
            "insufficient_events",
            InsufficientSampleError,
            id="clustering_hhi",
        ),
        pytest.param(
            lambda: signal_density(_panel()),
            "insufficient_events",
            InsufficientSampleError,
            id="signal_density",
        ),
        pytest.param(
            lambda: profit_factor(_panel(event=True)),
            "insufficient_events",
            InsufficientSampleError,
            id="profit_factor",
        ),
        pytest.param(
            lambda: event_skewness(_panel(event=True)),
            "insufficient_events",
            InsufficientSampleError,
            id="event_skewness",
        ),
        pytest.param(
            lambda: mfe_mae(
                _empty_frame(
                    date=pl.Datetime("ms"),
                    asset_id=pl.String,
                    mfe=pl.Float64,
                    mae=pl.Float64,
                )
            ),
            "no_price_data",
            UserInputError,
            id="mfe_mae",
        ),
        pytest.param(
            lambda: mfe_mae(
                pl.DataFrame(
                    {
                        "date": [datetime(2024, 1, 1)],
                        "asset_id": ["A"],
                        "mfe": [0.1],
                        "mae": [-0.1],
                    }
                )
            ),
            "insufficient_events",
            InsufficientSampleError,
            id="mfe_mae_thin_events",
        ),
        pytest.param(
            lambda: rank_turnover(_panel()),
            "insufficient_rank_turnover_periods",
            InsufficientSampleError,
            id="rank_turnover",
        ),
        pytest.param(
            lambda: rank_turnover(_panel(event=True, n_dates=5)),
            "insufficient_rank_turnover_periods",
            InsufficientSampleError,
            id="rank_turnover_no_rank_dispersion",
        ),
        pytest.param(
            lambda: notional_turnover(_panel(n_assets=10)),
            "insufficient_notional_turnover_periods",
            InsufficientSampleError,
            id="notional_turnover",
        ),
        pytest.param(
            lambda: notional_turnover(_panel(n_assets=4, n_dates=5), rebalance_lag=1),
            "insufficient_assets_for_quantile_groups",
            InsufficientSampleError,
            id="notional_turnover_unfilled_groups",
        ),
        pytest.param(
            lambda: common_beta_profile(
                _empty_frame(asset_id=pl.String, beta=pl.Float64)
            ),
            "insufficient_asset_beta_observations",
            InsufficientSampleError,
            id="common_beta_profile",
        ),
        pytest.param(
            lambda: common_beta_r_squared(
                _empty_frame(asset_id=pl.String, r_squared=pl.Float64)
            ),
            "insufficient_asset_r_squared_observations",
            InsufficientSampleError,
            id="common_beta_r_squared",
        ),
        pytest.param(
            lambda: common_beta_sign_consistency(
                _empty_frame(asset_id=pl.String, beta=pl.Float64)
            ),
            "insufficient_assets_for_sign_consistency",
            InsufficientSampleError,
            id="common_beta_sign_consistency",
        ),
        pytest.param(
            lambda: fm_beta_sign_consistency(
                _empty_frame(date=pl.Datetime("ms"), beta=pl.Float64)
            ),
            "insufficient_fm_beta_observations",
            InsufficientSampleError,
            id="fm_beta_sign_consistency",
        ),
    ],
)
def test_descriptive_short_circuits_share_shape_reason_and_exception(
    factory: CaseFactory,
    reason: str,
    error_type: type[Exception],
) -> None:
    result = factory()

    assert math.isnan(result.value)
    assert result.p_value is None
    assert result.alternative is None
    assert result.stat is None
    assert result.metadata["reason"] == reason

    with pytest.raises(error_type):
        fx._enforce_strict({"metric": result})
