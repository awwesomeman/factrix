"""Full price paths stay separate from the forward-return sample (#1051)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import factrix as fx
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix.metrics import ic
from factrix.metrics.event_horizon import compute_event_returns, event_around_return
from factrix.metrics.mfe_mae import compute_mfe_mae, mfe_mae


def _event_panel(
    *,
    n_assets: int = 6,
    n_dates: int = 100,
    event_at: int = 70,
) -> pl.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(n_dates)]
    rows: list[dict[str, object]] = []
    for asset_index in range(n_assets):
        growth = 1.004 + asset_index * 0.0001
        for index, current_date in enumerate(dates):
            rows.append(
                {
                    "date": current_date,
                    "asset_id": f"A{asset_index}",
                    "factor": 1.0 if index == event_at else 0.0,
                    "price": 100.0 * growth**index,
                }
            )
    return pl.DataFrame(rows)


def test_full_price_data_restores_offset_lost_from_forward_return_tail() -> None:
    raw = _event_panel(n_assets=1)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    truncated = compute_event_returns(panel, offsets=[24])
    restored = compute_event_returns(panel, price_data=raw, offsets=[24])

    assert truncated.is_empty()
    assert restored.height == 1
    prices = raw["price"].to_list()
    assert restored["signed_return"][0] == pytest.approx(prices[95] / prices[71] - 1)


def test_direct_path_primitives_honor_custom_price_column() -> None:
    raw = _event_panel(n_assets=1).rename({"price": "close"})
    panel = raw.with_columns(pl.lit(0.0).alias("forward_return"))

    event_returns = compute_event_returns(
        panel.drop("close"),
        price_data=raw,
        offsets=[1],
        price_col="close",
    )
    excursions = compute_mfe_mae(
        panel.drop("close"),
        price_data=raw,
        price_col="close",
        window=5,
        estimation_window=20,
    )

    assert event_returns.height == 1
    assert excursions.height == 1
    assert excursions["path_status"][0] == "computed"


def test_evaluate_routes_full_price_data_and_reports_offset_audit() -> None:
    raw = _event_panel()
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    result = fx.evaluate(
        panel,
        price_data=raw,
        metrics={"path": event_around_return(offsets=[-1, 24])},
        factor_cols=["factor"],
        strict=False,
    )["factor"].metrics["path"]

    audit = result.metadata["per_offset"][24]
    assert audit["eligible"] == 6
    assert audit["computed"] == 6
    assert audit["censored"] == 0
    assert audit["censor_reasons"] == {}
    assert audit["n"] == 6
    assert audit["mean"] is not None
    assert result.n_obs == 6


def test_mfe_mae_uses_full_price_data_without_entering_return_sample() -> None:
    raw = _event_panel()
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    result = fx.evaluate(
        panel,
        price_data=raw,
        metrics={"mfe": mfe_mae()},
        factor_cols=["factor"],
        strict=False,
    )["factor"].metrics["mfe"]

    assert result.n_obs == 6
    assert result.metadata["n_events_eligible"] == 6
    assert result.metadata["n_events_computed"] == 6
    assert result.metadata["n_events_censored"] == 0
    assert result.metadata["censor_reasons"] == {}


def test_price_data_does_not_change_forward_return_metric_sample() -> None:
    raw = fx.datasets.make_cs_panel(n_assets=40, n_dates=80, rng=1051)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    without_prices = fx.evaluate(
        panel,
        metrics={"ic": ic()},
        factor_cols=["factor"],
    )["factor"].metrics["ic"]
    with_prices = fx.evaluate(
        panel,
        price_data=raw,
        metrics={"ic": ic()},
        factor_cols=["factor"],
    )["factor"].metrics["ic"]

    assert with_prices.value == without_prices.value
    assert with_prices.p_value == without_prices.p_value
    assert with_prices.n_obs == without_prices.n_obs
    assert with_prices.metadata == without_prices.metadata


def test_full_price_grid_is_independent_of_coarser_evaluation_grid() -> None:
    raw = _event_panel(n_assets=1, event_at=60)
    grid = raw["date"].unique().sort()
    panel = fx.preprocess.compute_forward_return(
        raw,
        forward_periods=5,
        dates=grid.gather_every(10),
    )

    restored = compute_event_returns(panel, price_data=raw, offsets=[24])

    assert panel["date"].n_unique() < raw["date"].n_unique()
    assert restored.height == 1
    prices = raw["price"].to_list()
    assert restored["signed_return"][0] == pytest.approx(prices[85] / prices[61] - 1)


def test_ragged_price_path_reports_computed_and_censored_offsets() -> None:
    raw = _event_panel(n_assets=2, n_dates=50, event_at=20)
    missing_exit = date(2020, 1, 1) + timedelta(days=27)
    ragged_prices = raw.filter(
        ~((pl.col("asset_id") == "A1") & (pl.col("date") == missing_exit))
    )
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=2)

    result = event_around_return(
        panel,
        price_data=ragged_prices,
        offsets=[6],
    )

    audit = result.metadata["per_offset"][6]
    assert audit["eligible"] == 2
    assert audit["computed"] == 1
    assert audit["censored"] == 1
    assert audit["censor_reasons"] == {"missing_exit_price": 1}


def test_mfe_mae_retains_censored_events_with_reason() -> None:
    raw = _event_panel(n_assets=4, n_dates=20, event_at=19)
    event_data = raw.with_columns(pl.lit(0.0).alias("forward_return"))

    paths = compute_mfe_mae(event_data, price_data=raw, window=5)
    result = mfe_mae(paths)

    assert paths.height == 4
    assert paths["path_status"].unique().to_list() == ["censored"]
    assert paths["censor_reason"].unique().to_list() == ["window_out_of_bounds"]
    assert math.isnan(result.value)
    assert result.metadata["reason"] == "no_complete_event_paths"
    assert result.metadata["n_events_eligible"] == 4
    assert result.metadata["n_events_computed"] == 0
    assert result.metadata["n_events_censored"] == 4
    assert result.metadata["censor_reasons"] == {"window_out_of_bounds": 4}


def test_mfe_mae_counts_mixed_computed_and_censored_paths() -> None:
    raw = _event_panel(n_assets=6)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
    incomplete_prices = raw.filter(pl.col("asset_id") != "A5")

    paths = compute_mfe_mae(panel, price_data=incomplete_prices, window=5)
    result = mfe_mae(paths)

    assert paths.height == 6
    assert result.metadata["n_events_eligible"] == 6
    assert result.metadata["n_events_computed"] == 5
    assert result.metadata["n_events_censored"] == 1
    assert result.metadata["censor_reasons"] == {"asset_not_in_price_data": 1}
    assert result.n_obs == 5


def test_mfe_mae_sample_floor_counts_only_computed_paths() -> None:
    raw = _event_panel(n_assets=4)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
    incomplete_prices = raw.filter(pl.col("asset_id") != "A3")

    result = fx.evaluate(
        panel,
        price_data=incomplete_prices,
        metrics={"mfe": mfe_mae()},
        factor_cols=["factor"],
        strict=False,
    )["factor"].metrics["mfe"]

    assert math.isnan(result.value)
    assert result.metadata["reason"] == "insufficient_events"
    assert result.metadata["n_events_eligible"] == 4
    assert result.metadata["n_events_computed"] == 3
    assert result.metadata["n_events_censored"] == 1
    assert result.n_obs == 3


def test_evaluate_rejects_misaligned_price_key_dtype() -> None:
    raw = _event_panel()
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
    misaligned = raw.with_columns(pl.col("date").cast(pl.Datetime("us")))

    with pytest.raises(UserInputError) as excinfo:
        fx.evaluate(
            panel,
            price_data=misaligned,
            metrics={"path": event_around_return()},
            factor_cols=["factor"],
        )

    assert excinfo.value.func_name == "evaluate"
    assert excinfo.value.field == "price_data.date"


def test_evaluate_horizons_automatically_preserves_full_price_grid() -> None:
    raw = _event_panel()

    results = fx.evaluate_horizons(
        raw,
        metrics={"path": event_around_return(offsets=[-1, 24])},
        factor_cols=["factor"],
        forward_periods=[5, 10],
        strict=False,
    )

    assert len(results) == 2
    for result in results:
        audit = result.metrics["path"].metadata["per_offset"][24]
        assert audit["eligible"] == 6
        assert audit["computed"] == 6
        assert audit["censored"] == 0
