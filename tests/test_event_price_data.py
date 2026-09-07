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


def test_by_slice_routes_full_price_data_without_expanding_event_sample() -> None:
    n_dates = 100
    raw = _event_panel(n_assets=12, n_dates=n_dates)
    raw = raw.hstack(
        pl.DataFrame(
            {
                "cohort": [
                    cohort for cohort in ("first", "second") for _ in range(6 * n_dates)
                ]
            }
        )
    )
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    truncated = fx.by_slice(
        panel,
        event_around_return(offsets=[24]),
        by="cohort",
        factor_col="factor",
        strict=False,
    )
    restored = fx.by_slice(
        panel,
        event_around_return(offsets=[24]),
        by="cohort",
        factor_col="factor",
        price_data=raw,
        strict=False,
    )

    for cohort in ("first", "second"):
        truncated_audit = (
            truncated[cohort].metrics["event_around_return"].metadata["per_offset"][24]
        )
        restored_metric = restored[cohort].metrics["event_around_return"]
        restored_audit = restored_metric.metadata["per_offset"][24]
        # by_slice restricts price_data to the slice's own assets, so the
        # reference call is the one a caller would write by hand for this
        # cohort: the same evaluation rows against the same price scope.
        # Comparing against the whole raw panel instead would agree only
        # because both sides read a baseline formed from the other cohort.
        cohort_panel = panel.filter(pl.col("cohort") == cohort).drop("cohort")
        cohort_prices = raw.filter(
            pl.col("asset_id").is_in(cohort_panel["asset_id"].unique().implode())
        )
        direct_metric = fx.evaluate(
            cohort_panel,
            price_data=cohort_prices,
            metrics={"path": event_around_return(offsets=[24])},
            factor_cols=["factor"],
            strict=False,
        )["factor"].metrics["path"]

        assert truncated_audit["computed"] == 0
        assert restored_audit == direct_metric.metadata["per_offset"][24]
        assert restored_audit["eligible"] == 6
        assert restored_audit["computed"] == 6
        assert restored_audit["censored"] == 0
        assert restored_metric.n_obs == 6
        assert restored[cohort].n_assets == 6
        assert restored[cohort].n_periods == panel["date"].n_unique()


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


def test_duplicate_price_rows_are_explained_as_a_price_grid_defect() -> None:
    """A duplicated price row is not a fabricated forward return.

    ``price_data`` never feeds the forward-return shift, so the panel
    validator's explanation of a duplicate key names the wrong quantity
    on this path. The price grid's own failure is that one period holds
    two prices, and the excursion walk cannot say which one it entered
    at.
    """
    raw = _event_panel(n_assets=1)
    duplicated = pl.concat([raw, raw.head(1)])

    with pytest.raises(UserInputError) as excinfo:
        compute_event_returns(raw, price_data=duplicated, offsets=[1])

    message = str(excinfo.value)
    assert "price_data" in message
    assert "The forward return shifts by row position" not in message


def _two_cohort_panel(
    *,
    slow_growth: float = 1.0005,
    fast_growth: float = 1.008,
    n_dates: int = 120,
    event_at: int = 60,
) -> pl.DataFrame:
    """Two cohorts drifting at very different rates, no event information.

    Every price path is pure drift, so the correct excess return is zero at
    every offset in both cohorts. Any residue is a benchmark formed from
    the wrong sample.
    """
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(n_dates)]
    rows: list[dict[str, object]] = []
    for cohort, growth, assets in (
        ("slow", slow_growth, range(0, 6)),
        ("fast", fast_growth, range(6, 12)),
    ):
        for asset_index in assets:
            for index, current_date in enumerate(dates):
                rows.append(
                    {
                        "date": current_date,
                        "asset_id": f"A{asset_index}",
                        "cohort": cohort,
                        "factor": 1.0 if index == event_at else 0.0,
                        "price": 100.0 * growth**index,
                    }
                )
    return pl.DataFrame(rows)


def test_by_slice_baseline_is_formed_from_the_slice_not_the_price_panel() -> None:
    """A slice is benchmarked against its own drift, with or without prices.

    ``price_data`` supplies the price grid; it must not also decide which
    assets the unconditional baseline is formed from. Forwarding the whole
    panel hands every slice the pooled drift of assets it does not contain,
    which prices a pure-drift panel as event alpha.
    """
    raw = _two_cohort_panel()
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    without = fx.by_slice(
        panel,
        event_around_return(offsets=[6]),
        by="cohort",
        factor_col="factor",
        strict=False,
    )
    with_prices = fx.by_slice(
        panel,
        event_around_return(offsets=[6]),
        by="cohort",
        factor_col="factor",
        price_data=raw,
        strict=False,
    )

    for cohort, drift in (("slow", 0.0005), ("fast", 0.008)):
        for label, bundle in (("without", without), ("with", with_prices)):
            metadata = bundle[cohort].metrics["event_around_return"].metadata
            assert (
                metadata["baseline_bar_return"],
                metadata["n_assets_in_baseline"],
            ) == pytest.approx((drift, 6)), f"{cohort} {label} price_data"
            assert metadata["per_offset"][6]["mean"] == pytest.approx(0.0, abs=1e-12), (
                f"{cohort} {label} price_data prices pure drift as event alpha"
            )


def test_by_slice_raggedness_is_measured_on_the_slice_not_the_price_panel() -> None:
    """One ragged asset does not make every other slice ragged.

    The warning describes the sample the metric ran on. A dense slice that
    is told its grid is ragged sends the reader to reindex a panel that is
    already dense.
    """
    raw = _two_cohort_panel(slow_growth=1.002, fast_growth=1.002)
    ragged_raw = raw.filter(
        ~(
            (pl.col("asset_id") == "A11")
            & (pl.col("date") == raw["date"].unique().sort()[30])
        )
    )
    panel = fx.preprocess.compute_forward_return(ragged_raw, forward_periods=5)

    bundle = fx.by_slice(
        panel,
        event_around_return(offsets=[6]),
        by="cohort",
        factor_col="factor",
        price_data=ragged_raw,
        expected_warnings=("ragged_period_grid",),
        strict=False,
    )

    assert bundle["slow"].metrics["event_around_return"].warning_codes == ()
    assert bundle["fast"].metrics["event_around_return"].warning_codes == (
        "ragged_period_grid",
    )


def _regime_panel(
    *,
    n_assets: int = 8,
    n_dates: int = 100,
    switch_at: int = 50,
    calm_growth: float = 1.0002,
    hot_growth: float = 1.010,
) -> pl.DataFrame:
    """One panel, two regimes on the date axis, no event information.

    Every asset compounds at the regime's own rate, so within each regime
    the correct excess return is zero at every offset. A residue is the
    benchmark being formed from the other regime's drift.
    """
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(n_dates)]
    rows: list[dict[str, object]] = []
    for asset_index in range(n_assets):
        price = 100.0
        for index, current_date in enumerate(dates):
            price *= calm_growth if index < switch_at else hot_growth
            rows.append(
                {
                    "date": current_date,
                    "asset_id": f"A{asset_index}",
                    "regime": "calm" if index < switch_at else "hot",
                    "factor": 1.0 if index in (20, 70) else 0.0,
                    "price": price,
                }
            )
    return pl.DataFrame(rows)


def test_by_slice_date_axis_baseline_stays_inside_the_slice() -> None:
    """A regime is benchmarked against its own drift, not the whole span.

    Restricting the forwarded price panel by asset is a no-op on a
    date-axis partition: every slice holds every asset. The price panel
    then supplies periods belonging to the *other* regimes, and the
    unconditional baseline is pooled across all of them.
    """
    raw = _regime_panel()
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    bundle = fx.by_slice(
        panel,
        event_around_return(offsets=[6]),
        by="regime",
        factor_col="factor",
        price_data=raw,
        expected_warnings=("slice_boundary_truncation",),
        strict=False,
    )

    for regime, drift in (("calm", 0.0002), ("hot", 0.010)):
        metadata = bundle[regime].metrics["event_around_return"].metadata
        assert metadata["baseline_bar_return"] == pytest.approx(drift, rel=1e-9), (
            f"{regime} was benchmarked against {metadata['baseline_bar_return']}"
        )
        assert metadata["per_offset"][6]["mean"] == pytest.approx(0.0, abs=1e-12), (
            f"{regime} prices its own drift as event alpha"
        )


def test_by_slice_cross_sectional_keeps_the_price_tail() -> None:
    """The date restriction must not fire on a cross-sectional partition.

    A sector slice spans every period, so its price panel keeps the tail
    `compute_forward_return` dropped — the offset that tail restores is
    the whole point of forwarding prices at all.
    """
    raw = _event_panel(n_assets=12, n_dates=100)
    raw = raw.with_columns(
        pl.when(pl.col("asset_id").is_in(pl.Series([f"A{i}" for i in range(6)])))
        .then(pl.lit("first"))
        .otherwise(pl.lit("second"))
        .alias("cohort")
    )
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    bundle = fx.by_slice(
        panel,
        event_around_return(offsets=[24]),
        by="cohort",
        factor_col="factor",
        price_data=raw,
        strict=False,
    )

    for cohort in ("first", "second"):
        audit = bundle[cohort].metrics["event_around_return"].metadata["per_offset"][24]
        assert (audit["computed"], audit["censored"]) == (6, 0)
