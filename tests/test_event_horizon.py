"""Tests for factrix.metrics.event_horizon and signal_density."""

import math
from datetime import datetime, timedelta

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix.metrics.event_horizon import (
    compute_event_returns,
    event_around_return,
)
from factrix.metrics.event_quality import signal_density

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event_with_price(
    n_assets: int = 20,
    n_dates: int = 300,
    event_prob: float = 0.03,
    signal_strength: float = 0.02,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for a in assets:
        price = 100.0
        for d in dates:
            is_event = rng.random() < event_prob
            direction = rng.choice([-1.0, 1.0]) if is_event else 0.0
            daily_ret = rng.normal(0, 0.015)
            if is_event:
                daily_ret += signal_strength * direction
            price *= 1 + daily_ret
            rows.append(
                {
                    "date": d,
                    "asset_id": a,
                    "factor": direction,
                    "forward_return": daily_ret,
                    "price": price,
                }
            )

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def event_data() -> pl.DataFrame:
    return _make_event_with_price()


@pytest.fixture
def no_price_data(event_data) -> pl.DataFrame:
    return event_data.drop("price")


# ---------------------------------------------------------------------------
# compute_event_returns
# ---------------------------------------------------------------------------


class TestComputeEventReturns:
    def test_returns_expected_columns(self, event_data):
        result = compute_event_returns(event_data, offsets=[1, 6, 12])
        assert set(result.columns) >= {"offset", "date", "asset_id", "signed_return"}
        assert len(result) > 0

    def test_multiple_offsets(self, event_data):
        result = compute_event_returns(event_data, offsets=[-3, -1, 1, 6])
        offsets_found = result["offset"].unique().sort().to_list()
        assert -3 in offsets_found
        assert 1 in offsets_found

    def test_no_price_returns_empty(self, no_price_data):
        result = compute_event_returns(no_price_data)
        assert result.is_empty()

    def test_post_event_signed(self, event_data):
        """Post-event returns are direction-adjusted (signed)."""
        # Use longer horizon for stronger density-to-noise
        result = compute_event_returns(event_data, offsets=[12])
        assert result["signed_return"].mean() > 0

    def test_pre_event_signed(self):
        """Pre-event returns are direction-adjusted, so a directional
        pre-event drift does not cancel across +/- events.

        Two assets carry a consistent pre-event drift *in the direction of
        the event*: the +1 event is preceded by a price rise, the -1 event
        by a price fall. Raw (unsigned) pre-event returns would cancel to
        ~0; direction-signed returns both read positive and add up.
        """
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(6)]
        # asset_up: prices rise before a +1 event on the last date
        # asset_dn: prices fall before a -1 event on the last date
        rows = []
        for aid, prices, direction in [
            ("asset_up", [100.0, 101.0, 102.0, 103.0, 104.0, 105.0], 1.0),
            ("asset_dn", [100.0, 99.0, 98.0, 97.0, 96.0, 95.0], -1.0),
        ]:
            for i, (d, p) in enumerate(zip(dates, prices, strict=True)):
                factor = direction if i == len(dates) - 1 else 0.0
                rows.append({"date": d, "asset_id": aid, "factor": factor, "price": p})
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

        result = compute_event_returns(panel, offsets=[-1])
        signed = result["signed_return"].to_list()
        # Both the up-event and down-event pre-returns are signed positive.
        assert len(signed) == 2
        assert all(s > 0 for s in signed)
        assert result["signed_return"].mean() > 0

    def test_output_date_dtype_mirrors_input_us(self, event_data):
        df_us = event_data.with_columns(pl.col("date").cast(pl.Datetime("us")))
        result = compute_event_returns(df_us, offsets=[1, 6])
        assert result.schema["date"] == pl.Datetime("us")

    def test_output_date_dtype_mirrors_tz_aware(self, event_data):
        df_utc = event_data.with_columns(pl.col("date").dt.replace_time_zone("UTC"))
        result = compute_event_returns(df_utc, offsets=[1, 6])
        assert result.schema["date"] == pl.Datetime("ms", time_zone="UTC")


# ---------------------------------------------------------------------------
# event_around_return
# ---------------------------------------------------------------------------


class TestEventAroundReturn:
    def test_returns_metric_output(self, event_data):
        result = event_around_return(event_data)
        assert result is not None
        assert "per_offset" in result.metadata

    def test_descriptive_no_p_value(self, event_data):
        # Descriptive multi-horizon summary: no hypothesis test runs, so the
        # contract is p_value=None — not a fabricated 1.0 placeholder.
        assert event_around_return(event_data).p_value is None

    def test_short_circuit_is_descriptive(self, no_price_data):
        assert event_around_return(no_price_data).p_value is None

    def test_per_offset_has_stats(self, event_data):
        result = event_around_return(event_data, offsets=[-3, 1, 6])
        per_offset = result.metadata["per_offset"]
        assert 1 in per_offset
        assert "mean" in per_offset[1]
        assert "hit_rate" in per_offset[1]

    def test_leakage_value_small(self, event_data):
        """Pre-event leakage should be near zero for random events."""
        result = event_around_return(event_data, offsets=[-6, -3, -1, 1])
        assert result.value < 0.01  # leakage score

    def test_short_circuit_without_price(self, no_price_data):
        result = event_around_return(no_price_data)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "no_price_data"
        assert result.metadata["per_offset"] == {}

    def test_reports_its_sample_size_on_the_event_axis(self, event_data):
        # n_obs is the library's single source of truth for sample size, and
        # this was the one event metric leaving it null — breaking the uniform
        # column when the event battery is stacked with to_frame().
        result = event_around_return(event_data)
        n_events = event_data.filter(pl.col("factor") != 0).height
        assert result.n_obs == n_events
        assert result.n_obs_axis == "events"
        assert result.metadata["n_events"] == n_events

    def test_sample_size_counts_events_not_offset_rows(self, event_data):
        # One event contributes one row per offset, so a longer offset list
        # must not inflate the count.
        few = event_around_return(event_data, offsets=[1])
        many = event_around_return(event_data, offsets=[-3, -1, 1, 6, 12])
        assert few.n_obs == many.n_obs

    def test_short_circuit_stamps_a_zero_event_sample(self, no_price_data):
        result = event_around_return(no_price_data)
        assert result.n_obs == 0
        assert result.n_obs_axis == "events"

    def test_zero_price_withholds_contaminated_baseline(self, event_data):
        poisoned = event_data.with_row_index("_row").with_columns(
            pl.when(pl.col("_row") == 0)
            .then(0.0)
            .otherwise(pl.col("price"))
            .alias("price")
        )

        result = event_around_return(poisoned.drop("_row"))

        assert math.isnan(result.value)
        assert result.p_value is None
        assert result.metadata["reason"] == "invalid_price_data"
        assert result.metadata["n_invalid_prices"] == 1
        assert result.metadata["baseline_bar_return"] is None
        assert result.metadata["per_offset"] == {}
        assert fx.WarningCode.METRIC_UNAVAILABLE.value in result.warning_codes


class TestEventMetricThroughEvaluate:
    def test_price_survives_dag_projection(self, event_data):
        # The DAG executor projects a thin per-factor view; that projection
        # must retain ``price`` so event metrics do not falsely short-circuit
        # with ``no_price_data`` when the caller did supply prices.
        res = fx.evaluate(
            event_data,
            metrics={"ear": event_around_return()},
            factor_cols=["factor"],
            forward_periods=1,
            strict=False,
        )
        m = res["factor"].metrics["ear"]
        assert m.metadata.get("reason") != "no_price_data"
        assert not math.isnan(m.value)

    def test_short_circuits_when_price_absent(self, no_price_data):
        res = fx.evaluate(
            no_price_data,
            metrics={"ear": event_around_return()},
            factor_cols=["factor"],
            forward_periods=1,
            strict=False,
        )
        m = res["factor"].metrics["ear"]
        assert m.metadata["reason"] == "no_price_data"
        assert math.isnan(m.value)


# ---------------------------------------------------------------------------
# signal_density
# ---------------------------------------------------------------------------


class TestSignalDensityCountsEveryFiringAsset:
    """An undisclosed `n >= 2` filter dropped every asset that fired once,
    understating bars-per-event on exactly the sparse triggers this metric
    describes.
    """

    @staticmethod
    def _panel(n_bars: int = 200) -> pl.DataFrame:
        rows = []
        for a in range(10):
            # A0 fires 50 times (4 bars/event); A1..A9 fire once (200 each).
            for d in range(n_bars):
                is_event = d % 4 == 0 if a == 0 else d == 100
                rows.append(
                    {
                        "date": datetime(2020, 1, 1) + timedelta(days=d),
                        "asset_id": f"A{a}",
                        "factor": 1.0 if is_event else 0.0,
                        "forward_return": 0.01,
                    }
                )
        return pl.DataFrame(rows)

    def test_single_event_assets_are_counted(self):
        result = signal_density(self._panel())
        # (200/50 + 9 * 200/1) / 10 = 180.4, not the busy asset's 4.0.
        assert result.value == pytest.approx(180.4)
        assert result.metadata["n_assets_with_events"] == 10

    def test_whole_panel_floor_still_applies(self):
        rows = [
            {
                "date": datetime(2020, 1, 1),
                "asset_id": "A",
                "factor": 1.0,
                "forward_return": 0.01,
            }
        ]
        result = signal_density(pl.DataFrame(rows))
        assert result.metadata["reason"] == "insufficient_events"


class TestSignalDensity:
    def test_returns_metric_output(self, event_data):
        result = signal_density(event_data)
        assert result.value > 0

    def test_sparse_events_large_gap(self):
        """Low event_prob → large gap between events."""
        df = _make_event_with_price(event_prob=0.005, seed=99)
        result = signal_density(df)
        assert result.value > 50  # ~200 bars between events

    def test_dense_events_small_gap(self):
        """High event_prob → small gap."""
        df = _make_event_with_price(event_prob=0.10, seed=88)
        result = signal_density(df)
        assert result.value < 20

    def test_metadata_has_counts(self, event_data):
        result = signal_density(event_data)
        assert "n_events_total" in result.metadata
        assert "mean_events_per_asset" in result.metadata


# ---------------------------------------------------------------------------
# Standalone import
# ---------------------------------------------------------------------------


class TestImports:
    def test_all_importable(self):
        from factrix.metrics import (
            compute_event_returns,
            event_around_return,
            signal_density,
        )

        assert all(
            callable(f)
            for f in [
                compute_event_returns,
                event_around_return,
                signal_density,
            ]
        )


class TestLeakageHeadline:
    """The leakage score is positive by construction; drift must not add to it,
    and a score that was never computed must not read as perfect.
    """

    @staticmethod
    def _panel(drift: float, n: int = 400, seed: int = 0) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for a in range(4):
            rets = rng.normal(drift, 0.01, n)
            prices = 100.0 * np.cumprod(1.0 + rets)
            for d in range(n):
                rows.append(
                    {
                        "date": datetime(2020, 1, 1) + timedelta(days=d),
                        "asset_id": f"A{a}",
                        "factor": 1.0 if d >= 40 and d % 17 == 0 else 0.0,
                        "price": float(prices[d]),
                    }
                )
        return pl.DataFrame(rows)

    def test_drift_does_not_inflate_the_leakage_score(self):
        # Both panels have zero true leakage; the drifting one used to score
        # ~2.5x higher purely from its trend.
        flat = event_around_return(self._panel(0.0))
        trending = event_around_return(self._panel(0.001))
        assert trending.metadata["baseline_bar_return"] > 5e-4
        assert trending.value == pytest.approx(flat.value, rel=1.5)

    @staticmethod
    def _signed_drift_panel(sign: float, n: int = 3000) -> pl.DataFrame:
        # Almost pure drift: 0.2% a period with a 0.01% noise, an event every
        # 40 periods, all carrying the same factor sign.
        rng = np.random.default_rng(0)
        rets = 0.002 + 0.0001 * rng.standard_normal(n)
        prices = 100.0 * np.cumprod(1.0 + rets)
        factor = np.zeros(n)
        factor[40::40] = sign
        return pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=d) for d in range(n)],
                "asset_id": ["A"] * n,
                "price": prices,
                "factor": factor,
            }
        )

    def test_drift_is_removed_on_both_factor_signs(self):
        # The returns are signed, so the baseline must be too. Subtracting the
        # unsigned drift from a short event's -mu scored 2 x mu (t ~ -400).
        long = event_around_return(self._signed_drift_panel(1.0))
        short = event_around_return(self._signed_drift_panel(-1.0))
        assert long.value < 1e-4
        assert short.value < 1e-4
        assert short.value == pytest.approx(long.value, abs=1e-5)
        # Pre-event offsets are single bars, so the signed baseline removes
        # their drift entirely; post-event offsets are cumulative and carry
        # k bars of it by design.
        for result in (long, short):
            for k, stats in result.metadata["per_offset"].items():
                if k < 0 and stats.get("t") is not None:
                    assert abs(stats["t"]) < 3

    def test_score_is_reported_with_its_null_scale(self):
        result = event_around_return(self._panel(0.0))
        scale = result.metadata["leakage_null_scale"]
        assert scale is not None and scale > 0
        # A clean panel sits within a small multiple of the null scale — the
        # number the score has to be read against, since it is never 0.
        assert result.value < 3 * scale
        for stats in result.metadata["per_offset"].values():
            if stats.get("mean") is not None:
                assert stats["se"] > 0
                assert stats["t"] is not None

    def test_no_pre_event_offset_reports_nan_not_zero(self):
        result = event_around_return(self._panel(0.0), offsets=[1, 6, 12])
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "no_pre_event_offset_with_enough_events"
