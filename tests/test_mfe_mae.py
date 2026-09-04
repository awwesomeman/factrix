"""Tests for factrix.metrics.mfe_mae and factrix.metrics.event_quality."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.event_quality import (
    event_skewness,
    profit_factor,
)
from factrix.metrics.mfe_mae import (
    compute_mfe_mae,
    mfe_mae,
)

from .conftest import with_estimation_window

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event_with_price(
    n_assets: int = 10,
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
def no_price_data() -> pl.DataFrame:
    df = _make_event_with_price()
    return df.drop("price")


# ---------------------------------------------------------------------------
# compute_mfe_mae
# ---------------------------------------------------------------------------


class TestComputeMfeMae:
    def test_returns_expected_columns(self, event_data):
        result = compute_mfe_mae(event_data, window=10)
        assert set(result.columns) >= {
            "date",
            "asset_id",
            "mfe",
            "mae",
            "bars_to_mfe",
            "bars_to_mae",
        }
        assert len(result) > 0

    def test_mfe_positive_mae_negative(self, event_data):
        result = compute_mfe_mae(event_data, window=10)
        assert result["mfe"].mean() > 0
        assert result["mae"].mean() < 0

    def test_z_score_columns_present_and_consistent(self, event_data):
        result = compute_mfe_mae(event_data, window=10, estimation_window=60)
        assert {"mfe_z", "mae_z", "est_sigma"}.issubset(result.columns)
        # Where est_sigma is finite-positive the z-scores must match
        # the raw excursion divided by σ · √window.
        finite = result.filter(
            pl.col("est_sigma").is_finite() & (pl.col("est_sigma") > 0.0),
        )
        if finite.is_empty():
            pytest.skip("No events had enough look-back for σ estimation")
        scale = finite["est_sigma"] * math.sqrt(10)
        assert finite["mfe_z"].to_numpy() == pytest.approx(
            (finite["mfe"] / scale).to_numpy(),
        )
        assert finite["mae_z"].to_numpy() == pytest.approx(
            (finite["mae"] / scale).to_numpy(),
        )

    def test_no_price_returns_censored_events(self, no_price_data):
        result = compute_mfe_mae(no_price_data, window=10)
        assert result.height == no_price_data.filter(pl.col("factor") != 0).height
        assert result["path_status"].unique().to_list() == ["censored"]
        assert result["censor_reason"].unique().to_list() == ["missing_price_column"]

    def test_no_events_returns_empty(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [0.0],
                "price": [100.0],
            }
        )
        result = compute_mfe_mae(df)
        assert result.is_empty()

    def test_output_date_dtype_mirrors_input_us(self, event_data):
        df_us = event_data.with_columns(pl.col("date").cast(pl.Datetime("us")))
        result = compute_mfe_mae(df_us, window=10)
        assert result.schema["date"] == pl.Datetime("us"), (
            "us-precision input should survive to the output"
        )

    def test_output_date_dtype_mirrors_tz_aware(self, event_data):
        df_utc = event_data.with_columns(pl.col("date").dt.replace_time_zone("UTC"))
        result = compute_mfe_mae(df_utc, window=10)
        assert result.schema["date"] == pl.Datetime("ms", time_zone="UTC")

    def test_empty_output_also_mirrors_dtype(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("us")),
                "asset_id": ["A"],
                "factor": [0.0],
                "price": [100.0],
            }
        )
        result = compute_mfe_mae(df)
        assert result.is_empty()
        assert result.schema["date"] == pl.Datetime("us")

    def test_min_estimation_periods_lower_admits_more_z_scores(
        self,
        event_data,
    ):
        """Lowering the threshold (e.g. for weekly data) lets early events
        in the panel get a finite est_sigma where the BMP-default 20
        would NaN them out. Verifies the parameter actually plumbs into
        the σ̂ guard."""
        strict = compute_mfe_mae(
            event_data,
            window=10,
            estimation_window=30,
            min_estimation_periods=20,
        )
        loose = compute_mfe_mae(
            event_data,
            window=10,
            estimation_window=30,
            min_estimation_periods=5,
        )
        strict_finite = strict["est_sigma"].is_finite().sum()
        loose_finite = loose["est_sigma"].is_finite().sum()
        assert loose_finite >= strict_finite
        assert loose_finite > strict_finite, (
            "Fixture should have at least one event whose look-back "
            "fits the loose threshold but not the strict one."
        )

    def test_min_estimation_periods_below_two_raises(self, event_data):
        with pytest.raises(ValueError, match="min_estimation_periods"):
            compute_mfe_mae(event_data, window=10, min_estimation_periods=1)


# ---------------------------------------------------------------------------
# mfe_mae
# ---------------------------------------------------------------------------


class TestMfeMae:
    def test_returns_metric_output(self, event_data):
        mfe_df = compute_mfe_mae(event_data, window=10)
        result = mfe_mae(mfe_df)
        assert result is not None
        assert "mfe_p50" in result.metadata
        # MAE is a signed non-positive excursion: the WORST adverse quartile
        # is p25, not p75.
        assert "mae_p25" in result.metadata
        assert "mae_p75" not in result.metadata

    def test_short_circuit_when_empty(self):
        empty = pl.DataFrame(
            schema={
                "date": pl.Datetime("ms"),
                "asset_id": pl.String,
                "mfe": pl.Float64,
                "mae": pl.Float64,
                "bars_to_mfe": pl.Int32,
                "bars_to_mae": pl.Int32,
            },
        )
        result = mfe_mae(empty)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "no_price_data"
        assert result.metadata["n_events"] == 0


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def _event_outcomes(self, returns: list[float]) -> pl.DataFrame:
        """Events on consecutive days behind a zero-return estimation window.

        ``profit_factor`` sums *abnormal* returns, so the events need history;
        the zero warm-up leaves the abnormal return equal to the raw return
        each case below is written against.
        """
        n = len(returns)
        return with_estimation_window(
            pl.DataFrame(
                {
                    "date": [
                        datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)
                    ],
                    "asset_id": ["A"] * n,
                    "factor": [1.0] * n,
                    "forward_return": returns,
                }
            )
        )

    def test_strong_signal_above_one(self, event_data):
        result = profit_factor(event_data)
        assert result.value > 0

    def test_metadata_has_gains_losses(self, event_data):
        result = profit_factor(event_data)
        assert "total_gains" in result.metadata
        assert "total_losses" in result.metadata

    def test_no_losses_returns_unbounded_ratio(self):
        result = profit_factor(self._event_outcomes([0.01, 0.02, 0.03, 0.04]))
        assert math.isinf(result.value)
        assert result.metadata["no_losses"] is True
        assert result.metadata["no_gains"] is False
        assert result.metadata["profit_factor_status"] == "unbounded_no_losses"

    def test_no_gains_with_losses_returns_zero(self):
        result = profit_factor(self._event_outcomes([-0.01, -0.02, -0.03, -0.04]))
        assert result.value == 0.0
        assert result.metadata["no_gains"] is True
        assert result.metadata["no_losses"] is False
        assert result.metadata["profit_factor_status"] == "finite"

    def test_no_gains_or_losses_returns_nan(self):
        result = profit_factor(self._event_outcomes([0.0, 0.0, 0.0, 0.0]))
        assert math.isnan(result.value)
        assert result.metadata["no_gains"] is True
        assert result.metadata["no_losses"] is True
        assert result.metadata["profit_factor_status"] == "undefined_no_gains_or_losses"

    def test_mixed_events_return_finite_ratio(self):
        result = profit_factor(self._event_outcomes([0.03, -0.01, 0.02, -0.04]))
        assert result.value == pytest.approx(1.0)
        assert result.metadata["profit_factor_status"] == "finite"

    def test_insufficient_events(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [1.0],
                "forward_return": [0.01],
            }
        )
        result = profit_factor(df)
        assert math.isnan(result.value)


# ---------------------------------------------------------------------------
# event_skewness
# ---------------------------------------------------------------------------


class TestEventSkewness:
    def test_returns_metric(self, event_data):
        result = event_skewness(event_data)
        assert isinstance(result.value, float)

    def test_insufficient_events(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [1.0],
                "forward_return": [0.01],
            }
        )
        result = event_skewness(df)
        assert math.isnan(result.value)


# ---------------------------------------------------------------------------
# Excursion sign / quantile conventions (regression)
# ---------------------------------------------------------------------------


def _price_path(prices: list[float], direction: float = 1.0) -> pl.DataFrame:
    """Single-asset panel whose only event is on the first bar."""
    n = len(prices)
    return pl.DataFrame(
        {
            "date": pl.Series(
                [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                dtype=pl.Datetime("ms"),
            ),
            "asset_id": ["A"] * n,
            "factor": [direction] + [0.0] * (n - 1),
            "forward_return": [0.0] * n,
            "price": prices,
        }
    )


class TestSweeneyFloor:
    """MFE >= 0 and MAE <= 0 by construction (Sweeney / Tharp)."""

    def test_monotonically_losing_trade_has_zero_mfe(self):
        # Every post-entry bar is below entry, so the trade was never
        # favorable. Old behaviour reported the least-bad bar as a
        # *negative* MFE.
        out = compute_mfe_mae(_price_path([100.0, 99.0, 98.0, 97.0]), window=3)
        assert out["mfe"][0] == 0.0
        assert out["bars_to_mfe"][0] == 0
        assert out["mae"][0] == pytest.approx(-0.03)
        assert out["bars_to_mae"][0] == 3

    def test_monotonically_winning_trade_has_zero_mae(self):
        # Old behaviour reported the least-good bar as a *positive* MAE.
        out = compute_mfe_mae(_price_path([100.0, 101.0, 102.0, 103.0]), window=3)
        assert out["mae"][0] == 0.0
        assert out["bars_to_mae"][0] == 0
        assert out["mfe"][0] == pytest.approx(0.03)
        assert out["bars_to_mfe"][0] == 3

    def test_short_direction_is_floored_the_same_way(self):
        # direction = -1: rising prices are adverse for the position.
        out = compute_mfe_mae(
            _price_path([100.0, 101.0, 102.0, 103.0], direction=-1.0), window=3
        )
        assert out["mfe"][0] == 0.0
        assert out["mae"][0] == pytest.approx(-0.03)

    def test_two_sided_path_keeps_both_excursions(self):
        out = compute_mfe_mae(_price_path([100.0, 102.0, 98.0, 100.0]), window=3)
        assert out["mfe"][0] == pytest.approx(0.02)
        assert out["bars_to_mfe"][0] == 1
        assert out["mae"][0] == pytest.approx(-0.02)
        assert out["bars_to_mae"][0] == 2

    def test_signs_hold_across_a_realistic_panel(self, event_data):
        out = compute_mfe_mae(event_data, window=10)
        assert (out["mfe"] >= 0.0).all()
        assert (out["mae"] <= 0.0).all()

    def test_z_siblings_derive_from_the_floored_values(self, event_data):
        out = compute_mfe_mae(event_data, window=10).drop_nulls().drop_nans()
        assert (out["mfe_z"] >= 0.0).all()
        assert (out["mae_z"] <= 0.0).all()
        # mfe_z / mfe == mae_z / mae == 1 / window_scale wherever both are
        # non-zero, i.e. the z pair is the same rescaling of the floored pair.
        both = out.filter((pl.col("mfe") > 0) & (pl.col("mae") < 0))
        assert both.height > 0
        ratio_f = (both["mfe_z"] / both["mfe"]).to_numpy()
        ratio_a = (both["mae_z"] / both["mae"]).to_numpy()
        assert np.allclose(ratio_f, ratio_a)


class TestWorstAdverseQuartile:
    def test_headline_divides_by_the_worst_quartile_not_the_mildest(self):
        # mae values: -0.10 .. -0.01. quantile(0.25) is the WORST quartile
        # (most negative); quantile(0.75) is the mildest.
        maes = [-0.01 * (i + 1) for i in range(10)]
        mfes = [0.02] * 10
        per_event = pl.DataFrame(
            {
                "date": pl.Series(
                    [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
                    dtype=pl.Datetime("ms"),
                ),
                "asset_id": ["A"] * 10,
                "mfe": mfes,
                "mae": maes,
            }
        )
        result = mfe_mae(per_event)

        expected_p25 = float(pl.Series(maes).quantile(0.25))
        assert result.metadata["mae_p25"] == pytest.approx(expected_p25)
        assert expected_p25 < float(pl.Series(maes).quantile(0.75))
        assert result.value == pytest.approx(0.02 / abs(expected_p25))
        # The old (mildest-quartile) denominator would have inflated the ratio.
        assert result.value < 0.02 / abs(float(pl.Series(maes).quantile(0.75)))

    def test_z_sibling_uses_p25_too(self, event_data):
        per_event = compute_mfe_mae(event_data, window=10)
        result = mfe_mae(per_event)
        if "mae_z_p25" in result.metadata:
            mae_z = per_event["mae_z"].drop_nulls().drop_nans()
            assert result.metadata["mae_z_p25"] == pytest.approx(
                float(mae_z.quantile(0.25))
            )
            assert "mae_z_p75" not in result.metadata


class TestExcursionRatioEdges:
    """The best possible outcome must not share a score with the worst."""

    @staticmethod
    def _per_event(mfe: list[float], mae: list[float]) -> pl.DataFrame:
        return pl.DataFrame({"mfe": mfe, "mae": mae})

    def test_no_adverse_excursion_is_unbounded_not_zero(self):
        n = 40
        rng = np.random.default_rng(0)
        result = mfe_mae(self._per_event(list(rng.uniform(0.05, 0.09, n)), [0.0] * n))
        assert math.isinf(result.value)
        assert result.metadata["mfe_mae_ratio_status"] == (
            "unbounded_no_adverse_excursion"
        )

    def test_no_excursion_at_all_is_undefined(self):
        n = 40
        result = mfe_mae(self._per_event([0.0] * n, [0.0] * n))
        assert math.isnan(result.value)
        assert result.metadata["mfe_mae_ratio_status"] == "undefined_no_excursion"

    def test_ordinary_sample_is_finite(self):
        n = 40
        rng = np.random.default_rng(1)
        result = mfe_mae(
            self._per_event(
                list(rng.uniform(0.02, 0.06, n)), list(-rng.uniform(0.01, 0.03, n))
            )
        )
        assert math.isfinite(result.value)
        assert result.metadata["mfe_mae_ratio_status"] == "finite"
