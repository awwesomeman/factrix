"""The structural contract every public entry point shares.

`_normalize_panel` enforces three things that were previously left to each
producer: a temporal `date`, unique `(date, asset_id)` keys, and non-finite
numerics blanked to null *before* any arithmetic. Each of the cases below is a
number the library used to return without error or warning.
"""

from datetime import datetime, timedelta

import factrix as fx
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._data_input import _normalize_panel
from factrix._errors import UserInputError
from factrix.preprocess import compute_forward_return


def _panel(prices, dates=None, asset="A"):
    dates = dates or [
        datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(prices))
    ]
    return pl.DataFrame(
        {
            "date": dates,
            "asset_id": [asset] * len(prices),
            "price": prices,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestDateDtypeContract:
    def test_string_date_is_rejected(self):
        """Lexicographic ordering silently reorders any non-ISO format."""
        df = pl.DataFrame(
            {
                "date": ["01/02/2024", "02/01/2024", "03/01/2024"],
                "asset_id": ["A", "A", "A"],
                "price": [110.0, 120.0, 100.0],
            }
        )
        with pytest.raises(UserInputError, match="date"):
            _normalize_panel(df)
        with pytest.raises(UserInputError, match="date"):
            compute_forward_return(df, forward_periods=1)

    def test_integer_date_is_rejected(self):
        df = pl.DataFrame(
            {"date": [0, 1, 2], "asset_id": ["A"] * 3, "price": [1.0, 2.0, 3.0]}
        )
        with pytest.raises(UserInputError, match="date"):
            _normalize_panel(df)

    @pytest.mark.parametrize(
        "dtype",
        [pl.Date, pl.Datetime("ms"), pl.Datetime("ns"), pl.Datetime("us", "UTC")],
    )
    def test_temporal_dtypes_pass(self, dtype):
        df = _panel([1.0, 2.0, 3.0]).with_columns(pl.col("date").cast(dtype))
        assert _normalize_panel(df).height == 3


class TestKeyUniqueness:
    def test_duplicate_keys_are_rejected(self):
        """A duplicate makes the 'next period' the same date's twin."""
        clean = _panel([100.0, 101.0, 102.0, 103.0])
        dup = pl.concat([clean, clean])
        with pytest.raises(UserInputError, match=r"\(date, asset_id\)"):
            _normalize_panel(dup)
        with pytest.raises(UserInputError, match=r"\(date, asset_id\)"):
            compute_forward_return(dup, forward_periods=1)

    def test_duplicates_used_to_fabricate_zero_returns(self):
        """Half the surviving panel was 0.0 returns, silently."""
        clean = _panel([100.0, 101.0, 102.0, 103.0])
        out = compute_forward_return(clean, forward_periods=1)
        assert 0.0 not in out["forward_return"].to_list()

    def test_unique_keys_across_assets_are_fine(self):
        a = _panel([100.0, 101.0, 102.0], asset="A")
        b = _panel([100.0, 101.0, 102.0], asset="B")
        assert _normalize_panel(pl.concat([a, b])).height == 6


class TestNonFiniteNumerics:
    def test_inf_price_no_longer_fabricates_a_minus_one_return(self):
        """`finite / inf` is 0.0, so the quotient was a finite -100% return."""
        with_inf = _panel([100.0, 101.0, float("inf"), 103.0, 104.0, 105.0])
        null_twin = _panel([100.0, 101.0, None, 103.0, 104.0, 105.0])
        out = compute_forward_return(with_inf, forward_periods=1)
        twin = compute_forward_return(null_twin, forward_periods=1)
        assert -1.0 not in out["forward_return"].to_list()
        assert out["forward_return"].to_list() == pytest.approx(
            twin["forward_return"].to_list()
        )

    def test_nan_and_inf_become_null(self):
        df = _panel([1.0, float("nan"), float("-inf"), 4.0])
        out = _normalize_panel(df)
        assert out["price"].to_list() == [1.0, None, None, 4.0]

    def test_non_numeric_columns_are_untouched(self):
        df = _panel([1.0, 2.0, 3.0]).with_columns(
            pl.lit("tech").alias("sector"),
        )
        out = _normalize_panel(df)
        assert out["sector"].to_list() == ["tech"] * 3


class TestHorizonIsMeasuredOnThePeriodGrid:
    @staticmethod
    def _ragged_panel():
        """Asset A is missing periods 10-29; asset B is complete."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
        rows = []
        for i, d in enumerate(dates):
            for asset in ("A", "B"):
                if asset == "A" and 10 <= i < 30:
                    continue
                rows.append({"date": d, "asset_id": asset, "price": 100.0 * (1.01**i)})
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_gap_no_longer_stretches_the_window(self):
        """A "5-period" return used to span 25 real periods across the gap."""
        panel = self._ragged_panel()
        with pytest.warns(UserWarning, match="ragged_period_grid"):
            out = compute_forward_return(panel, forward_periods=5)
        # Every surviving return is the same 5-period compounded move, whichever
        # asset it came from — the row-shift version gave asset A a 25-period
        # return at the gap boundary.
        returns = out["forward_return"].to_list()
        expected = (1.01**5 - 1) / 5
        assert returns == pytest.approx([expected] * len(returns))

    def test_asset_with_a_gap_contributes_no_row_across_it(self):
        panel = self._ragged_panel()
        with pytest.warns(UserWarning, match="ragged_period_grid"):
            out = compute_forward_return(panel, forward_periods=5)
        a_dates = out.filter(pl.col("asset_id") == "A")["date"].to_list()
        # Period index 4 would exit at index 10, which asset A does not have.
        assert datetime(2024, 1, 5) not in a_dates

    def test_complete_panel_does_not_warn(self):
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        rows = [
            {"date": d, "asset_id": a, "price": 100.0 + i}
            for i, d in enumerate(dates)
            for a in ("A", "B")
        ]
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        with warnings_as_errors():
            compute_forward_return(panel, forward_periods=5)

    def test_ragged_grid_code_is_documented(self):
        assert WarningCode.RAGGED_PERIOD_GRID.description


class TestGateReachesEveryEntryPoint:
    def test_evaluate_rejects_duplicate_keys(self):
        raw = fx.datasets.make_cs_panel(n_assets=10, n_dates=40, rng=0)
        panel = compute_forward_return(raw, forward_periods=1)
        dup = pl.concat([panel, panel.head(1)])
        with pytest.raises(UserInputError, match=r"\(date, asset_id\)"):
            fx.evaluate(
                dup,
                factor_cols=["factor"],
                metrics={"ic": fx.metrics.ic()},
                forward_periods=1,
            )

    def test_inspect_data_rejects_duplicate_keys(self):
        raw = fx.datasets.make_cs_panel(n_assets=10, n_dates=40, rng=0)
        panel = compute_forward_return(raw, forward_periods=1)
        dup = pl.concat([panel, panel.head(1)])
        with pytest.raises(UserInputError, match=r"\(date, asset_id\)"):
            fx.inspect_data(dup)


def warnings_as_errors():
    import warnings as _w
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            yield

    return _ctx()


class TestNonFloatColumnsUntouched:
    """Only float dtypes can carry NaN / ±inf; every other dtype passes through."""

    def test_decimal_column_passes_through(self):
        from decimal import Decimal

        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "asset_id": ["A", "A"],
                "market_cap": pl.Series(
                    [Decimal("1.50"), Decimal("2.25")], dtype=pl.Decimal(10, 2)
                ),
                "n": pl.Series([1, 2], dtype=pl.Int64),
                "flag": [True, False],
            }
        )
        out = _normalize_panel(df)
        assert out.schema["market_cap"] == pl.Decimal(10, 2)
        assert out["market_cap"].to_list() == df["market_cap"].to_list()
        assert out["n"].to_list() == [1, 2]
        assert out["flag"].to_list() == [True, False]
