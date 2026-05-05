"""Two-tier sample-size guards for inference primitives (issue #48 D).

Each primitive (``fama_macbeth``, ``caar``, ``top_concentration``)
must implement three tiers:

1. ``n < HARD`` → short-circuit (NaN ``MetricOutput``).
2. ``HARD ≤ n < WARN`` → return stat AND emit ``UserWarning`` AND
   surface the relevant ``WarningCode.value`` in ``metadata["warning_codes"]``.
3. ``n ≥ WARN`` → silent: stat returned, no warning, no warning_codes.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._types import (
    MIN_EVENTS_HARD,
    MIN_EVENTS_WARN,
    MIN_PORTFOLIO_PERIODS_HARD,
    MIN_PORTFOLIO_PERIODS_WARN,
)
from factrix.metrics.caar import caar
from factrix.metrics.concentration import top_concentration
from factrix.metrics.fama_macbeth import (
    MIN_FM_PERIODS_HARD,
    MIN_FM_PERIODS_WARN,
    fama_macbeth,
)


def _beta_df(n: int, *, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    betas = rng.normal(0.0, 0.01, n)
    return pl.DataFrame({"date": dates, "beta": betas}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


def _caar_df(n: int, *, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    vals = rng.normal(0.0, 0.01, n)
    return pl.DataFrame({"date": dates, "caar": vals}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


def _concentration_panel(
    n_dates: int, n_assets: int = 30, *, seed: int = 0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = datetime(2024, 1, 1) + timedelta(days=d)
        factors = rng.normal(0.0, 1.0, n_assets)
        rets = rng.normal(0.0, 0.02, n_assets)
        for a in range(n_assets):
            rows.append(
                {
                    "date": date,
                    "asset_id": f"A{a}",
                    "factor": float(factors[a]),
                    "forward_return": float(rets[a]),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


# ---------------------------------------------------------------------------
# fama_macbeth
# ---------------------------------------------------------------------------


class TestFamaMacbethTwoTier:
    def test_below_hard_short_circuits(self) -> None:
        df = _beta_df(MIN_FM_PERIODS_HARD - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = fama_macbeth(df)
        assert math.isnan(out.value)
        assert out.stat is None
        assert out.metadata["reason"] == "insufficient_fm_periods"

    def test_borderline_warns_and_tags_metadata(self) -> None:
        n = (MIN_FM_PERIODS_HARD + MIN_FM_PERIODS_WARN) // 2
        df = _beta_df(n)
        with pytest.warns(UserWarning, match="MIN_FM_PERIODS_WARN"):
            out = fama_macbeth(df)
        assert out.stat is not None
        assert (
            WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value
            in out.metadata["warning_codes"]
        )

    def test_at_or_above_warn_is_silent(self) -> None:
        df = _beta_df(MIN_FM_PERIODS_WARN + 5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = fama_macbeth(df)
        assert out.stat is not None
        assert "warning_codes" not in out.metadata


# ---------------------------------------------------------------------------
# caar
# ---------------------------------------------------------------------------


class TestCaarTwoTier:
    def test_below_hard_short_circuits(self) -> None:
        # caar uses _scaled_min_periods(MIN_EVENTS_HARD, forward_periods);
        # forward_periods=1 keeps the scaled floor equal to the raw floor.
        df = _caar_df(MIN_EVENTS_HARD - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = caar(df, forward_periods=1)
        assert math.isnan(out.value)
        assert out.stat is None
        assert out.metadata["reason"] == "insufficient_event_dates"

    def test_borderline_warns_and_tags_metadata(self) -> None:
        n = (MIN_EVENTS_HARD + MIN_EVENTS_WARN) // 2
        df = _caar_df(n)
        with pytest.warns(UserWarning, match="MIN_EVENTS_WARN"):
            out = caar(df, forward_periods=1)
        assert out.stat is not None
        assert (
            WarningCode.FEW_EVENTS_BROWN_WARNER.value in out.metadata["warning_codes"]
        )

    def test_at_or_above_warn_is_silent(self) -> None:
        df = _caar_df(MIN_EVENTS_WARN + 5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = caar(df, forward_periods=1)
        assert out.stat is not None
        assert "warning_codes" not in out.metadata


# ---------------------------------------------------------------------------
# top_concentration
# ---------------------------------------------------------------------------


class TestTopConcentrationTwoTier:
    def test_below_hard_short_circuits(self) -> None:
        # Only 2 dates ⇒ HHI series length 2 < MIN_PORTFOLIO_PERIODS_HARD=3.
        df = _concentration_panel(MIN_PORTFOLIO_PERIODS_HARD - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = top_concentration(df, forward_periods=1, q_top=0.2)
        assert math.isnan(out.value)
        assert out.metadata["reason"] == "insufficient_portfolio_periods"

    def test_borderline_warns_and_tags_metadata(self) -> None:
        n = (MIN_PORTFOLIO_PERIODS_HARD + MIN_PORTFOLIO_PERIODS_WARN) // 2
        df = _concentration_panel(n)
        with pytest.warns(UserWarning, match="MIN_PORTFOLIO_PERIODS_WARN"):
            out = top_concentration(df, forward_periods=1, q_top=0.2)
        assert out.stat is not None
        assert (
            WarningCode.BORDERLINE_PORTFOLIO_PERIODS.value
            in out.metadata["warning_codes"]
        )

    def test_at_or_above_warn_is_silent(self) -> None:
        df = _concentration_panel(MIN_PORTFOLIO_PERIODS_WARN + 5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = top_concentration(df, forward_periods=1, q_top=0.2)
        assert out.stat is not None
        assert "warning_codes" not in out.metadata
