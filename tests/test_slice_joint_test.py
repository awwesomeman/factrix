"""Omnibus joint Wald χ² cross-slice verb (data-first)."""

from __future__ import annotations

import polars as pl
import pytest
from factrix import slice_joint_test
from factrix._data_input import _stamp_horizons
from factrix._errors import UserInputError
from factrix.metrics import ic, monotonicity

from tests._slice_panel import (
    build_autocorrelated_ic_panel,
    build_labelled_raw_panel,
)

_JOINT_COLS = [
    "n_obs",
    "k_slices",
    "stat",
    "p_value",
    "stat_type",
    "reference_dist",
    "df_num",
    "df_denom",
    "multiplicity",
]


def test_two_slice_returns_single_row() -> None:
    df = build_labelled_raw_panel(
        n_dates=120, seed=1, signal={"a": 0.1, "b": 0.1}, label_col="sector"
    )
    out = slice_joint_test(
        df, ic(), by="sector", factor_col="factor", overlap_periods=1
    )
    assert out.height == 1
    assert out.columns == _JOINT_COLS
    assert out["n_obs"][0] == 120
    assert out["k_slices"][0] == 2
    assert out["df_num"][0] == 1
    assert out["stat_type"][0] == "wald"
    assert out["reference_dist"][0] == "F"
    assert out["df_denom"][0] == 119
    assert out["multiplicity"][0] is None


def test_three_slice_df_equals_k_minus_one() -> None:
    df = build_labelled_raw_panel(
        n_dates=120,
        seed=2,
        signal={"a": 0.1, "b": 0.1, "c": 0.1},
        label_col="sector",
    )
    out = slice_joint_test(
        df, ic(), by="sector", factor_col="factor", overlap_periods=1
    )
    assert out["k_slices"][0] == 3
    assert out["df_num"][0] == 2


def test_detects_omnibus_signal() -> None:
    df = build_labelled_raw_panel(
        n_dates=240,
        seed=3,
        signal={"hot": 0.4, "cold": -0.1, "neutral": 0.0},
        label_col="sector",
    )
    out = slice_joint_test(
        df, ic(), by="sector", factor_col="factor", overlap_periods=1
    )
    assert out["p_value"][0] < 0.01


def test_null_means_no_omnibus_rejection() -> None:
    df = build_labelled_raw_panel(
        n_dates=240,
        seed=4,
        signal={"a": 0.1, "b": 0.1, "c": 0.1},
        label_col="sector",
    )
    out = slice_joint_test(
        df, ic(), by="sector", factor_col="factor", overlap_periods=1
    )
    assert out["p_value"][0] > 0.10


def test_rejects_bare_class() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=5, signal={"a": 0.0, "b": 0.0}, label_col="sector"
    )
    with pytest.raises(UserInputError, match="instance"):
        slice_joint_test(df, ic, by="sector", factor_col="factor", overlap_periods=1)  # type: ignore[arg-type]


def test_rejects_non_eligible_metric() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=6, signal={"a": 0.0, "b": 0.0}, label_col="sector"
    )
    with pytest.raises(TypeError, match="slice-test-eligible"):
        slice_joint_test(
            df, monotonicity(), by="sector", factor_col="factor", overlap_periods=1
        )


def test_rejects_missing_factor_col() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=7, signal={"a": 0.0, "b": 0.0}, label_col="sector"
    )
    with pytest.raises(UserInputError, match="factor_col"):
        slice_joint_test(df, ic(), by="sector", factor_col="absent", overlap_periods=1)


def test_raises_when_single_slice() -> None:
    df = build_labelled_raw_panel(
        n_dates=60, seed=8, signal={"only": 0.0}, label_col="sector"
    )
    with pytest.raises(ValueError, match="≥2 distinct slice values"):
        slice_joint_test(df, ic(), by="sector", factor_col="factor", overlap_periods=1)


def test_raises_when_dates_dont_align() -> None:
    df_a = build_labelled_raw_panel(
        n_dates=30, seed=9, signal={"a": 0.1}, label_col="regime"
    )
    df_b = build_labelled_raw_panel(
        n_dates=30, seed=10, signal={"b": 0.1}, label_col="regime"
    ).with_columns(pl.col("date") + pl.duration(days=100))
    with pytest.raises(ValueError, match="aligned dates"):
        slice_joint_test(
            pl.concat([df_a, df_b]),
            ic(),
            by="regime",
            factor_col="factor",
            overlap_periods=1,
        )


class TestOverlapPeriodsResolution:
    """``overlap_periods`` resolves the HAC bandwidth the way the
    ``slice_period_*`` pair resolves it: the panel's stamp is the truth, a
    disagreeing declaration is rejected, and an unstamped panel must declare
    it rather than fall back to a silent default."""

    @staticmethod
    def _panel(overlap_periods: int | None) -> pl.DataFrame:
        df = build_autocorrelated_ic_panel(
            n_dates=120,
            seed=42,
            signal={"a": 0.1, "b": 0.0},
            label_col="sector",
            n_assets=120,
            phi=0.95,
            noise=0.1,
        )
        if overlap_periods is None:
            return df
        return _stamp_horizons(
            df, forward_periods=overlap_periods, overlap_periods=overlap_periods
        )

    def test_stamped_panel_reads_the_stamp(self) -> None:
        """Omitting the parameter and declaring the stamped value agree."""
        stamped = self._panel(12)
        omitted = slice_joint_test(stamped, ic(), by="sector", factor_col="factor")
        declared = slice_joint_test(
            stamped, ic(), by="sector", factor_col="factor", overlap_periods=12
        )
        assert omitted.equals(declared)

    def test_declared_overlap_disagreeing_with_the_stamp_raises(self) -> None:
        with pytest.raises(UserInputError, match="stamped evaluation-grid overlap"):
            slice_joint_test(
                self._panel(12),
                ic(),
                by="sector",
                factor_col="factor",
                overlap_periods=5,
            )

    def test_unstamped_panel_requires_a_declared_overlap(self) -> None:
        with pytest.raises(UserInputError, match="overlap_periods"):
            slice_joint_test(self._panel(None), ic(), by="sector", factor_col="factor")

    def test_declaration_on_an_unstamped_panel_drives_the_bandwidth(self) -> None:
        """A declared overlap widens the kernel exactly as the same stamp
        does — and on an autocorrelated series a wider kernel inflates the
        SE, so the Wald statistic shrinks."""
        unstamped = self._panel(None)
        short = slice_joint_test(
            unstamped, ic(), by="sector", factor_col="factor", overlap_periods=1
        )
        long = slice_joint_test(
            unstamped, ic(), by="sector", factor_col="factor", overlap_periods=12
        )
        assert long["stat"][0] < short["stat"][0]
        assert long["p_value"][0] > short["p_value"][0]
        stamped_long = slice_joint_test(
            self._panel(12), ic(), by="sector", factor_col="factor"
        )
        assert long.equals(stamped_long)

    def test_rejects_a_non_positive_overlap(self) -> None:
        with pytest.raises(UserInputError, match="overlap_periods"):
            slice_joint_test(
                self._panel(None),
                ic(),
                by="sector",
                factor_col="factor",
                overlap_periods=0,
            )
