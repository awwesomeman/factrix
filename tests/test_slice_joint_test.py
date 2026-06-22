"""Omnibus joint Wald χ² cross-slice verb (data-first)."""

from __future__ import annotations

import polars as pl
import pytest
from factrix import slice_joint_test
from factrix._errors import UserInputError
from factrix.metrics import ic, monotonicity

from tests._slice_panel import build_labelled_raw_panel

_JOINT_COLS = ["n_obs", "k_slices", "df", "stat", "p_value"]


def test_two_slice_returns_single_row() -> None:
    df = build_labelled_raw_panel(
        n_dates=120, seed=1, signal={"a": 0.1, "b": 0.1}, label_col="sector"
    )
    out = slice_joint_test(df, ic(), by="sector", factor_col="factor")
    assert out.height == 1
    assert out.columns == _JOINT_COLS
    assert out["n_obs"][0] == 120
    assert out["k_slices"][0] == 2
    assert out["df"][0] == 1


def test_three_slice_df_equals_k_minus_one() -> None:
    df = build_labelled_raw_panel(
        n_dates=120,
        seed=2,
        signal={"a": 0.1, "b": 0.1, "c": 0.1},
        label_col="sector",
    )
    out = slice_joint_test(df, ic(), by="sector", factor_col="factor")
    assert out["k_slices"][0] == 3
    assert out["df"][0] == 2


def test_detects_omnibus_signal() -> None:
    df = build_labelled_raw_panel(
        n_dates=240,
        seed=3,
        signal={"hot": 0.4, "cold": -0.1, "neutral": 0.0},
        label_col="sector",
    )
    out = slice_joint_test(df, ic(), by="sector", factor_col="factor")
    assert out["p_value"][0] < 0.01


def test_null_means_no_omnibus_rejection() -> None:
    df = build_labelled_raw_panel(
        n_dates=240,
        seed=4,
        signal={"a": 0.1, "b": 0.1, "c": 0.1},
        label_col="sector",
    )
    out = slice_joint_test(df, ic(), by="sector", factor_col="factor")
    assert out["p_value"][0] > 0.10


def test_rejects_bare_class() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=5, signal={"a": 0.0, "b": 0.0}, label_col="sector"
    )
    with pytest.raises(UserInputError, match="instance"):
        slice_joint_test(df, ic, by="sector", factor_col="factor")  # type: ignore[arg-type]


def test_rejects_non_eligible_metric() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=6, signal={"a": 0.0, "b": 0.0}, label_col="sector"
    )
    with pytest.raises(TypeError, match="slice-test-eligible"):
        slice_joint_test(df, monotonicity(), by="sector", factor_col="factor")


def test_rejects_missing_factor_col() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=7, signal={"a": 0.0, "b": 0.0}, label_col="sector"
    )
    with pytest.raises(UserInputError, match="factor_col"):
        slice_joint_test(df, ic(), by="sector", factor_col="absent")


def test_raises_when_single_slice() -> None:
    df = build_labelled_raw_panel(
        n_dates=60, seed=8, signal={"only": 0.0}, label_col="sector"
    )
    with pytest.raises(ValueError, match="≥2 slice values"):
        slice_joint_test(df, ic(), by="sector", factor_col="factor")


def test_raises_when_dates_dont_align() -> None:
    df_a = build_labelled_raw_panel(
        n_dates=30, seed=9, signal={"a": 0.1}, label_col="regime"
    )
    df_b = build_labelled_raw_panel(
        n_dates=30, seed=10, signal={"b": 0.1}, label_col="regime"
    ).with_columns(pl.col("date") + pl.duration(days=100))
    with pytest.raises(ValueError, match="aligned dates"):
        slice_joint_test(
            pl.concat([df_a, df_b]), ic(), by="regime", factor_col="factor"
        )
