"""Omnibus joint Wald χ² cross-slice verb."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix import slice_joint_test
from factrix.metrics import ic
from factrix.stats import BlockBootstrap, WaldNWCluster

from tests._slice_panel import build_labelled_ic_panel


def test_two_slice_returns_single_row() -> None:
    df = build_labelled_ic_panel(
        n_dates=120, seed=1, means={"a": 0.02, "b": 0.02}, label_col="regime"
    )
    out = slice_joint_test(ic, df, label="regime")
    assert out.height == 1
    assert out.columns == ["n_obs", "k_slices", "df", "stat", "p"]
    assert out["n_obs"][0] == 120
    assert out["k_slices"][0] == 2
    assert out["df"][0] == 1


def test_three_slice_df_equals_k_minus_one() -> None:
    df = build_labelled_ic_panel(
        n_dates=120, seed=2, means={"a": 0.0, "b": 0.0, "c": 0.0}, label_col="regime"
    )
    out = slice_joint_test(ic, df, label="regime")
    assert out["k_slices"][0] == 3
    assert out["df"][0] == 2


def test_detects_omnibus_signal() -> None:
    df = build_labelled_ic_panel(
        n_dates=240,
        seed=3,
        means={"hot": 0.10, "cold": -0.05, "neutral": 0.0},
        label_col="regime",
    )
    out = slice_joint_test(ic, df, label="regime")
    assert out["p"][0] < 0.01


def test_null_means_no_omnibus_rejection() -> None:
    df = build_labelled_ic_panel(
        n_dates=240, seed=4, means={"a": 0.0, "b": 0.0, "c": 0.0}, label_col="regime"
    )
    out = slice_joint_test(ic, df, label="regime")
    assert out["p"][0] > 0.10


def test_rejects_non_eligible_metric() -> None:
    def fake_metric(df: pl.DataFrame) -> None:
        return None

    df = build_labelled_ic_panel(
        n_dates=10, seed=5, means={"a": 0.0, "b": 0.0}, label_col="regime"
    )
    with pytest.raises(TypeError, match="slice-test-eligible"):
        slice_joint_test(fake_metric, df, label="regime")


def test_rejects_block_bootstrap_estimator() -> None:
    df = build_labelled_ic_panel(
        n_dates=60, seed=6, means={"a": 0.0, "b": 0.0}, label_col="regime"
    )
    with pytest.raises(NotImplementedError, match="BlockBootstrap"):
        slice_joint_test(ic, df, label="regime", estimator=BlockBootstrap())


def test_accepts_default_waldnwcluster_explicit() -> None:
    df = build_labelled_ic_panel(
        n_dates=60, seed=7, means={"a": 0.0, "b": 0.0}, label_col="regime"
    )
    out = slice_joint_test(ic, df, label="regime", estimator=WaldNWCluster())
    assert out.height == 1


def test_raises_when_single_slice() -> None:
    df = build_labelled_ic_panel(
        n_dates=60, seed=8, means={"only": 0.0}, label_col="regime"
    )
    with pytest.raises(ValueError, match="≥2 slice values"):
        slice_joint_test(ic, df, label="regime")


def test_raises_when_dates_dont_align() -> None:
    rng = np.random.default_rng(9)
    base = dt.date(2024, 1, 1)
    df_a = pl.DataFrame(
        {
            "date": [base + dt.timedelta(days=i) for i in range(30)],
            "ic": rng.normal(0, 0.05, 30),
            "regime": ["a"] * 30,
        }
    )
    df_b = pl.DataFrame(
        {
            "date": [base + dt.timedelta(days=100 + i) for i in range(30)],
            "ic": rng.normal(0, 0.05, 30),
            "regime": ["b"] * 30,
        }
    )
    with pytest.raises(ValueError, match="aligned dates"):
        slice_joint_test(ic, pl.concat([df_a, df_b]), label="regime")
