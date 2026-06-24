"""Pairwise cross-slice Wald contrast verb (data-first)."""

from __future__ import annotations

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix import slice_pairwise_test
from factrix._data_input import _stamp_forward_periods
from factrix._errors import UserInputError
from factrix.metrics import caar, fm_beta, ic, monotonicity

from tests._slice_panel import (
    build_autocorrelated_ic_panel,
    build_labelled_raw_panel,
)

_PAIRWISE_COLS = ["slice_a", "slice_b", "n_obs", "mean_diff", "stat", "p_raw", "p_adj"]


def test_two_slice_returns_one_row() -> None:
    df = build_labelled_raw_panel(
        n_dates=120, seed=1, signal={"a": 0.1, "b": 0.1}, label_col="universe"
    )
    out = slice_pairwise_test(df, ic(), by="universe", factor_col="factor")
    assert out.height == 1
    assert out.columns == _PAIRWISE_COLS
    assert out["n_obs"][0] == 120


def test_three_slice_returns_three_rows() -> None:
    df = build_labelled_raw_panel(
        n_dates=120,
        seed=2,
        signal={"a": 0.1, "b": 0.1, "c": 0.1},
        label_col="universe",
    )
    out = slice_pairwise_test(df, ic(), by="universe", factor_col="factor")
    assert out.height == 3
    pairs = set(zip(out["slice_a"].to_list(), out["slice_b"].to_list(), strict=False))
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


def test_holm_adjustment_dominates_raw() -> None:
    df = build_labelled_raw_panel(
        n_dates=120,
        seed=3,
        signal={"a": 0.1, "b": 0.2, "c": -0.1},
        label_col="universe",
    )
    out = slice_pairwise_test(df, ic(), by="universe", factor_col="factor")
    for raw, adj in zip(out["p_raw"].to_list(), out["p_adj"].to_list(), strict=False):
        assert adj >= raw - 1e-12


def test_detects_signal_difference() -> None:
    df = build_labelled_raw_panel(
        n_dates=240, seed=5, signal={"hot": 0.4, "cold": -0.1}, label_col="universe"
    )
    out = slice_pairwise_test(df, ic(), by="universe", factor_col="factor")
    assert out["p_raw"][0] < 0.01


def test_mean_diff_sign_matches_direction() -> None:
    df = build_labelled_raw_panel(
        n_dates=240, seed=15, signal={"hot": 0.4, "cold": -0.1}, label_col="universe"
    )
    out = slice_pairwise_test(df, ic(), by="universe", factor_col="factor")
    row = out.row(0, named=True)
    # mean_diff = μ_a − μ_b; positive iff slice_a is the higher-IC universe.
    assert (row["mean_diff"] > 0) == (row["slice_a"] == "hot")


def test_fama_macbeth_metric_accepted() -> None:
    df = build_labelled_raw_panel(
        n_dates=60, seed=6, signal={"x": 0.1, "y": 0.1}, label_col="regime"
    )
    out = slice_pairwise_test(df, fm_beta(), by="regime", factor_col="factor")
    assert out.height == 1
    assert out.columns == _PAIRWISE_COLS


def test_caar_metric_accepted_for_event_slices() -> None:
    raw = fx.datasets.make_multi_factor_event_panel(
        n_factors=1,
        n_assets=40,
        n_dates=160,
        event_rate=0.20,
        post_event_drift_bps=30.0,
        seed=6,
    )
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
    assets = panel["asset_id"].unique().sort().to_list()
    universe = {a: ("u1" if i % 2 else "u0") for i, a in enumerate(assets)}
    panel = panel.with_columns(
        pl.col("asset_id").replace_strict(universe).alias("universe")
    )
    out = slice_pairwise_test(panel, caar(), by="universe", factor_col="factor_0000")
    assert out.height == 1
    assert out.columns == _PAIRWISE_COLS


def test_rejects_bare_class() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=7, signal={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    with pytest.raises(UserInputError, match="instance"):
        slice_pairwise_test(df, ic, by="universe", factor_col="factor")  # type: ignore[arg-type]


def test_rejects_non_metric() -> None:
    def fake_metric(df: pl.DataFrame) -> None:
        return None

    df = build_labelled_raw_panel(
        n_dates=20, seed=8, signal={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    with pytest.raises(UserInputError, match="metric instance"):
        slice_pairwise_test(df, fake_metric, by="universe", factor_col="factor")  # type: ignore[arg-type]


def test_rejects_non_eligible_metric() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=9, signal={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    with pytest.raises(TypeError, match="slice-test-eligible"):
        slice_pairwise_test(df, monotonicity(), by="universe", factor_col="factor")


def test_rejects_missing_factor_col() -> None:
    df = build_labelled_raw_panel(
        n_dates=20, seed=10, signal={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    with pytest.raises(UserInputError, match="factor_col"):
        slice_pairwise_test(df, ic(), by="universe", factor_col="absent")


def test_raises_when_single_slice() -> None:
    df = build_labelled_raw_panel(
        n_dates=60, seed=11, signal={"only": 0.0}, label_col="universe"
    )
    with pytest.raises(ValueError, match="≥2 slice values"):
        slice_pairwise_test(df, ic(), by="universe", factor_col="factor")


def test_raises_when_dates_dont_align() -> None:
    df_a = build_labelled_raw_panel(
        n_dates=30, seed=12, signal={"a": 0.1}, label_col="regime"
    )
    df_b = build_labelled_raw_panel(
        n_dates=30, seed=13, signal={"b": 0.1}, label_col="regime"
    ).with_columns(pl.col("date") + pl.duration(days=100))
    # Genuinely date-disjoint slices share no raw dates: the message points
    # at the slice_period_* path, not at a small-sample cause.
    with pytest.raises(ValueError, match="date-disjoint partition") as exc:
        slice_pairwise_test(
            pl.concat([df_a, df_b]), ic(), by="regime", factor_col="factor"
        )
    assert "slice_period_pairwise_test" in str(exc.value)


def test_aligned_slices_but_metric_dropped_reports_small_sample() -> None:
    """Date-aligned slices whose tiny cross-sections (N < MIN_IC_ASSETS) make
    every per-date IC drop must blame the thin universe, not call the
    partition date-disjoint."""
    df = build_labelled_raw_panel(
        n_dates=40,
        seed=14,
        signal={"a": 0.1, "b": 0.1},
        label_col="universe",
        n_assets=5,
    )
    with pytest.raises(ValueError, match="too few assets") as exc:
        slice_pairwise_test(df, ic(), by="universe", factor_col="factor")
    msg = str(exc.value)
    assert "date-aligned" in msg
    assert "date-disjoint" not in msg


def test_overlap_bandwidth_inflates_variance() -> None:
    """A longer HAC bandwidth (forward_periods overlap) widens the SE on an
    autocorrelated IC series → smaller Wald χ², larger p than the naive
    ``floor(T^(1/3))`` bandwidth. T=120 → floor=4; forward_periods=12 → 11.

    The overlap horizon is a property of the data (the stamp), not a metric
    knob, so the two regimes are two differently-stamped panels.
    """
    df = build_autocorrelated_ic_panel(
        n_dates=120,
        seed=42,
        signal={"a": 0.1, "b": 0.0},
        label_col="universe",
        n_assets=120,
        phi=0.95,
        noise=0.1,
    )
    short = slice_pairwise_test(
        _stamp_forward_periods(df, 1), ic(), by="universe", factor_col="factor"
    )
    long = slice_pairwise_test(
        _stamp_forward_periods(df, 12), ic(), by="universe", factor_col="factor"
    )
    assert long["stat"][0] < short["stat"][0]
    assert long["p_adj"][0] > short["p_adj"][0]
    # mean_diff is bandwidth-invariant — only the variance estimate changes.
    np.testing.assert_allclose(long["mean_diff"][0], short["mean_diff"][0])
