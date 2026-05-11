"""Pairwise cross-slice Wald contrast verb."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix import slice_pairwise_test
from factrix.metrics import fama_macbeth, ic
from factrix.stats import BlockBootstrap, WaldNWCluster

from tests._slice_panel import build_labelled_ic_panel


def test_two_slice_returns_one_row() -> None:
    df = build_labelled_ic_panel(
        n_dates=120, seed=1, means={"a": 0.02, "b": 0.02}, label_col="universe"
    )
    out = slice_pairwise_test(ic, df, label="universe")
    assert out.height == 1
    assert out.columns == ["slice_a", "slice_b", "n_obs", "stat", "p_raw", "p_adj"]
    assert out["n_obs"][0] == 120


def test_three_slice_returns_three_rows() -> None:
    df = build_labelled_ic_panel(
        n_dates=120,
        seed=2,
        means={"a": 0.02, "b": 0.02, "c": 0.02},
        label_col="universe",
    )
    out = slice_pairwise_test(ic, df, label="universe")
    assert out.height == 3
    pairs = set(zip(out["slice_a"].to_list(), out["slice_b"].to_list(), strict=False))
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


def test_holm_adjustment_dominates_raw() -> None:
    df = build_labelled_ic_panel(
        n_dates=120,
        seed=3,
        means={"a": 0.02, "b": 0.02, "c": 0.02},
        label_col="universe",
    )
    out = slice_pairwise_test(ic, df, label="universe", multiple_testing="holm")
    p_raw = out["p_raw"].to_list()
    p_adj = out["p_adj"].to_list()
    for raw, adj in zip(p_raw, p_adj, strict=False):
        assert adj >= raw - 1e-12


def test_bonferroni_factor_matches_k() -> None:
    df = build_labelled_ic_panel(
        n_dates=120,
        seed=4,
        means={"a": 0.02, "b": 0.02, "c": 0.02},
        label_col="universe",
    )
    out = slice_pairwise_test(ic, df, label="universe", multiple_testing="bonferroni")
    p_raw = np.array(out["p_raw"].to_list())
    p_adj = np.array(out["p_adj"].to_list())
    expected = np.minimum(p_raw * len(p_raw), 1.0)
    np.testing.assert_allclose(p_adj, expected)


def test_detects_signal_difference() -> None:
    df = build_labelled_ic_panel(
        n_dates=240, seed=5, means={"hot": 0.10, "cold": -0.01}, label_col="universe"
    )
    out = slice_pairwise_test(ic, df, label="universe")
    assert out["p_raw"][0] < 0.01


def test_fama_macbeth_metric_accepted() -> None:
    rng = np.random.default_rng(6)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    frames = []
    for lbl, mu in {"x": 0.001, "y": 0.001}.items():
        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "beta": rng.normal(mu, 0.01, len(dates)),
                    "regime": [lbl] * len(dates),
                }
            )
        )
    df = pl.concat(frames)
    out = slice_pairwise_test(fama_macbeth, df, label="regime")
    assert out.height == 1
    assert out.columns == ["slice_a", "slice_b", "n_obs", "stat", "p_raw", "p_adj"]


def test_rejects_non_eligible_metric() -> None:
    def fake_metric(df: pl.DataFrame) -> None:
        return None

    df = build_labelled_ic_panel(
        n_dates=10, seed=7, means={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    with pytest.raises(TypeError, match="slice-test-eligible"):
        slice_pairwise_test(fake_metric, df, label="universe")


def test_block_bootstrap_path_returns_signed_mean_diff_stat() -> None:
    df = build_labelled_ic_panel(
        n_dates=120, seed=8, means={"a": 0.10, "b": -0.05}, label_col="universe"
    )
    out = slice_pairwise_test(
        ic,
        df,
        label="universe",
        estimator=BlockBootstrap(n_resamples=199, rng_seed=42),
    )
    assert out.height == 1
    assert out["stat"][0] == pytest.approx(0.10 - (-0.05), abs=0.05)
    assert out["p_raw"][0] < 0.05


def test_block_bootstrap_defaults_to_romano_wolf() -> None:
    df = build_labelled_ic_panel(
        n_dates=120,
        seed=80,
        means={"a": 0.10, "b": -0.05, "c": 0.0},
        label_col="universe",
    )
    rw_out = slice_pairwise_test(
        ic,
        df,
        label="universe",
        estimator=BlockBootstrap(n_resamples=199, rng_seed=7),
    )
    holm_out = slice_pairwise_test(
        ic,
        df,
        label="universe",
        estimator=BlockBootstrap(n_resamples=199, rng_seed=7),
        multiple_testing="holm",
    )
    np.testing.assert_array_equal(
        rw_out["p_raw"].to_numpy(), holm_out["p_raw"].to_numpy()
    )
    assert not np.allclose(rw_out["p_adj"].to_numpy(), holm_out["p_adj"].to_numpy())


def test_romano_wolf_requires_block_bootstrap() -> None:
    df = build_labelled_ic_panel(
        n_dates=60, seed=81, means={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    with pytest.raises(ValueError, match="romano_wolf"):
        slice_pairwise_test(ic, df, label="universe", multiple_testing="romano_wolf")


def test_rejects_unknown_estimator_type() -> None:
    class FakeEstimator:
        pass

    df = build_labelled_ic_panel(
        n_dates=60, seed=82, means={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    with pytest.raises(NotImplementedError, match="FakeEstimator"):
        slice_pairwise_test(
            ic,
            df,
            label="universe",
            estimator=FakeEstimator(),  # type: ignore[arg-type]
        )


def test_accepts_default_waldnwcluster_explicit() -> None:
    df = build_labelled_ic_panel(
        n_dates=60, seed=9, means={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    out = slice_pairwise_test(ic, df, label="universe", estimator=WaldNWCluster())
    assert out.height == 1


def test_raises_when_single_slice() -> None:
    df = build_labelled_ic_panel(
        n_dates=60, seed=10, means={"only": 0.0}, label_col="universe"
    )
    with pytest.raises(ValueError, match="≥2 slice values"):
        slice_pairwise_test(ic, df, label="universe")


def test_raises_when_dates_dont_align() -> None:
    rng = np.random.default_rng(11)
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
        slice_pairwise_test(ic, pl.concat([df_a, df_b]), label="regime")


def test_rejects_unknown_multiple_testing() -> None:
    df = build_labelled_ic_panel(
        n_dates=60, seed=12, means={"a": 0.0, "b": 0.0}, label_col="universe"
    )
    with pytest.raises(ValueError, match="not recognized"):
        slice_pairwise_test(
            ic,
            df,
            label="universe",
            multiple_testing="fdr",  # type: ignore[arg-type]
        )
