"""Pairwise cross-slice Wald contrast verb."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix.metrics import fama_macbeth, ic, slice_pairwise_test
from factrix.stats import BlockBootstrap, WaldNWCluster


def _ic_panel(
    *,
    n_dates: int,
    seed: int,
    means: dict[str, float],
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    frames = []
    for lbl, mu in means.items():
        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "ic": rng.normal(mu, 0.05, n_dates),
                    "universe": [lbl] * n_dates,
                }
            )
        )
    return pl.concat(frames)


def test_two_slice_returns_one_row() -> None:
    df = _ic_panel(n_dates=120, seed=1, means={"a": 0.02, "b": 0.02})
    out = slice_pairwise_test(ic, df, label="universe")
    assert out.height == 1
    assert out.columns == ["slice_a", "slice_b", "n_obs", "stat", "p_raw", "p_adj"]
    assert out["n_obs"][0] == 120


def test_three_slice_returns_three_rows() -> None:
    df = _ic_panel(n_dates=120, seed=2, means={"a": 0.02, "b": 0.02, "c": 0.02})
    out = slice_pairwise_test(ic, df, label="universe")
    assert out.height == 3
    pairs = set(zip(out["slice_a"].to_list(), out["slice_b"].to_list(), strict=False))
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


def test_holm_adjustment_dominates_raw() -> None:
    df = _ic_panel(n_dates=120, seed=3, means={"a": 0.02, "b": 0.02, "c": 0.02})
    out = slice_pairwise_test(ic, df, label="universe", multiple_testing="holm")
    p_raw = out["p_raw"].to_list()
    p_adj = out["p_adj"].to_list()
    for raw, adj in zip(p_raw, p_adj, strict=False):
        assert adj >= raw - 1e-12


def test_bonferroni_factor_matches_k() -> None:
    df = _ic_panel(n_dates=120, seed=4, means={"a": 0.02, "b": 0.02, "c": 0.02})
    out = slice_pairwise_test(ic, df, label="universe", multiple_testing="bonferroni")
    p_raw = np.array(out["p_raw"].to_list())
    p_adj = np.array(out["p_adj"].to_list())
    expected = np.minimum(p_raw * len(p_raw), 1.0)
    np.testing.assert_allclose(p_adj, expected)


def test_detects_signal_difference() -> None:
    df = _ic_panel(n_dates=240, seed=5, means={"hot": 0.10, "cold": -0.01})
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

    df = _ic_panel(n_dates=10, seed=7, means={"a": 0.0, "b": 0.0})
    with pytest.raises(TypeError, match="slice-test-eligible"):
        slice_pairwise_test(fake_metric, df, label="universe")


def test_rejects_unimplemented_estimator() -> None:
    df = _ic_panel(n_dates=60, seed=8, means={"a": 0.0, "b": 0.0})
    with pytest.raises(NotImplementedError, match="BlockBootstrap"):
        slice_pairwise_test(ic, df, label="universe", estimator=BlockBootstrap())


def test_accepts_default_waldnwcluster_explicit() -> None:
    df = _ic_panel(n_dates=60, seed=9, means={"a": 0.0, "b": 0.0})
    out = slice_pairwise_test(ic, df, label="universe", estimator=WaldNWCluster())
    assert out.height == 1


def test_raises_when_single_slice() -> None:
    df = _ic_panel(n_dates=60, seed=10, means={"only": 0.0})
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
    df = _ic_panel(n_dates=60, seed=12, means={"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError, match="not recognized"):
        slice_pairwise_test(
            ic,
            df,
            label="universe",
            multiple_testing="fdr",  # type: ignore[arg-type]
        )
