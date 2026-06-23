"""Date-disjoint omnibus cross-slice Wald verb (regime / period)."""

from __future__ import annotations

import pytest
from factrix import slice_period_joint_test
from factrix._errors import UserInputError
from factrix.metrics import ic, monotonicity

from tests._slice_panel import build_disjoint_period_panel

_JOINT_COLS = ["k_slices", "df", "stat", "p_value"]


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_single_row_shape(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=1,
        spans={"bull": (80, 0.1), "bear": (80, 0.1), "flat": (80, 0.1)},
        label_col="regime",
    )
    out = slice_period_joint_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng_seed=1
    )
    assert out.height == 1
    assert out.columns == _JOINT_COLS
    assert out["k_slices"][0] == 3
    assert out["df"][0] == 2


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_two_slice_df_is_one(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=2, spans={"a": (80, 0.1), "b": (80, 0.1)}, label_col="regime"
    )
    out = slice_period_joint_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng_seed=2
    )
    assert out["df"][0] == 1


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_detects_difference(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=3,
        spans={"hot": (200, 0.4), "cold": (200, -0.2)},
        label_col="regime",
    )
    out = slice_period_joint_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng_seed=3
    )
    assert out["p_value"][0] < 0.05


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_identical_signal_less_significant_than_split(method: str) -> None:
    """Same true edge across regimes → far weaker omnibus than a real split.

    Two independent samples with identical means still differ by sampling,
    so a single-seed ``p > α`` is not a reliable property; the robust,
    calibration-revealing check is *relative*: an identical-signal
    partition must yield a much larger p than a genuinely-different one.
    """
    same = build_disjoint_period_panel(
        seed=4, spans={"a": (200, 0.1), "b": (200, 0.1)}, label_col="regime"
    )
    split = build_disjoint_period_panel(
        seed=4, spans={"a": (200, 0.3), "b": (200, -0.2)}, label_col="regime"
    )
    p_same = slice_period_joint_test(
        same, ic(), by="regime", factor_col="factor", method=method, rng_seed=4
    )["p_value"][0]
    p_split = slice_period_joint_test(
        split, ic(), by="regime", factor_col="factor", method=method, rng_seed=4
    )["p_value"][0]
    assert p_same > p_split


def test_bootstrap_joint_is_bootstrap_native_for_two_slices() -> None:
    """For K=2 the omnibus *is* the single pairwise contrast, so the
    bootstrap joint p must track the bootstrap pairwise p (both empirical),
    not diverge to a χ² asymptotic p on the identical statistic."""
    from factrix import slice_period_pairwise_test

    df = build_disjoint_period_panel(
        seed=11, spans={"a": (60, 0.15), "b": (60, 0.0)}, label_col="regime"
    )
    pw = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", rng_seed=5
    )
    jt = slice_period_joint_test(df, ic(), by="regime", factor_col="factor", rng_seed=5)
    # Both empirical with the same B → identical 1/(B+1) granularity.
    assert jt["p_value"][0] == pytest.approx(pw["p_raw"][0])


def test_bootstrap_reproducible_under_seed() -> None:
    df = build_disjoint_period_panel(
        seed=5,
        spans={"a": (80, 0.1), "b": (80, -0.05), "c": (80, 0.0)},
        label_col="regime",
    )
    a = slice_period_joint_test(df, ic(), by="regime", factor_col="factor", rng_seed=42)
    b = slice_period_joint_test(df, ic(), by="regime", factor_col="factor", rng_seed=42)
    assert a["stat"][0] == b["stat"][0]


def test_rejects_bare_class() -> None:
    df = build_disjoint_period_panel(
        seed=6, spans={"a": (20, 0.0), "b": (20, 0.0)}, label_col="regime"
    )
    with pytest.raises(UserInputError, match="instance"):
        slice_period_joint_test(df, ic, by="regime", factor_col="factor")  # type: ignore[arg-type]


def test_rejects_non_eligible_metric() -> None:
    df = build_disjoint_period_panel(
        seed=7, spans={"a": (20, 0.0), "b": (20, 0.0)}, label_col="regime"
    )
    with pytest.raises(TypeError, match="slice-test-eligible"):
        slice_period_joint_test(df, monotonicity(), by="regime", factor_col="factor")


def test_rejects_invalid_method() -> None:
    df = build_disjoint_period_panel(
        seed=8, spans={"a": (20, 0.0), "b": (20, 0.0)}, label_col="regime"
    )
    with pytest.raises(UserInputError, match="method"):
        slice_period_joint_test(
            df,
            ic(),
            by="regime",
            factor_col="factor",
            method="welch",  # type: ignore[arg-type]
        )


def test_raises_when_single_slice() -> None:
    df = build_disjoint_period_panel(
        seed=9, spans={"only": (60, 0.0)}, label_col="regime"
    )
    with pytest.raises(ValueError, match="≥2 slice values"):
        slice_period_joint_test(df, ic(), by="regime", factor_col="factor")


def test_raises_when_slice_too_short() -> None:
    df = build_disjoint_period_panel(
        seed=10, spans={"a": (1, 0.1), "b": (40, 0.1)}, label_col="regime"
    )
    with pytest.raises(ValueError, match="<2 dates"):
        slice_period_joint_test(df, ic(), by="regime", factor_col="factor")


def test_raises_when_slice_below_metric_floor() -> None:
    """Below ic's min_periods floor (50) the omnibus refuses rather than emit
    an uncalibrated contrast — the same sub-floor size by_slice NaNs."""
    df = build_disjoint_period_panel(
        seed=11, spans={"a": (30, 0.1), "b": (30, 0.1)}, label_col="regime"
    )
    with pytest.raises(ValueError, match="sample floor"):
        slice_period_joint_test(df, ic(), by="regime", factor_col="factor", rng_seed=11)
