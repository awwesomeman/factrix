"""Date-disjoint pairwise cross-slice contrast verb (regime / period)."""

from __future__ import annotations

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix import slice_period_pairwise_test
from factrix._errors import UserInputError
from factrix.metrics import caar, fm_beta, ic, monotonicity

from tests._slice_panel import build_disjoint_period_panel

_PAIRWISE_COLS = [
    "slice_a",
    "slice_b",
    "n_periods_a",
    "n_periods_b",
    "mean_diff",
    "stat",
    "p_raw",
    "p_adj",
    "stat_type",
    "reference_dist",
    "df_num",
    "df_denom",
    "multiplicity",
]


@pytest.mark.parametrize(
    ("method", "reference_dist", "multiplicity"),
    [("bootstrap", "bootstrap_null", "romano_wolf"), ("analytic", "f", "holm")],
)
def test_two_slice_returns_one_row(
    method: str, reference_dist: str, multiplicity: str
) -> None:
    df = build_disjoint_period_panel(
        seed=1, spans={"bull": (60, 0.1), "bear": (60, 0.1)}, label_col="regime"
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng_seed=1
    )
    assert out.height == 1
    assert out.columns == _PAIRWISE_COLS
    assert out["stat_type"][0] == "wald"
    assert out["reference_dist"][0] == reference_dist
    assert out["df_num"][0] == 1
    if method == "bootstrap":
        assert out["df_denom"][0] is None
    else:
        # Welch-Satterthwaite ν: two equal-length slices of equal per-period
        # variance give ν ≈ 2(n−1). A wide-but-open lower bound (``1.0 <=``)
        # would silently accept the degenerate floor, so anchor at the
        # theoretical value instead.
        nu = out["df_denom"][0]
        n_a, n_b = out["n_periods_a"][0], out["n_periods_b"][0]
        assert nu is not None and nu == pytest.approx(n_a + n_b - 2, rel=0.1)
    assert out["multiplicity"][0] == multiplicity


def test_analytic_reference_is_welch_f() -> None:
    """The analytic p is ``F_{1, ν}`` on the disclosed ``df_denom``, not χ²₁.

    Two disjoint slices with their own HAC variances is the Welch two-sample
    setting, and the disclosed ``ν`` is what lets a reader recompute the p
    from ``stat`` alone.
    """
    from scipy import stats as sp_stats

    df = build_disjoint_period_panel(
        seed=4, spans={"a": (60, 0.1), "b": (120, 0.1)}, label_col="regime"
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", method="analytic"
    )
    stat, nu, p = out["stat"][0], out["df_denom"][0], out["p_raw"][0]
    assert p == pytest.approx(float(sp_stats.f.sf(stat, dfn=1, dfd=nu)))
    # Unequal slice lengths must give a non-integer, pair-specific ν.
    assert nu != pytest.approx(round(nu))


def test_welch_df_does_not_collapse_at_long_slices() -> None:
    """ν must track 2(n−1) at long slices, not fall to the 1.0 floor.

    The HAC variances of the mean shrink like σ²/n, so any absolute
    tolerance on the raw Welch denominator (∝ σ⁴/n³) trips on healthy data
    once the slices are long enough — pinning ν at 1.0 and turning the
    reference into F₁,₁, which has essentially no power. Regression for
    the scale-free share-based formula.
    """
    df = build_disjoint_period_panel(
        seed=7,
        spans={"a": (150, 0.1), "b": (150, 0.1), "c": (150, 0.1)},
        label_col="regime",
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", method="analytic"
    )
    for nu in out["df_denom"].to_list():
        assert nu == pytest.approx(298.0, rel=0.05)


def test_satterthwaite_df_is_scale_free() -> None:
    """Rescaling all variances by a constant leaves ν unchanged.

    The df is computed on variance *shares*, so there is no absolute
    quantity to underflow against a tolerance — the defect that pinned
    ν at 1.0 on larger slices before the shares form was adopted.
    """
    import numpy as np
    from factrix.slicing.period_inference import _satterthwaite_df

    v = np.array([2.0e-4, 5.0e-4])
    n = np.array([60, 120])
    base = _satterthwaite_df(v, n)
    assert base > 1.0
    for scale in (1e-6, 1e-3, 1e3):
        assert _satterthwaite_df(v * scale, n) == pytest.approx(base)
    # Only a genuinely degenerate set (all variances zero) hits the floor.
    assert _satterthwaite_df(np.zeros(2), n) == 1.0
    # K = 3 is the same function the omnibus uses; still scale-free.
    v3, n3 = np.array([1e-4, 2e-4, 3e-4]), np.array([60, 90, 120])
    assert _satterthwaite_df(v3 * 1e-5, n3) == pytest.approx(_satterthwaite_df(v3, n3))


def test_three_slice_returns_three_rows() -> None:
    df = build_disjoint_period_panel(
        seed=2,
        spans={"bull": (60, 0.1), "bear": (60, 0.1), "flat": (60, 0.1)},
        label_col="regime",
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", rng_seed=2
    )
    assert out.height == 3
    pairs = set(zip(out["slice_a"].to_list(), out["slice_b"].to_list(), strict=False))
    assert pairs == {("bull", "bear"), ("bull", "flat"), ("bear", "flat")}


def test_per_slice_period_counts_reported() -> None:
    """Disjoint spans differ in length → n_periods_a / n_periods_b differ."""
    df = build_disjoint_period_panel(
        seed=3, spans={"early": (50, 0.1), "late": (90, 0.1)}, label_col="regime"
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", rng_seed=3
    )
    row = out.row(0, named=True)
    assert row["n_periods_a"] == 50
    assert row["n_periods_b"] == 90


def test_disjoint_dates_do_not_raise() -> None:
    """The cross-sectional pair raises `<2 aligned dates` here; this pair runs."""
    df = build_disjoint_period_panel(
        seed=4, spans={"a": (50, 0.1), "b": (50, 0.1)}, label_col="regime"
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", rng_seed=4
    )
    assert out.height == 1
    assert np.isfinite(out["stat"][0])


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_detects_signal_difference(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=5, spans={"hot": (200, 0.4), "cold": (200, -0.1)}, label_col="regime"
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng_seed=5
    )
    assert out["p_raw"][0] < 0.05


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_mean_diff_sign_matches_direction(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=15, spans={"hot": (200, 0.4), "cold": (200, -0.1)}, label_col="regime"
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng_seed=6
    )
    row = out.row(0, named=True)
    # mean_diff = μ_a − μ_b; positive iff slice_a is the higher-IC regime.
    assert (row["mean_diff"] > 0) == (row["slice_a"] == "hot")


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_p_adj_dominates_raw(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=6,
        spans={"a": (120, 0.1), "b": (120, 0.2), "c": (120, -0.1)},
        label_col="regime",
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng_seed=7
    )
    for raw, adj in zip(out["p_raw"].to_list(), out["p_adj"].to_list(), strict=False):
        assert adj >= raw - 1e-12


def test_bootstrap_reproducible_under_seed() -> None:
    df = build_disjoint_period_panel(
        seed=8, spans={"a": (80, 0.1), "b": (80, -0.05)}, label_col="regime"
    )
    a = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", rng_seed=99
    )
    b = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", rng_seed=99
    )
    assert a["stat"].to_list() == b["stat"].to_list()
    assert a["p_adj"].to_list() == b["p_adj"].to_list()


def test_fama_macbeth_metric_accepted() -> None:
    df = build_disjoint_period_panel(
        seed=9, spans={"x": (60, 0.1), "y": (60, 0.1)}, label_col="regime"
    )
    out = slice_period_pairwise_test(
        df, fm_beta(), by="regime", factor_col="factor", rng_seed=9
    )
    assert out.height == 1
    assert out.columns == _PAIRWISE_COLS


def test_caar_metric_accepted_for_event_regimes() -> None:
    raw = fx.datasets.make_multi_factor_event_panel(
        n_factors=1,
        n_assets=30,
        n_dates=180,
        event_rate=0.18,
        post_event_drift_bps=30.0,
        seed=9,
    )
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
    midpoint = panel["date"].median()
    panel = panel.with_columns(
        pl.when(pl.col("date") < midpoint)
        .then(pl.lit("early"))
        .otherwise(pl.lit("late"))
        .alias("regime")
    )
    out = slice_period_pairwise_test(
        panel, caar(), by="regime", factor_col="factor_0000", rng_seed=9
    )
    assert out.height == 1
    assert out.columns == _PAIRWISE_COLS


def test_rejects_bare_class() -> None:
    df = build_disjoint_period_panel(
        seed=10, spans={"a": (20, 0.0), "b": (20, 0.0)}, label_col="regime"
    )
    with pytest.raises(UserInputError, match="instance"):
        slice_period_pairwise_test(df, ic, by="regime", factor_col="factor")  # type: ignore[arg-type]


def test_rejects_non_metric() -> None:
    def fake_metric(df: pl.DataFrame) -> None:
        return None

    df = build_disjoint_period_panel(
        seed=11, spans={"a": (20, 0.0), "b": (20, 0.0)}, label_col="regime"
    )
    with pytest.raises(UserInputError, match="metric instance"):
        slice_period_pairwise_test(df, fake_metric, by="regime", factor_col="factor")  # type: ignore[arg-type]


def test_rejects_non_eligible_metric() -> None:
    df = build_disjoint_period_panel(
        seed=12, spans={"a": (20, 0.0), "b": (20, 0.0)}, label_col="regime"
    )
    with pytest.raises(TypeError, match="slice-test-eligible"):
        slice_period_pairwise_test(df, monotonicity(), by="regime", factor_col="factor")


def test_rejects_invalid_method() -> None:
    df = build_disjoint_period_panel(
        seed=13, spans={"a": (20, 0.0), "b": (20, 0.0)}, label_col="regime"
    )
    with pytest.raises(UserInputError, match="method"):
        slice_period_pairwise_test(
            df,
            ic(),
            by="regime",
            factor_col="factor",
            method="hac",  # type: ignore[arg-type]
        )


def test_rejects_missing_factor_col() -> None:
    df = build_disjoint_period_panel(
        seed=14, spans={"a": (20, 0.0), "b": (20, 0.0)}, label_col="regime"
    )
    with pytest.raises(UserInputError, match="factor_col"):
        slice_period_pairwise_test(df, ic(), by="regime", factor_col="absent")


def test_raises_when_single_slice() -> None:
    df = build_disjoint_period_panel(
        seed=16, spans={"only": (60, 0.0)}, label_col="regime"
    )
    with pytest.raises(ValueError, match="≥2 slice values"):
        slice_period_pairwise_test(df, ic(), by="regime", factor_col="factor")


def test_raises_when_slice_too_short() -> None:
    df = build_disjoint_period_panel(
        seed=17, spans={"a": (1, 0.1), "b": (40, 0.1)}, label_col="regime"
    )
    with pytest.raises(ValueError, match="<2 dates"):
        slice_period_pairwise_test(df, ic(), by="regime", factor_col="factor")


def test_raises_when_slice_below_metric_floor() -> None:
    """A slice shorter than ic's min_periods floor (50) is the size at which
    by_slice short-circuits to NaN; the date-disjoint test must refuse rather
    than emit a contrast that is not calibrated."""
    df = build_disjoint_period_panel(
        seed=18, spans={"a": (30, 0.1), "b": (30, 0.1)}, label_col="regime"
    )
    with pytest.raises(ValueError, match="sample floor"):
        slice_period_pairwise_test(
            df, ic(), by="regime", factor_col="factor", rng_seed=18
        )


def test_runs_at_metric_floor() -> None:
    """The floor is strict (``<``): a slice exactly at ic's floor (50) runs."""
    df = build_disjoint_period_panel(
        seed=19, spans={"a": (50, 0.1), "b": (60, 0.1)}, label_col="regime"
    )
    out = slice_period_pairwise_test(
        df, ic(), by="regime", factor_col="factor", rng_seed=19
    )
    assert out.height == 1
