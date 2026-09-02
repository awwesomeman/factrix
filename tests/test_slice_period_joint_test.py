"""Date-disjoint omnibus cross-slice Wald verb (regime / period)."""

from __future__ import annotations

import math

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix import slice_period_joint_test, slice_period_pairwise_test
from factrix._errors import UserInputError
from factrix.metrics import ic, monotonicity, positive_rate

from tests._slice_panel import build_disjoint_period_panel, build_labelled_raw_panel

_JOINT_COLS = [
    "k_slices",
    "n_periods_min",
    "stat",
    "p_value",
    "stat_type",
    "reference_dist",
    "df_num",
    "df_denom",
    "multiplicity",
    "min_periods",
    "seed",
    "short_slice_periods",
    "warning_codes",
    "unexpected_warning_codes",
    "reason",
]


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_single_row_shape(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=1,
        spans={"bull": (80, 0.1), "bear": (80, 0.1), "flat": (80, 0.1)},
        label_col="regime",
    )
    out = slice_period_joint_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng=1
    )
    assert out.height == 1
    assert out.columns == _JOINT_COLS
    assert out["k_slices"][0] == 3
    assert out["df_num"][0] == 2
    assert out["stat_type"][0] == "wald"
    assert out["reference_dist"][0] == (
        "bootstrap_null" if method == "bootstrap" else "f"
    )
    if method == "bootstrap":
        assert out["df_denom"][0] is None
    else:
        # Satterthwaite ν on three 80-period slices: well above the 1.0
        # floor (which is the degenerate value) and at most the pooled dof.
        assert 50.0 < out["df_denom"][0] <= 3 * 80 - 3
    assert out["multiplicity"][0] is None


def test_analytic_omnibus_reference_is_satterthwaite_f() -> None:
    """The analytic omnibus p is ``F_{K-1, ν}`` on the disclosed ``df_denom``.

    Same finite-sample reference as the pairwise path, generalised to K
    slices; the earlier asymptotic χ²_{K-1} over-rejected at short slices.
    """
    from scipy import stats as sp_stats

    df = build_disjoint_period_panel(
        seed=5,
        spans={"a": (60, 0.1), "b": (90, 0.1), "c": (120, 0.1)},
        label_col="regime",
    )
    out = slice_period_joint_test(
        df, ic(), by="regime", factor_col="factor", method="analytic"
    )
    k, stat, nu, p = (
        out["k_slices"][0],
        out["stat"][0],
        out["df_denom"][0],
        out["p_value"][0],
    )
    assert p == pytest.approx(float(sp_stats.f.sf(stat / (k - 1), dfn=k - 1, dfd=nu)))
    assert nu != pytest.approx(round(nu))  # unequal slices → fractional ν


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_two_slice_df_is_one(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=2, spans={"a": (80, 0.1), "b": (80, 0.1)}, label_col="regime"
    )
    out = slice_period_joint_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng=2
    )
    assert out["df_num"][0] == 1


@pytest.mark.parametrize("method", ["bootstrap", "analytic"])
def test_detects_difference(method: str) -> None:
    df = build_disjoint_period_panel(
        seed=3,
        spans={"hot": (200, 0.4), "cold": (200, -0.2)},
        label_col="regime",
    )
    out = slice_period_joint_test(
        df, ic(), by="regime", factor_col="factor", method=method, rng=3
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
        same, ic(), by="regime", factor_col="factor", method=method, rng=4
    )["p_value"][0]
    p_split = slice_period_joint_test(
        split, ic(), by="regime", factor_col="factor", method=method, rng=4
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
    pw = slice_period_pairwise_test(df, ic(), by="regime", factor_col="factor", rng=5)
    jt = slice_period_joint_test(df, ic(), by="regime", factor_col="factor", rng=5)
    # Both empirical with the same B → identical 1/(B+1) granularity.
    assert jt["p_value"][0] == pytest.approx(pw["p_raw"][0])


def test_bootstrap_reproducible_under_seed() -> None:
    df = build_disjoint_period_panel(
        seed=5,
        spans={"a": (80, 0.1), "b": (80, -0.05), "c": (80, 0.0)},
        label_col="regime",
    )
    a = slice_period_joint_test(df, ic(), by="regime", factor_col="factor", rng=42)
    b = slice_period_joint_test(df, ic(), by="regime", factor_col="factor", rng=42)
    assert a["stat"][0] == b["stat"][0]


def test_bootstrap_omnibus_withholds_nonfinite_input() -> None:
    """An undefined Wald matrix cannot become the minimum empirical p."""
    from factrix.slicing.period_inference import _wald_bootstrap_omnibus

    stat, p_value = _wald_bootstrap_omnibus(
        np.array([0.1, -0.1]),
        np.array([[0.0, 0.1, -0.1], [0.0, -0.1, 0.1]]),
        np.array([1e-4, np.nan]),
        np.array([[1.0, -1.0]]),
    )

    assert math.isnan(stat)
    assert math.isnan(p_value)


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
    with pytest.raises(ValueError, match="≥2 distinct slice values"):
        slice_period_joint_test(df, ic(), by="regime", factor_col="factor")


def test_raises_when_slice_too_short() -> None:
    df = build_disjoint_period_panel(
        seed=10, spans={"a": (1, 0.1), "b": (40, 0.1)}, label_col="regime"
    )
    with pytest.raises(ValueError, match="≥2 per-period observations"):
        slice_period_joint_test(df, ic(), by="regime", factor_col="factor")


def test_raises_when_slice_below_metric_floor() -> None:
    """Below ic's min_periods floor (50) the omnibus refuses rather than emit
    an uncalibrated contrast — the same sub-floor size by_slice NaNs."""
    df = build_disjoint_period_panel(
        seed=11, spans={"a": (30, 0.1), "b": (30, 0.1)}, label_col="regime"
    )
    with pytest.raises(ValueError, match="sample floor"):
        slice_period_joint_test(df, ic(), by="regime", factor_col="factor", rng=11)


class TestShortSliceDisclosure:
    """Short slices with K >= 3 warn, and the measured size band is pinned.

    The characterisation test asserts the *measured* band, not ``<= nominal``:
    this path is known to over-reject on short slices. At 120 reps the band
    is ~±2.5 SE wide, so it pins the order of magnitude and catches a large
    regression (or a large improvement, which should prompt re-measuring
    the grid) — it will not detect a shift of a percentage point or two.
    """

    @staticmethod
    def _null_panel(seed: int, k: int, t: int):
        return build_disjoint_period_panel(
            seed=seed, spans={f"s{i}": (t, 0.1) for i in range(k)}, label_col="regime"
        )

    def test_short_slices_with_three_or_more_warn(self):
        with pytest.warns(UserWarning, match="over-rejects"):
            slice_period_joint_test(
                self._null_panel(0, 3, 60),
                ic(),
                by="regime",
                factor_col="factor",
                method="analytic",
            )

    @pytest.mark.parametrize(("k", "t"), [(2, 60), (3, 150)])
    def test_no_warning_outside_the_regime(self, k, t):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            slice_period_joint_test(
                self._null_panel(0, k, t),
                ic(),
                by="regime",
                factor_col="factor",
                method="analytic",
            )

    @pytest.mark.parametrize(
        ("k", "t", "low", "high"),
        [
            # measured 0.094 at 500 seeds; band is ~±2.5 SE at 120 reps
            (5, 50, 0.04, 0.16),
            # measured 0.052 at 500 seeds
            (2, 150, 0.01, 0.11),
        ],
    )
    def test_realised_size_band(self, k, t, low, high):
        import warnings

        reps = 120
        rejected = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(reps):
                out = slice_period_joint_test(
                    self._null_panel(30000 + seed, k, t),
                    ic(),
                    by="regime",
                    factor_col="factor",
                    method="analytic",
                )
                rejected += out["p_value"][0] < 0.05
        assert low <= rejected / reps <= high


class TestFloorFollowsPanelStamp:
    """The slice-test floor is resolved at the panel's stamped — or, on an
    unstamped panel, declared — ``overlap_periods``, the same value
    ``by_slice`` / ``evaluate`` resolve, not at the metric's default."""

    @staticmethod
    def _panel(overlap_periods: int | None) -> pl.DataFrame:
        return build_disjoint_period_panel(
            seed=7,
            spans={"a": (20, 0.02), "b": (20, 0.02)},
            label_col="regime",
            overlap_periods=overlap_periods,
        )

    def test_stamp_one_admits_twenty_period_slices(self) -> None:
        panel = self._panel(1)
        per_slice = fx.by_slice(
            panel, positive_rate(), by="regime", factor_col="factor", strict=False
        )
        assert all(r.metrics["positive_rate"].n_obs == 20 for r in per_slice.values())
        joint = slice_period_joint_test(
            panel, positive_rate(), by="regime", factor_col="factor", rng=1
        )
        assert joint["k_slices"].item() == 2
        pairwise = slice_period_pairwise_test(
            panel, positive_rate(), by="regime", factor_col="factor", rng=1
        )
        assert pairwise.select("n_periods_a", "n_periods_b").row(0) == (20, 20)

    def test_stamp_five_refuses_and_names_the_horizon(self) -> None:
        with pytest.raises(
            ValueError, match=r"floor \(50, resolved at .*overlap_periods=5"
        ):
            slice_period_joint_test(
                self._panel(5), positive_rate(), by="regime", factor_col="factor"
            )

    def test_unstamped_panel_requires_a_declared_overlap(self) -> None:
        """No silent fallback to the metric's default: declare it, as
        ``evaluate`` demands on the same panel."""
        with pytest.raises(UserInputError, match="overlap_periods"):
            slice_period_pairwise_test(
                self._panel(None),
                positive_rate(),
                by="regime",
                factor_col="factor",
            )

    def test_declared_overlap_resolves_the_floor(self) -> None:
        """A declared overlap of 1 admits the same 20-period slices a stamp
        of 1 does."""
        panel = self._panel(None)
        joint = slice_period_joint_test(
            panel,
            positive_rate(),
            by="regime",
            factor_col="factor",
            overlap_periods=1,
            rng=1,
        )
        assert joint["k_slices"].item() == 2
        assert joint["min_periods"].item() == 10
        pairwise = slice_period_pairwise_test(
            panel,
            positive_rate(),
            by="regime",
            factor_col="factor",
            overlap_periods=1,
            rng=1,
        )
        assert pairwise.select("n_periods_a", "n_periods_b").row(0) == (20, 20)

    def test_declared_overlap_disagreeing_with_the_stamp_raises(self) -> None:
        with pytest.raises(UserInputError, match="stamped evaluation-grid overlap"):
            slice_period_joint_test(
                self._panel(1),
                positive_rate(),
                by="regime",
                factor_col="factor",
                overlap_periods=5,
            )

    def test_by_slice_and_slice_test_agree_on_an_unstamped_panel(self) -> None:
        """The descriptive and the inferential path resolve one overlap."""
        panel = self._panel(None)
        per_slice = fx.by_slice(
            panel,
            positive_rate(),
            by="regime",
            factor_col="factor",
            forward_periods=1,
            overlap_periods=1,
            strict=False,
        )
        assert all(r.metrics["positive_rate"].n_obs == 20 for r in per_slice.values())
        assert all(r.metrics["positive_rate"].is_applicable for r in per_slice.values())
        joint = slice_period_joint_test(
            panel,
            positive_rate(),
            by="regime",
            factor_col="factor",
            overlap_periods=1,
            rng=1,
        )
        assert joint["reason"].item() is None
        assert (
            joint["min_periods"].item()
            == fx.sample_requirements(positive_rate(), overlap_periods=1).min_periods
        )


class TestNonStrict:
    """``strict=False``: a thin slice yields a structured unavailable row."""

    @staticmethod
    def _panel() -> pl.DataFrame:
        # ic() floor at the default horizon is 50; "bear" sits below it.
        return build_disjoint_period_panel(
            seed=5,
            spans={"bull": (60, 0.1), "bear": (30, 0.1), "flat": (60, 0.1)},
            label_col="regime",
        )

    def test_strict_default_still_raises(self) -> None:
        with pytest.raises(ValueError, match="sample floor"):
            slice_period_joint_test(
                self._panel(), ic(), by="regime", factor_col="factor"
            )

    @pytest.mark.parametrize("method", ["bootstrap", "analytic"])
    def test_thin_slice_returns_unavailable_row(self, method: str) -> None:
        out = slice_period_joint_test(
            self._panel(),
            ic(),
            by="regime",
            factor_col="factor",
            method=method,
            strict=False,
        )
        assert out.columns == _JOINT_COLS
        row = out.row(0, named=True)
        assert row["reason"] == "insufficient_periods"
        assert (row["k_slices"], row["n_periods_min"], row["min_periods"]) == (
            3,
            30,
            50,
        )
        assert math.isnan(row["stat"]) and math.isnan(row["p_value"])
        assert row["df_denom"] is None

    def test_tested_row_carries_null_reason_and_floor(self) -> None:
        df = build_disjoint_period_panel(
            seed=5, spans={"a": (60, 0.1), "b": (60, 0.1)}, label_col="regime"
        )
        out = slice_period_joint_test(
            df, ic(), by="regime", factor_col="factor", rng=1, strict=False
        )
        row = out.row(0, named=True)
        assert row["reason"] is None
        assert (row["n_periods_min"], row["min_periods"]) == (60, 50)
        assert math.isfinite(row["p_value"])

    def test_non_strict_matches_strict_when_all_clear(self) -> None:
        df = build_disjoint_period_panel(
            seed=5, spans={"a": (60, 0.1), "b": (60, 0.1)}, label_col="regime"
        )
        kw = dict(by="regime", factor_col="factor", rng=1)
        strict = slice_period_joint_test(df, ic(), **kw)
        loose = slice_period_joint_test(df, ic(), strict=False, **kw)
        assert strict.equals(loose)


class TestShortSliceWarningContract:
    """#879 — the short-slice advisory is a structured, declarable record.

    The row carries the code in ``warning_codes`` whether or not the caller
    declared it; ``expected_warnings`` only keeps it out of
    ``unexpected_warning_codes`` and stops the stderr echo — the same
    marked-never-dropped contract as ``evaluate``. The triggering context
    (``k_slices``, ``n_periods_min``, ``short_slice_periods``) sits beside it.
    """

    @staticmethod
    def _null_panel(seed: int, k: int, t: int):
        return build_disjoint_period_panel(
            seed=seed, spans={f"s{i}": (t, 0.1) for i in range(k)}, label_col="regime"
        )

    @staticmethod
    def _run(panel, **kwargs):
        return slice_period_joint_test(
            panel, ic(), by="regime", factor_col="factor", method="analytic", **kwargs
        )

    def test_undeclared_regime_is_recorded_and_echoed(self):
        with pytest.warns(UserWarning, match="short_slice_joint_test"):
            out = self._run(self._null_panel(0, 3, 60))
        row = out.row(0, named=True)
        assert row["warning_codes"] == ["short_slice_joint_test"]
        assert row["unexpected_warning_codes"] == ["short_slice_joint_test"]
        assert row["k_slices"] == 3
        assert row["n_periods_min"] == 60
        assert row["short_slice_periods"] == 150
        assert row["reason"] is None

    def test_declared_regime_keeps_the_record_and_stops_the_echo(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = self._run(
                self._null_panel(0, 3, 60),
                expected_warnings=("short_slice_joint_test",),
            )
        row = out.row(0, named=True)
        assert row["warning_codes"] == ["short_slice_joint_test"]
        assert row["unexpected_warning_codes"] == []
        assert math.isfinite(row["p_value"])

    def test_batch_of_candidates_prints_nothing_when_declared(self):
        """The reporter's case: one call per candidate, the same accepted
        limitation every time, zero stderr noise, every row still auditable."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = [
                self._run(
                    self._null_panel(seed, 3, 60),
                    expected_warnings=("short_slice_joint_test",),
                )
                for seed in range(5)
            ]
        assert not [w for w in caught if issubclass(w.category, UserWarning)]
        stacked = pl.concat(rows)
        assert stacked["warning_codes"].to_list() == [["short_slice_joint_test"]] * 5
        assert stacked["unexpected_warning_codes"].to_list() == [[]] * 5

    @pytest.mark.parametrize(("k", "t"), [(2, 60), (3, 150)])
    def test_outside_the_regime_the_lists_are_empty(self, k, t):
        row = self._run(self._null_panel(0, k, t)).row(0, named=True)
        assert row["warning_codes"] == []
        assert row["unexpected_warning_codes"] == []
        assert row["short_slice_periods"] == 150

    def test_unavailable_row_raises_nothing(self):
        df = build_disjoint_period_panel(
            seed=11,
            spans={"a": (30, 0.1), "b": (80, 0.1), "c": (80, 0.1)},
            label_col="regime",
        )
        row = self._run(df, strict=False).row(0, named=True)
        assert row["reason"] == "insufficient_periods"
        assert row["warning_codes"] == []
        assert row["unexpected_warning_codes"] == []

    def test_unknown_code_is_rejected_under_this_functions_name(self):
        with pytest.raises(UserInputError, match="slice_period_joint_test"):
            self._run(self._null_panel(0, 3, 60), expected_warnings=("nope",))
        with pytest.raises(UserInputError, match="expected_warnings"):
            self._run(
                self._null_panel(0, 3, 60),
                expected_warnings="short_slice_joint_test",  # type: ignore[arg-type]
            )


class TestDateAlignedPartitionRefused:
    """Mirror of the cross-sectional `<2 aligned dates` guard."""

    def test_fully_aligned_partition_names_the_cross_sectional_entry_point(
        self,
    ) -> None:
        df = build_labelled_raw_panel(
            n_dates=120, seed=21, signal={"tech": 0.2, "fin": 0.05}, label_col="sector"
        )
        with pytest.raises(UserInputError) as excinfo:
            slice_period_joint_test(
                df, ic(), by="sector", factor_col="factor", overlap_periods=5
            )
        message = str(excinfo.value)
        assert "share 120 dates" in message
        assert "slice_joint_test" in message
        assert excinfo.value.field == "by"
        assert excinfo.value.value == "sector"

    def test_partially_overlapping_spans_refused(self) -> None:
        df = build_disjoint_period_panel(
            seed=22,
            spans={"early": (80, 0.1), "late": (80, 0.1)},
            label_col="regime",
            shared_periods=8,
        )
        with pytest.raises(UserInputError, match="share 8 dates"):
            slice_period_joint_test(df, ic(), by="regime", factor_col="factor")

    def test_single_boundary_date_tolerated(self) -> None:
        df = build_disjoint_period_panel(
            seed=23,
            spans={"early": (80, 0.1), "late": (80, 0.1)},
            label_col="regime",
            shared_periods=1,
        )
        out = slice_period_joint_test(
            df, ic(), by="regime", factor_col="factor", rng=23
        )
        assert out.height == 1
        assert math.isfinite(out["stat"][0])
