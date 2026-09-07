"""Tests for factrix.metrics.oos_decay."""

import math

import pytest
from factrix._results import MetricResult
from factrix.metrics.oos_decay import oos_decay, oos_decay_splits


class TestOOSDecay:
    def test_stable_series_passes(self, ic_series_positive):
        result = oos_decay(ic_series_positive)
        assert result.metadata["status"] == "PASS"
        assert result.value > 0.5
        assert result.metadata["sign_flipped"] is False

    def test_sign_flip_vetoed(self, ic_series_sign_flip):
        result = oos_decay(ic_series_sign_flip)
        assert result.metadata["status"] == "VETOED"
        assert result.metadata["sign_flipped"] is True

    def test_insufficient_data(self):
        from datetime import datetime, timedelta

        import polars as pl

        # Only 6 rows — below MIN_OOS_PERIODS_HARD * 2 = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)]
        series = pl.DataFrame({"date": dates, "value": [0.01] * 6}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = oos_decay(series)
        assert result.metadata["status"] == "VETOED"
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_oos_periods"

    def test_custom_is_ratio(self, ic_series_positive):
        result = oos_decay(ic_series_positive, is_ratio=0.5)
        assert result.metadata["is_ratio"] == 0.5

    def test_survival_below_threshold_vetoed(self):
        from datetime import datetime, timedelta

        import numpy as np
        import polars as pl

        rng = np.random.default_rng(99)
        # IS strong, OOS very weak
        is_vals = rng.normal(0.10, 0.01, 30)
        oos_vals = rng.normal(0.01, 0.01, 20)
        values = np.concatenate([is_vals, oos_vals])
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        series = pl.DataFrame({"date": dates, "value": values}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = oos_decay(series, is_ratio=0.6, survival_threshold=0.5)
        # OOS mean / IS mean ≈ 0.01/0.10 = 0.1 < 0.5
        assert result.metadata["status"] == "VETOED"
        assert result.metadata["sign_flipped"] is False

    def test_returns_metric_output(self, ic_series_positive):
        """Single-contract check: oos_decay returns MetricResult."""

        result = oos_decay(ic_series_positive)
        assert isinstance(result, MetricResult)
        assert result.stat is None  # descriptive, not hypothesis test
        # Descriptive-only: no p_value emitted (would invite mis-routing
        # the diagnostic into BHY / gate logic).
        assert "p_value" not in result.metadata

    def test_metadata_shape(self, ic_series_positive):
        """metadata carries the single-split fields."""
        result = oos_decay(ic_series_positive)
        assert set(result.metadata.keys()) >= {
            "sign_flipped",
            "status",
            "is_ratio",
            "mean_is",
            "mean_oos",
            "survival_threshold",
        }


def test_oos_decay_ignores_nan_observations():
    """Float NaN must be dropped like a null, not fed into the IS / OOS means."""
    from datetime import datetime, timedelta

    import numpy as np
    import polars as pl
    import pytest

    rng = np.random.default_rng(1)
    vals = list(rng.normal(0.05, 0.02, 80))
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(80)]
    clean = pl.DataFrame({"date": dates, "value": vals}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )
    dirty_vals = vals + [float("nan")] * 10
    dirty_dates = dates + [dates[-1] + timedelta(days=i + 1) for i in range(10)]
    dirty = pl.DataFrame({"date": dirty_dates, "value": dirty_vals}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )
    r_dirty, r_clean = oos_decay(dirty), oos_decay(clean)
    assert math.isfinite(r_dirty.value)
    assert r_dirty.value == pytest.approx(r_clean.value)
    assert r_dirty.n_obs == r_clean.n_obs == 80


class TestIsRatioValidation:
    """Regression: is_ratio=1.0 produced an empty OOS slice → TypeError."""

    @pytest.mark.parametrize("is_ratio", [1.0, 0.0, -0.1, 1.5])
    def test_out_of_range_is_ratio_raises(self, is_ratio, ic_series_positive):
        with pytest.raises(ValueError, match="is_ratio"):
            oos_decay(ic_series_positive, is_ratio=is_ratio)

    def test_extreme_ratio_short_circuits_instead_of_splitting_to_one(self):
        """A one-observation window is not a window."""
        from datetime import datetime, timedelta

        import polars as pl

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)]
        series = pl.DataFrame(
            {"date": dates, "value": [0.01 * (i + 1) for i in range(12)]}
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

        result = oos_decay(series, is_ratio=0.95)
        assert result.metadata["reason"] == "insufficient_oos_periods"
        assert result.metadata["status"] == "VETOED"
        assert math.isnan(result.value)

        low = oos_decay(series, is_ratio=0.05)
        assert low.metadata["reason"] == "insufficient_oos_periods"

    def test_balanced_ratio_still_computes(self, ic_series_positive):
        result = oos_decay(ic_series_positive, is_ratio=0.5)
        assert result.metadata.get("reason") is None
        assert math.isfinite(result.value)


def _series(values: list[float]):
    from datetime import datetime, timedelta

    import polars as pl

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "value": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestDegenerateInSampleMean:
    """``survival = 0.0`` when ``mean_is ~ 0`` was a silent mis-verdict: a
    series with no in-sample signal and a LARGE out-of-sample mean scored
    0.0 and read as fully decayed."""

    #: Default is_ratio=0.7 splits 40 rows at index 28: the first 28 are
    #: 14 [+1, -1] pairs (mean exactly 0) and the last 12 are 5.0, so the
    #: ratio is 0/0 while the out-of-sample mean is as large as it gets.
    _NO_IS_SIGNAL = [1.0, -1.0] * 14 + [5.0] * 12

    def test_zero_is_mean_withholds_the_ratio(self):
        from factrix._codes import WarningCode

        result = oos_decay(_series(self._NO_IS_SIGNAL))
        assert math.isnan(result.value)
        assert WarningCode.DEGENERATE_VARIANCE.value in result.warning_codes
        assert result.metadata["signal_status"] == "degenerate_zero_variance"
        assert result.metadata["mean_is"] == pytest.approx(0.0, abs=1e-12)
        assert result.metadata["mean_oos"] == pytest.approx(5.0)

    def test_gate_stays_vetoed_rather_than_passing(self):
        """A gate must not read "cannot assess" as "passed"."""
        assert oos_decay(_series(self._NO_IS_SIGNAL)).metadata["status"] == "VETOED"

    def test_ordinary_split_is_unaffected(self):
        result = oos_decay(_series([1.0] * 20 + [0.5] * 20), is_ratio=0.5)
        assert result.value == pytest.approx(0.5)
        assert "signal_status" not in result.metadata


class TestSurvivalThresholdValidation:
    """``survival_threshold`` is a retention fraction, not a free float.

    An unvalidated threshold silently forces the gate: ``-1.0`` passes every
    series (a ratio is never negative), ``float("nan")`` fails every
    comparison and so VETOES every series, and ``True`` is ``1.0`` to Python.
    """

    @pytest.mark.parametrize(
        "survival_threshold",
        [
            0.0,
            -0.1,
            1.5,
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            "0.5",
            None,
        ],
    )
    def test_out_of_domain_threshold_raises(
        self, survival_threshold, ic_series_positive
    ):
        from factrix import UserInputError

        with pytest.raises(UserInputError, match="survival_threshold"):
            oos_decay(ic_series_positive, survival_threshold=survival_threshold)

    @pytest.mark.parametrize("survival_threshold", [1e-9, 0.5, 1.0])
    def test_in_domain_threshold_accepted(self, survival_threshold, ic_series_positive):
        result = oos_decay(ic_series_positive, survival_threshold=survival_threshold)
        assert result.metadata["survival_threshold"] == survival_threshold

    def test_threshold_boundary_is_inclusive(self):
        """``survival == survival_threshold`` PASSes: the gate is ``>=``."""
        series = _series([1.0] * 20 + [0.5] * 20)
        assert (
            oos_decay(series, is_ratio=0.5, survival_threshold=0.5).metadata["status"]
            == "PASS"
        )
        assert (
            oos_decay(series, is_ratio=0.5, survival_threshold=0.5 + 1e-9).metadata[
                "status"
            ]
            == "VETOED"
        )


class TestOneRowPerPeriod:
    """The series is indexed by period: one observation per distinct date."""

    def test_duplicate_dates_rejected(self):
        from datetime import datetime, timedelta

        import polars as pl
        from factrix import UserInputError

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)]
        # The 15th period carries two observations, so the row-count split
        # can cut *between* them and put one period on both sides.
        dup_dates = [*dates, dates[14]]
        values = [0.02] * 20 + [-5.0]
        series = pl.DataFrame({"date": dup_dates, "value": values}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        with pytest.raises(UserInputError, match="one row per period"):
            oos_decay(series)

    def test_infinite_values_are_dropped_like_nan(self):
        """±inf is not a finite observation; it must not reach a mean."""
        from datetime import datetime, timedelta

        import numpy as np
        import polars as pl

        rng = np.random.default_rng(7)
        vals = list(rng.normal(0.05, 0.02, 40))
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
        clean = pl.DataFrame({"date": dates, "value": vals}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        dirty = pl.DataFrame(
            {
                "date": dates + [dates[-1] + timedelta(days=i + 1) for i in range(2)],
                "value": [*vals, float("inf"), float("-inf")],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

        r_dirty, r_clean = oos_decay(dirty), oos_decay(clean)
        assert math.isfinite(r_dirty.value)
        assert r_dirty.value == pytest.approx(r_clean.value)
        assert r_dirty.n_obs == r_clean.n_obs == 40

    def test_missing_observations_do_not_shift_the_period_split(self):
        """The split counts *retained* periods, so a null period drops out
        of the grid entirely rather than straddling the boundary."""
        from datetime import datetime, timedelta

        import polars as pl

        values: list[float | None] = [1.0] * 20 + [0.5] * 20
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
        with_gaps = pl.DataFrame(
            {
                "date": dates + [dates[-1] + timedelta(days=i + 1) for i in range(2)],
                "value": values + [None] * 2,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        compact = pl.DataFrame({"date": dates, "value": values}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        gapped = oos_decay(with_gaps, is_ratio=0.5)
        assert gapped.value == pytest.approx(oos_decay(compact, is_ratio=0.5).value)
        assert gapped.value == pytest.approx(0.5)
        assert gapped.n_obs == 40

    def test_direct_call_matches_dag_produced_series(self):
        """A ``compute_ic`` frame is one row per period; the direct call on
        the same frame must agree with the DAG-routed one."""
        import factrix as fx
        from factrix.metrics.ic import compute_ic
        from factrix.metrics.oos_decay import oos_decay as _oos
        from factrix.preprocess import compute_forward_return

        panel = compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=40, n_dates=120, rng=3),
            forward_periods=5,
        )
        ic_df = compute_ic(panel)["factor"]
        assert ic_df["date"].n_unique() == ic_df.height
        direct = _oos(ic_df, value_col="ic")
        routed = fx.evaluate(
            panel, metrics={"oos_decay": _oos()}, factor_cols=["factor"]
        )["factor"].metrics["oos_decay"]
        assert direct.value == pytest.approx(routed.value)
        assert direct.metadata["status"] == routed.metadata["status"]


#: A structural break parked just before the default 0.7 cut point. The last
#: 30 periods sum to exactly zero, so the 0.7 split reads an out-of-sample
#: mean of 0 and VETOES; the 0.6 and 0.8 splits both survive comfortably.
_BREAK_AT_070 = [1.0] * 60 + [4.0] * 10 + [-4.0] * 10 + [2.0] * 20

#: Same shape, but the last 30 periods sum to ``-1`` instead of ``0``: the
#: 0.7 split now reads a *negative* out-of-sample mean against a positive
#: in-sample mean, so exactly one of the three splits sign-flips while the
#: median survival ratio stays well above the threshold.
_FLIP_AT_070 = [1.0] * 60 + [4.0] * 10 + [-4.0] * 10 + [1.95] * 20


class TestSplitSweepContract:
    """The declared split set, its aggregate rule, and its provenance."""

    def test_every_declared_split_is_returned(self):
        result = oos_decay_splits(_series(_BREAK_AT_070), forward_periods=1)
        splits = result.metadata["splits"]
        assert [s["split_fraction"] for s in splits] == [0.6, 0.7, 0.8]
        for split in splits:
            assert set(split) >= {
                "split_fraction",
                "n_is",
                "n_oos",
                "purge_periods",
                "survival",
                "sign_flipped",
                "status",
            }

    def test_provenance_records_what_ran(self):
        result = oos_decay_splits(
            _series(_BREAK_AT_070), split_fractions=(0.5, 0.75), forward_periods=3
        )
        assert result.metadata["split_fractions"] == (0.5, 0.75)
        assert result.metadata["n_splits"] == 2
        assert result.metadata["aggregate"] == "median"
        assert result.metadata["sign_flip_policy"] == "any_flip_vetoes"
        assert result.metadata["forward_periods"] == 3
        assert result.metadata["purge_periods"] == 3
        assert result.metadata["survival_threshold"] == 0.5
        assert result.n_obs == 100
        assert result.n_obs_axis == "periods"
        assert result.stat is None

    def test_aggregation_is_order_invariant(self):
        series = _series(_BREAK_AT_070)
        forward = dict(split_fractions=(0.6, 0.7, 0.8), forward_periods=1)
        reverse = dict(split_fractions=(0.8, 0.7, 0.6), forward_periods=1)
        shuffled = dict(split_fractions=(0.7, 0.6, 0.8), forward_periods=1)
        a = oos_decay_splits(series, **forward)
        b = oos_decay_splits(series, **reverse)
        c = oos_decay_splits(series, **shuffled)
        assert a.value == b.value == c.value
        assert a.metadata == b.metadata == c.metadata

    def test_structural_break_near_one_cut_point_does_not_decide_the_gate(self):
        """The whole point of the sweep: one arbitrary cut reverses the
        single-split gate, the pre-declared set does not follow it."""
        series = _series(_BREAK_AT_070)
        single = oos_decay(series, is_ratio=0.7)
        assert single.metadata["status"] == "VETOED"
        assert single.value < 0.5

        swept = oos_decay_splits(series, forward_periods=1)
        per_split = {s["split_fraction"]: s["status"] for s in swept.metadata["splits"]}
        assert per_split == {0.6: "PASS", 0.7: "VETOED", 0.8: "PASS"}
        assert swept.metadata["status"] == "PASS"
        assert swept.value == pytest.approx(1.0, rel=1e-6)

    def test_any_sign_flip_vetoes_even_when_the_median_survives(self):
        """Direction is a unanimity requirement, magnitude is a median."""
        result = oos_decay_splits(_series(_FLIP_AT_070), forward_periods=1)
        assert result.metadata["n_sign_flips"] == 1
        assert result.metadata["status"] == "VETOED"
        # The aggregate ratio is still reported — it is what ran; the veto
        # comes from the flip policy, not from the magnitude.
        assert result.value == pytest.approx(0.975, rel=1e-6)
        assert result.value > result.metadata["survival_threshold"]


#: Flat probe for the purge tests: every period carries the same value, so
#: anything that leaks out of the purge gap moves the answer loudly instead
#: of cancelling against the rest of the window.
_PURGE_PROBE = [1.0] * 100


class TestSplitSweepPurge:
    """Overlapping forward-return windows are purged off the IS tail."""

    def test_purge_shortens_is_only(self):
        result = oos_decay_splits(
            _series(_PURGE_PROBE), split_fractions=(0.7,), forward_periods=5
        )
        (split,) = result.metadata["splits"]
        assert split["purge_periods"] == 5
        assert split["n_is"] == 65  # int(100 * 0.7) - 5
        assert split["n_oos"] == 30  # the validated window is never shortened

    def test_purged_periods_cannot_reach_the_statistic(self):
        """The leakage guard: values inside the purge gap are not read.

        Both halves are asserted. Poisoning periods 65-69 has to *move* the
        un-purged single-split answer — otherwise the second half of this
        test would pass on any implementation, purge or no purge.
        """
        base = list(_PURGE_PROBE)
        poisoned = list(base)
        poisoned[65:70] = [1000.0] * 5  # the 5 periods the purge removes at 0.7

        unpurged_clean = oos_decay(_series(base), is_ratio=0.7)
        unpurged_dirty = oos_decay(_series(poisoned), is_ratio=0.7)
        assert unpurged_clean.value == pytest.approx(1.0)
        assert unpurged_clean.metadata["status"] == "PASS"
        assert unpurged_dirty.value < 0.02
        assert unpurged_dirty.metadata["status"] == "VETOED"

        kwargs = dict(split_fractions=(0.7,), forward_periods=5)
        clean = oos_decay_splits(_series(base), **kwargs)
        dirty = oos_decay_splits(_series(poisoned), **kwargs)
        assert clean.value == pytest.approx(1.0)
        assert dirty.value == pytest.approx(clean.value)
        assert dirty.metadata["splits"] == clean.metadata["splits"]
        assert dirty.metadata["status"] == "PASS"

    def test_a_one_period_purge_still_drops_a_period(self):
        """``forward_periods=1`` on a contemporaneous series still drops the
        single overlapping period; the split index itself is unchanged."""
        result = oos_decay_splits(
            _series(_PURGE_PROBE), split_fractions=(0.7,), forward_periods=1
        )
        (split,) = result.metadata["splits"]
        assert (split["n_is"], split["n_oos"]) == (69, 30)


class TestSplitSweepValidation:
    def test_empty_split_set_rejected(self):
        from factrix import UserInputError

        with pytest.raises(UserInputError, match="split_fractions"):
            oos_decay_splits(_series(_BREAK_AT_070), split_fractions=())

    def test_duplicate_fractions_rejected(self):
        """A repeated fraction would double-weight the median."""
        from factrix import UserInputError

        with pytest.raises(UserInputError, match="split_fractions"):
            oos_decay_splits(_series(_BREAK_AT_070), split_fractions=(0.6, 0.6, 0.8))

    @pytest.mark.parametrize("bad", [1.0, 0.0, -0.2, float("nan"), True, "0.7"])
    def test_out_of_range_fraction_rejected(self, bad):
        from factrix import UserInputError

        with pytest.raises(UserInputError, match="split_fractions"):
            oos_decay_splits(_series(_BREAK_AT_070), split_fractions=(0.6, bad))

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5])
    def test_non_count_forward_periods_rejected(self, bad):
        from factrix import UserInputError

        with pytest.raises(UserInputError, match="forward_periods"):
            oos_decay_splits(_series(_BREAK_AT_070), forward_periods=bad)

    @pytest.mark.parametrize("bad", [0.0, 1.5, float("nan"), True])
    def test_threshold_domain_matches_the_primitive(self, bad):
        from factrix import UserInputError

        with pytest.raises(UserInputError, match="survival_threshold"):
            oos_decay_splits(_series(_BREAK_AT_070), survival_threshold=bad)

    def test_duplicate_dates_rejected_like_the_primitive(self):
        from datetime import datetime, timedelta

        import polars as pl
        from factrix import UserInputError

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
        series = pl.DataFrame(
            {"date": [*dates, dates[9]], "value": [1.0] * 41}
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        with pytest.raises(UserInputError, match="one row per period"):
            oos_decay_splits(series)


class TestSplitSweepUnassessable:
    """A withheld split is not a passing one, and the aggregate is over the
    *declared* set, not over whichever subset happened to work."""

    def test_a_short_circuited_split_withholds_the_aggregate(self):
        # 18 periods, purge 8: the 0.5 cut leaves a 1-period in-sample
        # window, the 0.9 cut is still assessable.
        result = oos_decay_splits(
            _series([1.0] * 12 + [0.9] * 6),
            split_fractions=(0.5, 0.9),
            forward_periods=8,
        )
        assert math.isnan(result.value)
        assert result.metadata["status"] == "VETOED"
        assert result.metadata["reason"] == "unassessable_splits"
        assert result.metadata["n_assessable"] == 1
        assert result.metadata["n_splits"] == 2
        # Every declared split is still reported, for diagnosis — and the
        # purge-emptied one says so, rather than borrowing the primitive's
        # "the series is too short" sentinel.
        first, second = result.metadata["splits"]
        assert first["reason"] == "purged_split_too_short"
        assert math.isfinite(second["survival"])
