"""Tests for factrix.metrics.oos_decay."""

import math

import pytest
from factrix._results import MetricResult
from factrix.metrics.oos_decay import oos_decay


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
