"""Tests for factrix.metrics.monotonicity."""

import math

import factrix as fx
import pytest
from factrix.metrics.monotonicity import monotonicity


class TestComputeMonotonicity:
    @staticmethod
    def _perfect(panel):
        import polars as pl

        return panel.with_columns(
            pl.col("factor").rank(method="average").over("date").alias("forward_return")
        )

    def test_perfect_monotonic(self, noisy_panel):
        # WHY: tiny_panel only has 3 dates, < MIN_MONOTONICITY_PERIODS_HARD after
        # sampling. Use noisy_panel (20 dates × 30 assets) with perfect
        # factor-return alignment.
        result = monotonicity(
            self._perfect(noisy_panel), overlap_periods=1, n_groups=5, seed=0
        )["factor"]
        # Every adjacent bucket step is strictly positive and identical across
        # periods, so the MR statistic is the (positive) common step.
        assert result.value > 0
        assert result.metadata["mean_abs_spearman"] == pytest.approx(1.0)
        assert result.metadata["mean_signed"] == pytest.approx(1.0)

    def test_perfect_monotonicity_is_maximal_evidence(self, noisy_panel):
        """A perfectly ordered factor gets the smallest attainable p.

        The old headline reported ``value = mean|Spearman| = 1.0`` with the t
        withheld — a constant signed series has no dispersion to form an SE
        from. The MR test has no such hole: the recentred bootstrap puts every
        draw at 0 while the observed J sits above it, so p is the floor
        ``1 / (B + 1)``.
        """
        n_bootstrap = 200
        result = monotonicity(
            self._perfect(noisy_panel),
            overlap_periods=1,
            n_groups=5,
            n_bootstrap=n_bootstrap,
            seed=0,
        )["factor"]
        assert result.p_value == pytest.approx(1 / (n_bootstrap + 1))
        assert result.alternative == "greater"
        assert result.stat == result.value
        assert result.metadata["stat_type"] == "mr"
        assert result.warning_codes == ()
        # The descriptive Spearman pair is preserved, not the headline.
        assert result.metadata["mean_abs_spearman"] == pytest.approx(1.0)
        assert result.metadata["mean_signed"] == pytest.approx(1.0)

    def test_inverse_monotonic(self, noisy_panel):
        import polars as pl

        inverted = noisy_panel.with_columns(
            (-pl.col("factor").rank(method="average").over("date")).alias(
                "forward_return"
            )
        )
        result = monotonicity(inverted, overlap_periods=1, n_groups=5, seed=0)["factor"]
        # H1 is "increasing" by default, so a perfectly decreasing pattern is
        # the furthest thing from it: J < 0 and the p is at its ceiling.
        assert result.value < 0
        assert result.p_value == pytest.approx(1.0)
        assert result.metadata["mean_abs_spearman"] == pytest.approx(1.0)
        assert result.metadata["mean_signed"] == pytest.approx(-1.0)

    def test_direction_declares_the_alternative(self, noisy_panel):
        import polars as pl

        inverted = noisy_panel.with_columns(
            (-pl.col("factor").rank(method="average").over("date")).alias(
                "forward_return"
            )
        )
        result = monotonicity(
            inverted,
            overlap_periods=1,
            n_groups=5,
            direction="decreasing",
            n_bootstrap=200,
            seed=0,
        )["factor"]
        assert result.value > 0
        assert result.p_value == pytest.approx(1 / 201)
        assert result.metadata["mr_direction"] == "decreasing"

    def test_insufficient_periods(self):
        from datetime import datetime

        import polars as pl

        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "asset_id": ["A", "B", "C", "D", "E"],
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
                "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = monotonicity(df, overlap_periods=1, n_groups=5)["factor"]
        # Only 1 date < MIN_MONOTONICITY_PERIODS_HARD=5
        assert math.isnan(result.value)
        assert result.p_value is None or result.p_value >= 0.10

    def test_all_null_buckets_short_circuits_with_reason(self):
        """Raw date count clears the scaled floor, but every sampled date's
        bucket means are null (e.g. ``forward_return`` entirely missing), so
        there is nothing left to correlate. Must short-circuit with a reason
        rather than silently returning a degenerate NaN/t=0/p=1 triple.
        """
        from datetime import datetime, timedelta

        import polars as pl

        n_dates = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = [
            {"date": d, "asset_id": aid, "factor": float(j), "forward_return": None}
            for d in dates
            for j, aid in enumerate(["A", "B", "C", "D"])
        ]
        df = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Datetime("ms")),
            pl.col("forward_return").cast(pl.Float64),
        )
        result = monotonicity(df, overlap_periods=1, n_groups=2)["factor"]

        assert math.isnan(result.value)
        # The buckets could not be filled — an assets-axis failure, named as
        # one. (Here the cause is null returns rather than a thin universe,
        # but the unfillable-bucket reason is the same.)
        assert result.metadata["reason"] == "insufficient_assets_for_quantile_groups"
        assert result.n_obs_axis == "assets"


class TestSmallUniverse:
    """Default ``n_groups=10`` on an allocation-sized universe."""

    def _panel(self, n_assets: int):
        import factrix as fx
        from factrix.preprocess import compute_forward_return

        return compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=240, seed=7),
            forward_periods=5,
        )

    def test_names_the_assets_axis_not_periods(self):
        """T=240 is ample; the binding constraint is the 8-name cross-section,
        so the reason and the axis must say assets, not periods."""
        result = monotonicity(self._panel(8), overlap_periods=5)["factor"]
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_assets_for_quantile_groups"
        assert result.n_obs_axis == "assets"
        assert result.n_obs == 8
        assert result.metadata["min_required"] == 10

    def test_carries_a_warning_code(self):
        from factrix._codes import WarningCode

        result = monotonicity(self._panel(8), overlap_periods=5)["factor"]
        assert WarningCode.METRIC_UNAVAILABLE.value in result.warning_codes
        assert WarningCode.THIN_QUANTILE_GROUPS.value in result.warning_codes

    def test_declared_assets_floor_tracks_n_groups(self):
        from factrix.metrics import monotonicity as monotonicity_metric

        cls = type(monotonicity_metric(n_groups=3))
        assert (
            cls._resolve_sample_threshold(monotonicity_metric(n_groups=3)).min_assets
            == 3
        )
        assert cls._resolve_sample_threshold(monotonicity_metric()).min_assets == 10

    def test_downscaled_n_groups_runs_on_the_same_panel(self):
        result = monotonicity(self._panel(8), overlap_periods=5, n_groups=3)["factor"]
        assert not math.isnan(result.value)
        assert result.n_obs_axis == "periods"


class TestMonotonicityBatch:
    """Multi-factor path of ``monotonicity``."""

    def test_multi_factor_matches_list_of_one(self):
        from datetime import datetime, timedelta

        import numpy as np
        import polars as pl

        rng = np.random.default_rng(31)
        n_assets, n_dates = 80, 60
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for date in dates:
            returns = rng.standard_normal(n_assets) * 0.02
            f1 = returns + rng.standard_normal(n_assets) * 0.05
            f2 = -returns + rng.standard_normal(n_assets) * 0.05
            for asset_id in range(n_assets):
                rows.append(
                    {
                        "date": date,
                        "asset_id": asset_id,
                        "f1": float(f1[asset_id]),
                        "f2": float(f2[asset_id]),
                        "forward_return": float(returns[asset_id]),
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        batch = monotonicity(
            panel, overlap_periods=1, n_groups=5, factor_cols=["f1", "f2"]
        )
        for col in ("f1", "f2"):
            single = monotonicity(
                panel, overlap_periods=1, n_groups=5, factor_cols=[col]
            )[col]
            assert batch[col].value == pytest.approx(single.value)
            assert batch[col].stat == pytest.approx(single.stat)

    def test_empty_factor_list_rejected(self, noisy_panel):
        with pytest.raises(ValueError, match="non-empty"):
            monotonicity(noisy_panel, overlap_periods=1, n_groups=5, factor_cols=[])


class TestBatchTieRatio:
    """``_compute_tie_ratios_batch`` reports the per-period-then-median tie ratio."""

    def test_batch_matches_single_factor_per_date_median(self):
        from datetime import datetime, timedelta

        import polars as pl
        from factrix.metrics._helpers import _compute_tie_ratio
        from factrix.metrics.monotonicity import _compute_tie_ratios_batch

        # f_cont: continuous, unique within each date but the same value set
        # recurs across dates → per-period tie ratio 0. A global n_unique/len would
        # report ~1 here (spurious). f_bucket: 3 buckets → genuine per-period ties.
        n_assets, n_dates = 100, 40
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = [
            {
                "date": dt,
                "asset_id": a,
                "f_cont": float(a),
                "f_bucket": float(a % 3),
            }
            for dt in dates
            for a in range(n_assets)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

        batch = _compute_tie_ratios_batch(df, ["f_cont", "f_bucket"])
        assert batch["f_cont"] == pytest.approx(_compute_tie_ratio(df, "f_cont"))
        assert batch["f_bucket"] == pytest.approx(_compute_tie_ratio(df, "f_bucket"))
        # The continuous factor has no within-period ties — must not be flagged.
        assert batch["f_cont"] == pytest.approx(0.0)

    def test_empty_frame_returns_nan(self):
        import math

        import polars as pl
        from factrix.metrics.monotonicity import _compute_tie_ratios_batch

        empty = pl.DataFrame(
            {"date": [], "asset_id": [], "factor": []},
            schema={
                "date": pl.Datetime("ms"),
                "asset_id": pl.Int64,
                "factor": pl.Float64,
            },
        )
        assert math.isnan(_compute_tie_ratios_batch(empty, ["factor"])["factor"])


class TestPattonTimmermannMRTest:
    """The headline is the MR test, not a statistic with a null floor."""

    @staticmethod
    def _null_panel(n_dates=400, n_assets=100, seed=0):
        """Factor drawn independently of returns — H0 true by construction."""
        from datetime import datetime, timedelta

        import numpy as np
        import polars as pl

        rng = np.random.default_rng(seed)
        rows = n_dates * n_assets
        return pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1) + timedelta(days=d)
                    for d in range(n_dates)
                    for _ in range(n_assets)
                ],
                "asset_id": [f"A{i}" for _ in range(n_dates) for i in range(n_assets)],
                "factor": rng.standard_normal(rows),
                "forward_return": rng.standard_normal(rows) * 0.02,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    @pytest.mark.parametrize("n_groups", [3, 5, 10])
    def test_headline_no_longer_carries_an_n_groups_null_floor(self, n_groups):
        """mean|rho| read 0.66 / 0.43 / 0.27 at K = 3 / 5 / 10 under H0.

        E|rho| > 0 by Jensen, so the old headline was a noise floor a reader
        took for MR evidence, and the floor moved with n_groups. Measure the
        MR test's rejection frequency on panels where H0 holds by construction:
        it sits at or below nominal at every K, while the descriptive
        statistic's floor is still plainly there (in metadata, labelled as a
        shape statistic rather than as evidence).
        """
        n_reps = 20
        rejections = 0
        floors = []
        for rep in range(n_reps):
            panel = self._null_panel(n_dates=120, n_assets=40, seed=100 + rep)
            result = monotonicity(
                panel,
                overlap_periods=1,
                n_groups=n_groups,
                n_bootstrap=99,
                seed=rep,
            )["factor"]
            rejections += result.p_value < 0.05
            floors.append(result.metadata["mean_abs_spearman"])
        assert rejections / n_reps <= 0.15
        assert min(floors) > 0.15

    def test_mr_statistic_is_the_min_adjacent_difference(self):
        """Hand computation against the reported bucket means."""
        import numpy as np

        panel = self._null_panel(n_dates=60, n_assets=20, seed=3)
        result = monotonicity(
            panel, overlap_periods=1, n_groups=4, n_bootstrap=100, seed=2
        )["factor"]
        diffs = np.asarray(result.metadata["mr_adjacent_diffs"])
        assert len(diffs) == 3  # n_groups - 1 adjacent steps
        assert result.value == pytest.approx(float(diffs.min()))
        assert result.metadata["mr_min_diff"] == pytest.approx(result.value)

    def test_bootstrap_is_reproducible_and_reports_its_seed(self):
        panel = self._null_panel(n_dates=60, n_assets=20, seed=4)
        kwargs = {"overlap_periods": 1, "n_groups": 4, "n_bootstrap": 100}
        a = monotonicity(panel, seed=7, **kwargs)["factor"]
        b = monotonicity(panel, seed=7, **kwargs)["factor"]
        assert a.p_value == b.p_value
        assert a.metadata["bootstrap_seed"] == 7
        # An unseeded run resolves one and reports it, so the run stays
        # reproducible after the fact.
        c = monotonicity(panel, **kwargs)["factor"]
        assert isinstance(c.metadata["bootstrap_seed"], int)

    def test_monotone_signal_is_detected(self):
        from datetime import datetime, timedelta

        import numpy as np
        import polars as pl

        rng = np.random.default_rng(9)
        n_dates, n_assets = 200, 60
        rows = n_dates * n_assets
        factor = rng.standard_normal(rows)
        panel = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1) + timedelta(days=d)
                    for d in range(n_dates)
                    for _ in range(n_assets)
                ],
                "asset_id": [f"A{i}" for _ in range(n_dates) for i in range(n_assets)],
                "factor": factor,
                "forward_return": 0.01 * factor + 0.005 * rng.standard_normal(rows),
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = monotonicity(
            panel, overlap_periods=1, n_groups=5, n_bootstrap=200, seed=5
        )["factor"]
        assert result.value > 0
        assert result.p_value < 0.05


class TestTieRatioCountsFiniteValuesOnly:
    def test_nulls_are_not_counted_as_a_tied_level(self):
        """5 of 10 names null, the other 5 all distinct: the tie ratio is 0."""
        from datetime import datetime, timedelta

        import polars as pl

        n_dates = 40
        rows = []
        for d in range(n_dates):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for i in range(10):
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{i}",
                        "factor": None if i >= 5 else float(i) + d * 0.01,
                        "forward_return": 0.001 * i,
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = monotonicity(
            panel, overlap_periods=1, n_groups=3, n_bootstrap=50, seed=0
        )["factor"]
        assert result.metadata["tie_ratio"] == pytest.approx(0.0)

    def test_nan_is_not_counted_either(self):
        from datetime import datetime, timedelta

        import polars as pl

        n_dates = 40
        rows = []
        for d in range(n_dates):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for i in range(10):
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{i}",
                        "factor": float("nan") if i >= 5 else float(i) + d * 0.01,
                        "forward_return": 0.001 * i,
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = monotonicity(
            panel, overlap_periods=1, n_groups=3, n_bootstrap=50, seed=0
        )["factor"]
        assert result.metadata["tie_ratio"] == pytest.approx(0.0)

    def test_real_ties_are_still_reported(self):
        from datetime import datetime, timedelta

        import polars as pl

        n_dates = 40
        rows = []
        for d in range(n_dates):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for i in range(10):
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{i}",
                        "factor": float(i // 5),  # two distinct levels
                        "forward_return": 0.001 * i,
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        with pytest.warns(UserWarning, match="tie_ratio"):
            result = monotonicity(
                panel, overlap_periods=1, n_groups=3, n_bootstrap=50, seed=0
            )["factor"]
        assert result.metadata["tie_ratio"] == pytest.approx(0.8)


class TestMRArgumentValidation:
    @staticmethod
    def _panel():
        return fx.preprocess.compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=40, n_dates=120, seed=3),
            forward_periods=1,
        )

    def test_direction_typo_is_rejected(self):
        from factrix._errors import UserInputError
        from factrix.metrics.monotonicity import monotonicity

        with pytest.raises(UserInputError, match="direction"):
            monotonicity(self._panel(), n_bootstrap=50, seed=0, direction="decrease")

    def test_non_positive_n_bootstrap_is_rejected(self):
        from factrix._errors import UserInputError
        from factrix.metrics.monotonicity import monotonicity

        with pytest.raises(UserInputError, match="n_bootstrap"):
            monotonicity(self._panel(), n_bootstrap=0, seed=0)
