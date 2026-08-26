"""Tests for factrix.metrics.monotonicity."""

import math

import pytest
from factrix.metrics.monotonicity import monotonicity


class TestComputeMonotonicity:
    def test_perfect_monotonic(self, noisy_panel):
        # WHY: tiny_panel only has 3 dates, < MIN_MONOTONICITY_PERIODS_HARD after sampling
        # Use noisy_panel (20 dates × 30 assets) with perfect factor-return alignment
        import polars as pl

        perfect = noisy_panel.with_columns(
            pl.col("factor").rank(method="average").over("date").alias("forward_return")
        )
        result = monotonicity(perfect, forward_periods=1, n_groups=5)["factor"]
        assert result.value == pytest.approx(1.0)

    def test_perfect_monotonicity_keeps_its_value_and_withholds_the_test(
        self, noisy_panel
    ):
        """+1 on every period is maximal evidence, and has no t.

        The measurement (avg monotonicity = 1.0) survives; the hypothesis
        test does not, because a constant series has no dispersion to form
        an SE from. What must never appear is the old t=0 / p=1, which read
        a perfectly ordered factor as "no signal".
        """
        import polars as pl
        from factrix._codes import WarningCode

        perfect = noisy_panel.with_columns(
            pl.col("factor").rank(method="average").over("date").alias("forward_return")
        )
        result = monotonicity(perfect, forward_periods=1, n_groups=5)["factor"]
        assert result.value == pytest.approx(1.0)
        assert result.metadata["mean_signed"] == pytest.approx(1.0)
        assert result.stat is None
        assert result.p_value is None
        assert result.alternative is None
        assert WarningCode.DEGENERATE_VARIANCE.value in result.warning_codes
        assert result.metadata["signal_status"] == "degenerate_zero_variance"

    def test_inverse_monotonic(self, noisy_panel):
        import polars as pl

        inverted = noisy_panel.with_columns(
            (-pl.col("factor").rank(method="average").over("date")).alias(
                "forward_return"
            )
        )
        result = monotonicity(inverted, forward_periods=1, n_groups=5)["factor"]
        assert result.value == pytest.approx(1.0)
        assert result.metadata["mean_signed"] == pytest.approx(-1.0)

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
        result = monotonicity(df, forward_periods=1, n_groups=5)["factor"]
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
        result = monotonicity(df, forward_periods=1, n_groups=2)["factor"]

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
        result = monotonicity(self._panel(8), forward_periods=5)["factor"]
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_assets_for_quantile_groups"
        assert result.n_obs_axis == "assets"
        assert result.n_obs == 8
        assert result.metadata["min_required"] == 10

    def test_carries_a_warning_code(self):
        from factrix._codes import WarningCode

        result = monotonicity(self._panel(8), forward_periods=5)["factor"]
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
        result = monotonicity(self._panel(8), forward_periods=5, n_groups=3)["factor"]
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
            panel, forward_periods=1, n_groups=5, factor_cols=["f1", "f2"]
        )
        for col in ("f1", "f2"):
            single = monotonicity(
                panel, forward_periods=1, n_groups=5, factor_cols=[col]
            )[col]
            assert batch[col].value == pytest.approx(single.value)
            assert batch[col].stat == pytest.approx(single.stat)

    def test_empty_factor_list_rejected(self, noisy_panel):
        with pytest.raises(ValueError, match="non-empty"):
            monotonicity(noisy_panel, forward_periods=1, n_groups=5, factor_cols=[])


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
