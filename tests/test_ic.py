"""Tests for factrix.metrics.ic."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.ic import compute_ic, ic, ic_ir


class TestComputeIC:
    def test_perfect_rank(self, tiny_panel):
        result = compute_ic(tiny_panel)["factor"]
        # factor and return have identical ranking → IC = 1.0
        for ic_val in result["ic"].to_list():
            assert ic_val == pytest.approx(1.0)

    def test_inverse_rank(self, tiny_panel):
        # Reverse the returns
        inverted = tiny_panel.with_columns(
            (0.06 - pl.col("forward_return")).alias("forward_return")
        )
        result = compute_ic(inverted)["factor"]
        for ic_val in result["ic"].to_list():
            assert ic_val == pytest.approx(-1.0)

    def test_drops_small_dates(self):
        """Dates with < MIN_IC_ASSETS_HARD assets should be excluded."""
        # 1 asset < MIN_IC_ASSETS_HARD=2.
        dates = [datetime(2024, 1, 1)]
        rows = [
            {
                "date": dates[0],
                "asset_id": f"A{i}",
                "factor": float(i),
                "forward_return": float(i) * 0.01,
            }
            for i in range(1)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_ic(df)["factor"]
        assert len(result) == 0

    def test_gate_counts_valid_pairs_not_rows(self):
        """The cross-section floor gates on the per-date *valid-pair* count, not
        the raw row count: a date with many names but only one valid pair must
        be dropped because its IC is undefined.
        """
        n = 30
        factor = [0.0 if i == 0 else None for i in range(n)]  # 1 valid < hard floor
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * n,
                "asset_id": [f"A{i}" for i in range(n)],
                "factor": factor,
                "forward_return": [float(i) * 0.01 for i in range(n)],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        # 1 valid pair < MIN_IC_ASSETS_HARD=2: the date is dropped even though
        # the raw row count is 30.
        assert len(compute_ic(df)["factor"]) == 0

    def test_warn_band_dates_are_retained_with_n_assets(self):
        """Dates in [MIN_IC_ASSETS_HARD, MIN_IC_ASSETS_WARN) are computable."""
        n = 30
        factor = [float(i) if i < 6 else None for i in range(n)]
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * n,
                "asset_id": [f"A{i}" for i in range(n)],
                "factor": factor,
                "forward_return": [float(i) * 0.01 for i in range(n)],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_ic(df)["factor"]
        assert result.height == 1
        assert result["n_assets"][0] == 6

    def test_tie_ratio_denominated_by_valid_pairs(self):
        """tie_ratio uses the non-null factor count as denominator, so null
        names neither inflate the denominator nor count as a tie category.
        """
        n = 16
        # 12 valid names, 3 unique factor values → tie_ratio = 1 - 3/12 = 0.75.
        valid_vals = [float(i % 3) for i in range(12)]
        factor = valid_vals + [None] * 4
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * n,
                "asset_id": [f"A{i}" for i in range(n)],
                "factor": factor,
                "forward_return": [float(i) * 0.01 for i in range(n)],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_ic(df)["factor"]
        assert result["tie_ratio"][0] == pytest.approx(0.75)

    def test_null_return_does_not_distort_other_ranks(self):
        """A null in one column must not shift the surviving assets' ranks in
        the other: Spearman ρ is computed on the pairwise-complete set, so the
        result matches scipy's spearmanr over the non-null pairs rather than
        ranking the raw columns and dropping null pairs only afterward.
        """
        from scipy.stats import spearmanr

        n = 12
        factor = [float(i) for i in range(1, n + 1)]
        ret = [0.5, -0.2, 0.9, 0.1, -0.4, None, 0.3, 0.95, -0.7, -1.2, -0.6, 0.04]
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * n,
                "asset_id": [f"A{i}" for i in range(n)],
                "factor": factor,
                "forward_return": ret,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

        got = compute_ic(df)["factor"]["ic"][0]

        f = np.array(factor)
        r = np.array([np.nan if x is None else x for x in ret])
        mask = ~np.isnan(r)
        expected, _ = spearmanr(f[mask], r[mask])
        assert got == pytest.approx(expected)

    def test_output_schema(self, noisy_panel):
        result = compute_ic(noisy_panel)["factor"]
        # ``_drop_stats`` is an internal diagnostic struct column appended by
        # the primitive; ``n_assets`` carries the per-date valid-pair count.
        assert result.columns == ["date", "ic", "tie_ratio", "n_assets", "_drop_stats"]
        assert result["date"].dtype == pl.Datetime("ms")

    def test_tie_ratio_zero_on_unique_factor(self, noisy_panel):
        result = compute_ic(noisy_panel)["factor"]
        # noisy_panel factor is continuous noise — no per-date ties expected.
        assert result["tie_ratio"].max() == pytest.approx(0.0)

    def test_tie_ratio_detects_bucketed_factor(self):
        """Bucketed factor → tie_ratio surfaces non-trivially per date."""
        n_assets = 12
        dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(5)]
        rows = [
            {
                "date": dt,
                "asset_id": f"A{i}",
                "factor": float(i % 3),  # 3 buckets → ties
                "forward_return": float(i) * 0.01,
            }
            for dt in dates
            for i in range(n_assets)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_ic(df)["factor"]
        # 12 obs, 3 unique → tie_ratio = 1 - 3/12 = 0.75
        assert result["tie_ratio"].max() == pytest.approx(0.75)
        assert result["tie_ratio"].min() == pytest.approx(0.75)

    def test_tie_ratio_propagated_to_metadata(self, noisy_panel):
        from factrix.inference import NEWEY_WEST
        from factrix.metrics.ic import ic, ic_ir

        ic_df = compute_ic(noisy_panel)["factor"]
        for out in (
            ic(ic_df, forward_periods=1),
            ic(ic_df, forward_periods=1, inference=NEWEY_WEST),
            ic_ir(ic_df),
        ):
            assert "tie_ratio" in out.metadata
            assert 0.0 <= out.metadata["tie_ratio"] <= 1.0

    def test_high_tie_ratio_emits_warning(self):
        """ic / ic(NEWEY_WEST) / ic_ir warn when median tie_ratio > threshold."""
        import warnings

        from factrix.inference import NEWEY_WEST
        from factrix.metrics.ic import ic, ic_ir

        # 12 assets bucketed into 2 buckets per date → tie_ratio = 1 - 2/12
        # ≈ 0.83 (well above the 0.3 threshold).
        n_assets = 12
        dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(40)]
        rows = [
            {
                "date": dt,
                "asset_id": f"A{i}",
                "factor": float(i % 2),
                "forward_return": float(i) * 0.01,
            }
            for dt in dates
            for i in range(n_assets)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        ic_df = compute_ic(df)["factor"]
        for fn in (
            lambda d: ic(d, forward_periods=1),
            lambda d: ic(d, forward_periods=1, inference=NEWEY_WEST),
            ic_ir,
        ):
            with pytest.warns(UserWarning, match="tie_ratio"):
                fn(ic_df)

        # Low-tie panel must not trigger the warning.
        rng = np.random.default_rng(0)
        clean_rows = [
            {
                "date": dt,
                "asset_id": f"A{i}",
                "factor": float(rng.standard_normal()),
                "forward_return": float(rng.standard_normal()) * 0.01,
            }
            for dt in dates
            for i in range(n_assets)
        ]
        clean = pl.DataFrame(clean_rows).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        clean_ic = compute_ic(clean)["factor"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ic(clean_ic, forward_periods=1)
            ic(clean_ic, forward_periods=1, inference=NEWEY_WEST)
            ic_ir(clean_ic)


class TestComputeICBatch:
    """Multi-factor path of ``compute_ic``.

    Each element of a multi-factor batch must equal the corresponding
    list-of-1 call — divergence means a silent numerics regression for
    callers that consume one factor at a time from a shared panel.
    """

    def test_multi_factor_columns_match_list_of_one(self):
        rng = np.random.default_rng(7)
        n_assets, n_dates = 60, 40
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        records = []
        for date in dates:
            returns = rng.standard_normal(n_assets) * 0.02
            f1 = returns + rng.standard_normal(n_assets) * 0.05
            f2 = -returns + rng.standard_normal(n_assets) * 0.05
            for asset_id in range(n_assets):
                records.append(
                    {
                        "date": date,
                        "asset_id": asset_id,
                        "f1": float(f1[asset_id]),
                        "f2": float(f2[asset_id]),
                        "forward_return": float(returns[asset_id]),
                    }
                )
        panel = pl.DataFrame(records).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        batch = compute_ic(panel, factor_cols=["f1", "f2"])
        for col in ("f1", "f2"):
            single = compute_ic(panel, factor_cols=[col])[col]
            assert batch[col].equals(single), col

    def test_empty_factor_list_rejected(self, tiny_panel):
        with pytest.raises(ValueError, match="non-empty"):
            compute_ic(tiny_panel, factor_cols=[])


class TestIC:
    def test_positive_ic(self, noisy_panel):
        ic_df = compute_ic(noisy_panel)["factor"]
        result = ic(ic_df, forward_periods=1)
        assert result.value > 0  # noisy_panel has positive IC
        assert result.stat > 0
        assert result.p_value < 0.10

    def test_insufficient_periods(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
                "ic": [0.05, 0.03, 0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = ic(df, forward_periods=1)
        assert math.isnan(result.value)
        # Genuine date shortfall (no compute_ic carrier) → periods-axis reason.
        assert result.metadata["reason"] == "insufficient_ic_periods"

    def test_few_assets_warns_without_blocking(self):
        # 8 assets clears MIN_IC_ASSETS_HARD=2 but falls below
        # MIN_IC_ASSETS_WARN=10: the IC is computable and returned with an
        # asset-axis warning instead of a hard short-circuit.
        import factrix as fx
        from factrix._codes import WarningCode

        raw = fx.datasets.make_cs_panel(n_assets=8, n_dates=120, seed=0)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        with pytest.warns(UserWarning, match="MIN_IC_ASSETS_WARN"):
            result = ic(compute_ic(panel)["factor"], forward_periods=5)
        assert not math.isnan(result.value)
        assert WarningCode.FEW_ASSETS.value in result.warning_codes
        assert result.metadata["min_assets_per_period"] == 8
        assert result.metadata["warn_assets_per_period"] == 10


class TestICIR:
    def test_positive_ir(self, noisy_panel):
        ic_df = compute_ic(noisy_panel)["factor"]
        result = ic_ir(ic_df)
        assert result.value > 0
        assert result.stat is None
        assert "mean_ic" in result.metadata

    def test_insufficient_periods(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
                "ic": [0.05, 0.03, 0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = ic_ir(df)
        assert math.isnan(result.value)

    @staticmethod
    def _ic_series(n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
                "ic": [0.05 + 0.01 * (i % 3) for i in range(n)],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_warn_below_warn_periods(self):
        # 25 periods clears the HARD floor (MIN_PERIODS_HARD=20) but sits below
        # the WARN floor (MIN_PERIODS_WARN=30): a value is returned with the
        # degraded-tier warning attached.
        from factrix._codes import WarningCode

        with pytest.warns(UserWarning, match="below MIN_PERIODS_WARN"):
            result = ic_ir(self._ic_series(25))
        assert not math.isnan(result.value)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in result.warning_codes

    def test_no_warn_at_or_above_warn_periods(self):
        from factrix._codes import WarningCode

        result = ic_ir(self._ic_series(40))
        assert not math.isnan(result.value)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value not in result.warning_codes


class TestICInferenceWarningPropagation:
    """``ic()`` must surface the inference method's own thin-sample warning
    (``UNRELIABLE_SE_SHORT_PERIODS``) on the returned result — previously the
    ``InferenceResult.warnings`` were consumed for the stat but never folded
    into ``MetricResult.warning_codes``.
    """

    @staticmethod
    def _ic_series(n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
                "ic": [0.05 + 0.01 * (i % 3) for i in range(n)],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_thin_non_overlap_surfaces_warning(self):
        from factrix._codes import WarningCode

        # forward_periods=1 => n_sampled == n == 25: clears MIN_SERIES_PERIODS_HARD (10)
        # so no short-circuit, but below MIN_PERIODS_WARN (30) so the
        # non-overlapping inference flags UNRELIABLE_SE_SHORT_PERIODS.
        result = ic(self._ic_series(25), forward_periods=1)
        assert not math.isnan(result.value)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in result.warning_codes

    def test_ample_sample_no_warning(self):
        from factrix._codes import WarningCode

        result = ic(self._ic_series(40), forward_periods=1)
        assert not math.isnan(result.value)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value not in result.warning_codes

    def test_boundary_at_warn_floor(self):
        from factrix._codes import WarningCode

        # n_sampled == MIN_PERIODS_WARN (30) is the first clean count (strict <).
        clean = ic(self._ic_series(30), forward_periods=1)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value not in clean.warning_codes
        # One observation below the floor still warns.
        warned = ic(self._ic_series(29), forward_periods=1)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in warned.warning_codes

    def test_surfaced_codes_are_deduplicated(self):
        result = ic(self._ic_series(25), forward_periods=1)
        assert len(result.warning_codes) == len(set(result.warning_codes))


class TestICInferenceAllowlist:
    """``ic`` validates ``inference=`` against ``applicable_inference`` and
    rejects anything outside it — ``HansenHodrick`` (which would otherwise
    run a non-vetted HAC) and non-``Inference`` objects (which would hit an
    ``AttributeError`` on dispatch).
    """

    @staticmethod
    def _ic_series(n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
                "ic": [0.05 + 0.01 * (i % 3) for i in range(n)],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_hansen_hodrick_raises_not_silently_runs(self):
        import factrix as fx

        with pytest.raises(fx.IncompatibleInferenceError) as exc:
            ic(
                self._ic_series(40),
                forward_periods=1,
                inference=fx.inference.HANSEN_HODRICK,
            )
        assert exc.value.func_name == "ic"
        assert exc.value.applicable == (
            "NeweyWest",
            "NonOverlapping",
            "StationaryBootstrap",
        )
        assert "HansenHodrick" in str(exc.value)

    def test_non_inference_object_raises_cleanly(self):
        import factrix as fx

        with pytest.raises(fx.IncompatibleInferenceError):
            ic(self._ic_series(40), forward_periods=1, inference="newey")

    def test_allowlisted_members_pass(self):
        import factrix as fx

        for member in (
            fx.inference.NON_OVERLAPPING,
            fx.inference.NEWEY_WEST,
            fx.inference.STATIONARY_BOOTSTRAP,
        ):
            result = ic(self._ic_series(40), forward_periods=1, inference=member)
            assert not math.isnan(result.value)


class TestApplicableInferenceDiscovery:
    """``resolve_applicable_inference`` surfaces a metric's allowlist, and
    returns ``None`` for singleton-inference metrics — even when they share
    a module with an ``inference=``-bearing sibling.
    """

    def test_inference_metrics_expose_allowlist(self):
        from factrix.metrics._metric_capabilities import resolve_applicable_inference
        from factrix.metrics.k_spread import k_spread
        from factrix.metrics.quantile import quantile_spread

        # quantile_spread / k_spread dispatch through a hard isinstance(NeweyWest)
        # branch, so their allowlist stays the original vetted pair.
        for m in (quantile_spread, k_spread):
            allow = resolve_applicable_inference(m)
            assert allow is not None
            assert sorted(type(x).__name__ for x in allow) == [
                "NeweyWest",
                "NonOverlapping",
            ]

        # ic dispatches polymorphically, so its allowlist additionally admits
        # StationaryBootstrap.
        ic_allow = resolve_applicable_inference(ic)
        assert ic_allow is not None
        assert sorted(type(x).__name__ for x in ic_allow) == [
            "NeweyWest",
            "NonOverlapping",
            "StationaryBootstrap",
        ]

    def test_singleton_inference_metric_returns_none(self):
        from factrix.metrics._metric_capabilities import resolve_applicable_inference
        from factrix.metrics.positive_rate import positive_rate
        from factrix.metrics.quantile import quantile_spread_vw

        # quantile_spread_vw shares its module with quantile_spread but has no
        # inference= knob; positive_rate is in a module with no allowlist at all.
        assert resolve_applicable_inference(quantile_spread_vw) is None
        assert resolve_applicable_inference(positive_rate) is None


class TestICDispatch:
    """``ic()`` delegates its significance test to ``NonOverlappingSample``.

    These pin the result so the dispatch refactor stays numerically
    equivalent to the previously-inlined non-overlapping OLS t-test
    (to floating-point ULP — polars vs numpy variance reduction order).
    """

    def _ic_df(self):
        import factrix as fx
        from factrix.preprocess import compute_forward_return

        panel = compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
            forward_periods=5,
        )
        return compute_ic(panel)["factor"]

    @pytest.mark.parametrize(
        ("forward_periods", "value", "stat", "p_value"),
        [
            (1, 0.049163258267725024, 5.404723194889399, 2.1264376169332541e-07),
            (5, 0.049163258267725024, 4.136962357146272, 0.00021830177834358518),
        ],
    )
    def test_regression_pins_dispatch_output(
        self, forward_periods, value, stat, p_value
    ):
        result = ic(self._ic_df(), forward_periods=forward_periods)
        assert result.value == pytest.approx(value, rel=1e-12)
        assert result.stat == pytest.approx(stat, rel=1e-12)
        assert result.p_value == pytest.approx(p_value, rel=1e-12)
        assert result.metadata["method"] == "non-overlapping t-test"
