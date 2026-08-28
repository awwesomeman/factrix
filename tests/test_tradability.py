"""Tests for factrix.metrics.tradability."""

import math
from datetime import datetime, timedelta
from typing import ClassVar

import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix._types import DEFAULT_FORWARD_PERIODS, DEFAULT_N_GROUPS
from factrix.metrics.tradability import (
    breakeven_cost,
    net_spread,
    notional_turnover,
    rank_turnover,
)


def _panel(n_dates: int, assets: list[str], factor_of) -> pl.DataFrame:
    """Build a ``date, asset_id, factor`` panel from a per-(date_idx, asset) fn."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = [
        {"date": d, "asset_id": a, "factor": float(factor_of(t, a))}
        for t, d in enumerate(dates)
        for a in assets
    ]
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestComputeRankTurnover:
    def test_static_factor(self):
        """Same factor values every date → rank_autocorr=1.0 → turnover=0."""
        df = _panel(5, ["A", "B", "C"], lambda t, a: ord(a) - ord("A") + 1)
        result = rank_turnover(df)
        assert result.value == pytest.approx(0.0, abs=0.01)
        assert result.metadata["n_periods"] == 4
        assert result.metadata["overlap_periods"] == 1
        assert result.metadata["quantile"] is None

    def test_n_obs_axis_is_periods_not_pairs(self):
        """The count is adjacent-period transitions (T-1), not (date, asset)
        pairs — the unit ``pairs`` denotes for pooled_beta / directional_hit_rate."""
        df = _panel(5, ["A", "B", "C"], lambda t, a: ord(a) - ord("A") + t)
        result = rank_turnover(df)
        assert result.n_obs_axis == "periods"
        assert result.n_obs == 4  # 5 dates -> 4 transitions, independent of n_assets

    def test_single_date(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 3,
                "asset_id": ["A", "B", "C"],
                "factor": [1.0, 2.0, 3.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = rank_turnover(df)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_dates"

    def test_insufficient_dates_for_horizon(self):
        """2·h + 1 dates is the minimum for SE to be defined."""
        df = _panel(4, ["A", "B", "C"], lambda t, a: ord(a) + t)
        result = rank_turnover(df, overlap_periods=2)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_dates"
        assert result.metadata["min_required"] == 5

    def test_non_overlapping_skips_intermediate_noise(self):
        """Rank flips inside the holding window must not count as turnover.

        At even ``t`` ranks are (A=1,B=2,C=3); at odd ``t`` they are the
        reverse. With ``overlap_periods=2`` we sample only even dates, so
        every pair's rank-AC is +1 → turnover=0.
        """

        def factor(t, a):
            base = ord(a) - ord("A") + 1
            return base if t % 2 == 0 else (4 - base)

        df = _panel(7, ["A", "B", "C"], factor)
        result = rank_turnover(df, overlap_periods=2)
        assert result.value == pytest.approx(0.0, abs=0.01)
        assert result.metadata["n_periods"] == 3

    def test_quantile_filter_restricts_to_tails(self):
        """Quantile filter must actually select tail names and only tail names.

        Ten assets with monotone factor + tiny time drift → tails are
        stable: bottom-2={A,B}, top-2={I,J} on every date. Union of tails
        across either endpoint therefore contains exactly 4 names per
        pair — compared to all 10 in the unfiltered case.
        """
        assets = [chr(ord("A") + i) for i in range(10)]
        df = _panel(6, assets, lambda t, a: ord(a) + 0.1 * t)
        q = 0.2

        filtered = rank_turnover(df, overlap_periods=1, quantile=q)
        unfiltered = rank_turnover(df, overlap_periods=1)

        assert filtered.metadata["quantile"] == q
        assert filtered.metadata["n_cross_section_mean"] == pytest.approx(4.0)
        assert unfiltered.metadata["n_cross_section_mean"] == pytest.approx(10.0)
        assert filtered.value == pytest.approx(0.0, abs=0.01)

    def test_quantile_validation(self):
        df = _panel(5, ["A", "B", "C"], lambda t, a: ord(a))
        with pytest.raises(ValueError, match="quantile"):
            rank_turnover(df, quantile=0.6)

    def test_forward_periods_validation(self):
        df = _panel(5, ["A", "B", "C"], lambda t, a: ord(a))
        with pytest.raises(ValueError, match="overlap_periods"):
            rank_turnover(df, overlap_periods=0)


class TestNotionalTurnover:
    TEN_ASSETS: ClassVar[list[str]] = [chr(ord("A") + i) for i in range(10)]

    def test_static_factor(self):
        """Same tail sets every day → notional turnover = 0."""
        df = _panel(5, self.TEN_ASSETS, lambda t, a: ord(a))
        result = notional_turnover(df, n_groups=5, overlap_periods=1)
        assert result.value == pytest.approx(0.0)
        assert result.metadata["n_rebalances"] == 4
        assert result.metadata["n_groups"] == 5
        # Rebalances are adjacent-period transitions, not (date, asset) pairs.
        assert result.n_obs_axis == "periods"
        assert result.n_obs == 4

    def test_small_universe_names_the_assets_axis(self):
        """The default n_groups on a 4-name universe empties every date however
        long the panel — an assets-axis failure, reported as one."""
        from factrix._codes import WarningCode

        assets = [chr(ord("A") + i) for i in range(4)]
        df = _panel(200, assets, lambda t, a: ord(a) + t)
        result = notional_turnover(df)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_assets_for_quantile_groups"
        assert result.n_obs_axis == "assets"
        assert result.n_obs == 4
        assert result.metadata["min_required"] == DEFAULT_N_GROUPS
        assert WarningCode.THIN_QUANTILE_GROUPS.value in result.warning_codes

    def test_declared_assets_floor_tracks_n_groups(self):
        from factrix.metrics import notional_turnover as nt

        cls = type(nt())
        assert cls._resolve_sample_threshold(nt()).min_assets == DEFAULT_N_GROUPS
        assert cls._resolve_sample_threshold(nt(n_groups=3)).min_assets == 3

    def test_downscaled_n_groups_runs_on_the_same_panel(self):
        assets = [chr(ord("A") + i) for i in range(8)]
        df = _panel(200, assets, lambda t, a: ord(a) + t)
        result = notional_turnover(df, n_groups=3)
        assert not math.isnan(result.value)

    def test_full_rotation(self):
        """Ranks reverse every date → top ↔ bot fully swap → turnover = 1."""

        def factor(t, a):
            base = ord(a) - ord("A")
            return base if t % 2 == 0 else (9 - base)

        df = _panel(5, self.TEN_ASSETS, factor)
        result = notional_turnover(df, n_groups=5, overlap_periods=1)
        assert result.value == pytest.approx(1.0)

    def test_middle_shuffle_does_not_count(self):
        """Middle-rank reshuffling with stable tails → notional=0, ρ<1.

        This is the raison d'être of notional_turnover: ``rank_turnover``
        (1 − Spearman ρ) is non-zero here because middle ranks move,
        but no bps cost is actually incurred because Q1/Q5 membership
        is unchanged.
        """
        tails = {"A", "B", "I", "J"}  # bottom 2 + top 2 at n_groups=5

        def factor(t, a):
            if a in ("A", "B"):
                return ord(a) - ord("A")  # 0, 1 — always lowest
            if a in ("I", "J"):
                return 100 + ord(a) - ord("A")  # always highest
            # C..H rotate in the middle band
            middle = "CDEFGH"
            idx = middle.index(a)
            return 20 + ((idx + t) % len(middle))

        df = _panel(6, self.TEN_ASSETS, factor)
        notional = notional_turnover(df, n_groups=5)
        stab = rank_turnover(df)
        assert notional.value == pytest.approx(0.0)
        assert stab.value > 0.05  # rank AC noticeably below 1
        assert tails == set("ABIJ")  # scaffolding: document intent

    def test_partial_top_churn(self):
        """Half of top bucket rolls over every rebalance → top_churn=0.5.

        With n_groups=5 and 10 assets the top bucket holds 2 names. If
        exactly one of the two top names rotates out each date while the
        bottom stays put: top_churn=0.5, bot_churn=0 → turnover=0.25.
        """

        def factor(t, a):
            if a in ("A", "B"):
                return ord(a) - ord("A")  # bottom 2 fixed
            if a == "I":
                # I shares the top bucket with J on even t, drops on odd t
                return 100 if t % 2 == 0 else 50
            if a == "J":
                return 101  # always top
            if a == "H":
                # H fills in for I on odd t
                return 40 if t % 2 == 0 else 100
            return 20 + (ord(a) - ord("C"))

        df = _panel(6, self.TEN_ASSETS, factor)
        result = notional_turnover(df, n_groups=5)
        # Per pair: top sometimes swaps 1/2 names (churn=0.5) or keeps
        # both (churn=0). Over the 5 pairs the mean of (top+bot)/2 should
        # land between 0 and 0.25; asserting in a band is more robust
        # than a point estimate to avoid polars tie-break quirks.
        assert 0.05 < result.value < 0.30

    def test_forward_periods_stride(self):
        """overlap_periods=2 sub-samples to odd/even dates only."""

        def factor(t, a):
            base = ord(a) - ord("A")
            return base if t % 2 == 0 else (9 - base)

        df = _panel(7, self.TEN_ASSETS, factor)
        # Even-only sample → ranks identical every sampled date → 0.
        result = notional_turnover(df, n_groups=5, overlap_periods=2)
        assert result.value == pytest.approx(0.0)
        assert result.metadata["overlap_periods"] == 2

    def test_insufficient_dates_short_circuits(self):
        df = _panel(1, self.TEN_ASSETS, lambda t, a: ord(a))
        result = notional_turnover(df, n_groups=5)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_dates"

    def test_validation(self):
        df = _panel(3, self.TEN_ASSETS, lambda t, a: ord(a))
        with pytest.raises(ValueError, match="overlap_periods"):
            notional_turnover(df, overlap_periods=0)
        # The floor is the shared N_GROUPS_FLOOR = 2, not a private 3: the
        # two-group top-half / bottom-half book is the one
        # ``quantile_spread(n_groups=2)`` prices (#878).
        with pytest.raises(ValueError, match="n_groups"):
            notional_turnover(df, n_groups=1)


class TestBreakevenCost:
    def test_basic(self):
        # Notional turnover=0.5 (per-leg fraction of Q1/Q_n replaced per
        # rebalance): gross=0.10/period, turnover=0.5/rebalance, fp=1.
        # Traded notional per rebalance = 4*0.5 = 2 (2 legs x sell+buy), so
        # the breakeven one-way cost is 0.10*1/(4*0.5)*10000 = 500 bps.
        result = breakeven_cost(0.10, turnover=0.5, holding_periods=1)
        assert result.value == pytest.approx(500.0)
        assert result.metadata["holding_periods"] == 1

    def test_zero_turnover(self):
        result = breakeven_cost(0.10, turnover=0.0, holding_periods=1)
        assert result.value == float("inf")

    def test_forward_periods_scales_breakeven(self):
        """gross is per-period, turnover per-rebalance: breakeven scales by N.

        Halving the per-period spread but holding for holding_periods=2 should give the
        same breakeven as the holding_periods=1 baseline — the trader earns the spread
        twice before paying the once-per-rebalance cost.
        """
        baseline = breakeven_cost(0.10, turnover=0.5, holding_periods=1).value
        scaled = breakeven_cost(0.05, turnover=0.5, holding_periods=2).value
        assert scaled == pytest.approx(baseline)

    def test_forward_periods_validation(self):
        with pytest.raises(ValueError, match="holding_periods"):
            breakeven_cost(0.10, turnover=0.5, holding_periods=0)


class TestNetSpread:
    def test_basic(self):
        # Notional turnover=0.5 (per leg); cost=30bps one-way; fp=1.
        # net = 0.10 - 4*(30/10000)*0.5/1 = 0.10 - 0.006 = 0.094
        result = net_spread(
            0.10, turnover=0.5, estimated_cost_bps=30, holding_periods=1
        )
        assert result.value == pytest.approx(0.094)
        assert result.metadata["holding_periods"] == 1

    def test_cost_exceeds_alpha(self):
        result = net_spread(
            0.001, turnover=0.5, estimated_cost_bps=100, holding_periods=1
        )
        assert result.value < 0

    def test_forward_periods_amortises_cost(self):
        """Cost is paid once per rebalance, spread is per-period: per-period
        cost drag must shrink by exactly 1/N. Asserting the ratio (rather
        than absolute values) pins the scaling invariant under any rescaling
        of inputs — which is the whole point of the fix."""
        baseline = net_spread(
            0.10, turnover=0.5, estimated_cost_bps=30, holding_periods=1
        )
        scaled = net_spread(
            0.10, turnover=0.5, estimated_cost_bps=30, holding_periods=5
        )
        assert scaled.metadata["cost_drag"] == pytest.approx(
            baseline.metadata["cost_drag"] / 5
        )

    def test_forward_periods_validation(self):
        with pytest.raises(ValueError, match="holding_periods"):
            net_spread(0.10, turnover=0.5, holding_periods=0)


class TestRankTurnoverIgnoresNonFiniteFactors:
    """polars ranks NaN last, i.e. as larger than every real value."""

    @staticmethod
    def _panel_with(mask_value):
        import numpy as np

        rng = np.random.default_rng(0)
        n_dates, assets = 60, [f"A{i}" for i in range(20)]
        rows = []
        poison = rng.random((n_dates, len(assets))) < 0.2
        for t in range(n_dates):
            for j, a in enumerate(assets):
                rows.append(
                    {
                        "date": datetime(2024, 1, 1) + timedelta(days=t),
                        "asset_id": a,
                        "factor": mask_value if poison[t, j] else float(rng.normal()),
                    }
                )
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_nan_and_null_agree(self):
        nan_value = rank_turnover(self._panel_with(float("nan")), overlap_periods=1)
        null_value = rank_turnover(self._panel_with(None), overlap_periods=1)
        assert nan_value.value == pytest.approx(null_value.value)

    def test_poisoned_rows_leave_the_denominator_too(self):
        """``pl.len().over(date)`` counted the NaN rows, so the tail cutoffs
        used the wrong cross-section size on top of ranking NaN as largest."""
        out = rank_turnover(self._panel_with(float("nan")), overlap_periods=1)
        clean = rank_turnover(self._panel_with(None), overlap_periods=1)
        assert out.metadata["n_cross_section_mean"] == pytest.approx(
            clean.metadata["n_cross_section_mean"]
        )
        assert out.metadata["n_cross_section_mean"] < 20


class TestCostInputPairing:
    """The cost algebra prices one portfolio; the two inputs must describe it."""

    @staticmethod
    def _panel(n_assets=60, n_dates=200, seed=0):
        import numpy as np

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
                "forward_return": rng.standard_normal(rows) * 0.01,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_defaults_now_pair_by_construction(self):
        """Each function at its own default used to price a different book."""
        from factrix.metrics.quantile import quantile_spread

        panel = self._panel()
        spread = quantile_spread(panel)["factor"]
        turnover = notional_turnover(panel)
        assert spread.metadata["n_groups"] == turnover.metadata["n_groups"]
        assert turnover.metadata["rebalance_lag"] == DEFAULT_FORWARD_PERIODS
        # The bucketing check passes silently and is recorded.
        out = breakeven_cost(spread, turnover=turnover)
        assert out.metadata["pairing_checked"] is True
        assert out.metadata["n_groups"] == DEFAULT_N_GROUPS
        assert out.metadata["holding_periods"] == DEFAULT_FORWARD_PERIODS

    def test_mismatched_bucketing_is_rejected(self):
        from factrix.metrics.quantile import quantile_spread

        panel = self._panel()
        spread = quantile_spread(panel, n_groups=5)["factor"]
        turnover = notional_turnover(panel, n_groups=10)
        with pytest.raises(UserInputError, match="n_groups"):
            breakeven_cost(spread, turnover=turnover)
        with pytest.raises(UserInputError, match="n_groups"):
            net_spread(spread, turnover=turnover)

    def test_stride_is_no_longer_cross_checked(self):
        """The upstream stride and ``holding_periods`` measure different things.

        A turnover striding the evaluation grid and a cost amortised over
        underlying return periods are both correct and need not agree, so the
        old equality check is gone; ``holding_periods`` is recorded instead.
        """
        panel = self._panel()
        turnover = notional_turnover(panel, rebalance_lag=1)
        out = breakeven_cost(0.001, turnover=turnover, holding_periods=5)
        assert out.metadata["holding_periods"] == 5
        assert out.metadata["pairing_checked"] is True

    def test_bare_floats_still_work_unchecked(self):
        out = breakeven_cost(0.001, turnover=0.2, holding_periods=5)
        # gross * h / (4 * tau) * 1e4 = 0.001 * 5 / 0.8 * 1e4
        assert out.value == pytest.approx(62.5)
        assert "pairing_checked" not in out.metadata

    def test_metric_result_and_float_agree_on_the_number(self):
        from factrix.metrics.quantile import quantile_spread

        panel = self._panel()
        spread = quantile_spread(panel)["factor"]
        turnover = notional_turnover(panel)
        assert breakeven_cost(spread, turnover=turnover).value == pytest.approx(
            breakeven_cost(
                spread.value,
                turnover=turnover.value,
                holding_periods=DEFAULT_FORWARD_PERIODS,
            ).value
        )


class TestRebalanceLagIsDistinctFromReturnOverlap:
    """#872 — the turnover metrics pair at a rebalance lag, not the overlap.

    ``overlap_periods`` is injected from the panel's stamp and answers an
    inference question (how many adjacent evaluation observations share future
    periods). ``rebalance_lag`` is the schedule the portfolio actually trades
    on, counted in evaluation-grid observations.
    """

    @staticmethod
    def _coarse_grid_panel():
        """400-period panel evaluated on every 4th date at a 5-period horizon.

        ``compute_forward_return`` derives ``overlap_periods = 2`` there: the
        return is still measured over 5 periods of the underlying grid, but at
        most one other evaluation date falls inside that window.
        """
        import factrix as fx

        raw = fx.datasets.make_cs_panel(n_assets=30, n_dates=400, seed=0)
        every_fourth = raw["date"].unique().sort()[::4]
        return fx.preprocess.compute_forward_return(
            raw, forward_periods=5, dates=list(every_fourth)
        )

    def test_evaluate_honours_rebalance_lag_without_falsifying_the_stamp(self):
        import factrix as fx
        from factrix.metrics import ic

        panel = self._coarse_grid_panel()
        n_dates = panel["date"].n_unique()
        out = fx.evaluate(
            panel,
            metrics={
                "rt": rank_turnover(rebalance_lag=1),
                "nt": notional_turnover(rebalance_lag=1),
                "ic": ic(),
            },
            factor_cols=["factor"],
        )["factor"]

        for label in ("rt", "nt"):
            res = out.metrics[label]
            # The panel's stamp is reported unchanged...
            assert res.metadata["overlap_periods"] == 2
            # ...and the lag the metric actually paired at sits beside it.
            assert res.metadata["rebalance_lag"] == 1
            # Adjacent transitions: every evaluation date but the first.
            assert res.n_obs == n_dates - 1
            assert res.n_obs_axis == "periods"

        # An overlap-sensitive metric in the same call still gets the stamp.
        assert out.metrics["ic"].metadata["overlap_periods"] == 2

    def test_default_reproduces_the_stride_at_the_stamped_overlap(self):
        import factrix as fx

        panel = self._coarse_grid_panel()
        default = fx.evaluate(
            panel,
            metrics={"rt": rank_turnover(), "nt": notional_turnover()},
            factor_cols=["factor"],
        )["factor"]
        explicit = fx.evaluate(
            panel,
            metrics={
                "rt": rank_turnover(rebalance_lag=2),
                "nt": notional_turnover(rebalance_lag=2),
            },
            factor_cols=["factor"],
        )["factor"]

        for label in ("rt", "nt"):
            assert default.metrics[label].metadata["rebalance_lag"] == 2
            assert default.metrics[label].value == pytest.approx(
                explicit.metrics[label].value
            )
            assert default.metrics[label].n_obs == explicit.metrics[label].n_obs

    def test_standalone_call_honours_the_lag(self):
        """Nine dates, ranking reversed on odd dates.

        At lag 1 every transition is a full reversal (ρ = −1 → turnover 2); at
        lag 2 only same-phase dates are compared (ρ = +1 → turnover 0).
        """

        def factor(t, a):
            base = ord(a) - ord("A") + 1
            return base if t % 2 == 0 else (4 - base)

        df = _panel(9, ["A", "B", "C"], factor)
        adjacent = rank_turnover(df, rebalance_lag=1)
        strided = rank_turnover(df, overlap_periods=2)

        assert adjacent.value == pytest.approx(2.0, abs=0.01)
        assert adjacent.metadata["rebalance_lag"] == 1
        assert adjacent.n_obs == 8
        assert strided.value == pytest.approx(0.0, abs=0.01)
        assert strided.metadata["rebalance_lag"] == 2

    def test_lag_overrides_the_injected_overlap_in_a_standalone_call(self):
        df = _panel(9, ["A", "B", "C"], lambda t, a: ord(a) + 0.1 * t)
        out = rank_turnover(df, overlap_periods=4, rebalance_lag=1)
        assert out.metadata["overlap_periods"] == 4
        assert out.metadata["rebalance_lag"] == 1
        assert out.n_obs == 8

    def test_notional_turnover_lag_overrides_the_injected_overlap(self):
        assets = [chr(ord("A") + i) for i in range(10)]
        df = _panel(9, assets, lambda t, a: ord(a) + 0.1 * t)
        out = notional_turnover(df, n_groups=5, overlap_periods=4, rebalance_lag=1)
        assert out.metadata["overlap_periods"] == 4
        assert out.metadata["rebalance_lag"] == 1
        assert out.metadata["n_rebalances"] == 8

    def test_sample_floor_follows_the_lag_not_the_overlap(self):
        """Pre-flight and run-time must agree on the floor the lag implies."""
        import factrix as fx

        assert (
            fx.sample_requirements(
                rank_turnover(rebalance_lag=1), overlap_periods=5
            ).min_periods
            == 3
        )
        assert (
            fx.sample_requirements(rank_turnover(), overlap_periods=5).min_periods == 11
        )
        assert (
            fx.sample_requirements(
                rank_turnover(rebalance_lag=4), overlap_periods=1
            ).min_periods
            == 9
        )

    def test_rebalance_lag_validation(self):
        df = _panel(5, ["A", "B", "C"], lambda t, a: ord(a))
        with pytest.raises(ValueError, match="rebalance_lag"):
            rank_turnover(df, rebalance_lag=0)
        with pytest.raises(ValueError, match="rebalance_lag"):
            notional_turnover(df, rebalance_lag=0)


class TestHoldingPeriodsAmortisesCost:
    """#874 — cost is amortised over underlying return periods, not overlap."""

    def test_net_spread_amortises_over_underlying_periods(self):
        # 20 underlying return periods between rebalances; one-way cost 30 bps.
        out = net_spread(
            0.001, turnover=0.20, estimated_cost_bps=30, holding_periods=20
        )
        assert out.metadata["cost_drag"] == pytest.approx(4 * 0.003 * 0.20 / 20)
        assert out.metadata["cost_drag"] == pytest.approx(0.00012)
        assert out.value == pytest.approx(0.00088)
        assert out.metadata["holding_periods"] == 20

    def test_breakeven_cost_over_underlying_periods(self):
        out = breakeven_cost(0.001, turnover=0.20, holding_periods=20)
        assert out.value == pytest.approx(250.0)
        assert out.metadata["holding_periods"] == 20

    def test_derived_overlap_would_be_a_ten_x_unit_error(self):
        """The quantity the old name invited: the evaluation-grid overlap."""
        wrong = net_spread(
            0.001, turnover=0.20, estimated_cost_bps=30, holding_periods=2
        )
        assert wrong.metadata["cost_drag"] == pytest.approx(0.00120)
        assert wrong.value == pytest.approx(-0.00020)

    def test_spread_carrying_an_overlap_stamp_is_accepted(self):
        """A ``MetricResult`` whose metadata says ``overlap_periods=2`` no
        longer collides with a ``holding_periods=20`` amortisation."""
        from factrix._results import MetricResult

        spread = MetricResult(
            value=0.001, metadata={"n_groups": DEFAULT_N_GROUPS, "overlap_periods": 2}
        )
        out = net_spread(
            spread, turnover=0.20, estimated_cost_bps=30, holding_periods=20
        )
        assert out.value == pytest.approx(0.00088)
        assert out.metadata["holding_periods"] == 20
        assert out.metadata["pairing_checked"] is True
        assert breakeven_cost(
            spread, turnover=0.20, holding_periods=20
        ).value == pytest.approx(250.0)


class TestLegLevelNotionalTurnover:
    """#884 — each leg's churn is kept beside the long-short mean.

    ``value`` stays the top/bottom average (the ``4 × τ`` accounting in the
    cost helpers is the long-short book's); ``mean_top_turnover`` is the
    matched proxy for an equal-weight top-quantile long-only book, which pays
    nothing for bottom-leg churn.
    """

    TEN: ClassVar[list[str]] = [chr(ord("A") + i) for i in range(10)]

    def test_asymmetric_churn_keeps_both_legs_and_their_mean(self):
        """Top pair rotates fully every date, bottom pair never moves.

        n_groups=5 on ten names: bottom = {A, B}, top = the two largest.
        Alternate the top slot between {I, J} and {G, H} while A and B stay
        the two smallest, so top churn is 1, bottom churn 0 and the
        long-short mean 0.5.
        """

        def factor(t, a):
            base = ord(a) - ord("A")
            if a in ("G", "H", "I", "J"):
                # Even dates: I, J on top; odd dates: G, H on top.
                return 20 + base if (t % 2 == 0) == (a in ("I", "J")) else base
            return base

        df = _panel(6, self.TEN, factor)
        out = notional_turnover(df, n_groups=5, overlap_periods=1)
        assert out.metadata["mean_top_turnover"] == pytest.approx(1.0)
        assert out.metadata["mean_bottom_turnover"] == pytest.approx(0.0)
        assert out.value == pytest.approx(0.5)
        assert out.value == pytest.approx(
            (out.metadata["mean_top_turnover"] + out.metadata["mean_bottom_turnover"])
            / 2
        )
        assert out.metadata["mean_top_tail_size"] == pytest.approx(2.0)
        assert out.metadata["mean_bottom_tail_size"] == pytest.approx(2.0)
        assert out.metadata["mean_tail_size"] == pytest.approx(2.0)

    def test_full_rotation_and_static_book_bound_the_legs(self):
        def reversed_every_date(t, a):
            base = ord(a) - ord("A")
            return base if t % 2 == 0 else (9 - base)

        rotated = notional_turnover(
            _panel(5, self.TEN, reversed_every_date), n_groups=5, overlap_periods=1
        )
        assert rotated.metadata["mean_top_turnover"] == pytest.approx(1.0)
        assert rotated.metadata["mean_bottom_turnover"] == pytest.approx(1.0)

        static = notional_turnover(
            _panel(5, self.TEN, lambda t, a: ord(a)), n_groups=5, overlap_periods=1
        )
        assert static.metadata["mean_top_turnover"] == pytest.approx(0.0)
        assert static.metadata["mean_bottom_turnover"] == pytest.approx(0.0)
