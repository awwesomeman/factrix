"""``compute_forward_return(..., dates=)`` — a caller-chosen evaluation grid.

Two horizons live on a panel: ``forward_periods`` (the return horizon, the
hypothesis) and ``overlap_periods`` (how many adjacent evaluation
observations share future periods, the quantity inference consumes). On the
full grid they coincide; on a coarser evaluation grid the overlap is derived
from the kept dates' spacing on the *full* period grid.
"""

from __future__ import annotations

import warnings

import factrix as fx
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._data_input import _FORWARD_PERIODS_COL, _OVERLAP_PERIODS_COL
from factrix._errors import UserInputError
from factrix.metrics import ic, notional_turnover, quantile_spread, rank_turnover
from factrix.multi_factor import bhy
from factrix.preprocess import compute_forward_return
from factrix.preprocess.returns import _overlap_on_grid

from tests.conftest import make_result, make_spec

H = 60


@pytest.fixture(scope="module")
def raw() -> pl.DataFrame:
    return fx.datasets.make_cs_panel(n_assets=30, n_dates=1500, seed=11)


@pytest.fixture(scope="module")
def grid(raw: pl.DataFrame) -> pl.Series:
    return raw["date"].unique().sort()


def _stamps(panel: pl.DataFrame) -> tuple[int, int]:
    return int(panel[_FORWARD_PERIODS_COL][0]), int(panel[_OVERLAP_PERIODS_COL][0])


class TestOverlapDerivation:
    def test_full_grid_stamps_overlap_equal_to_horizon(self, raw):
        assert _stamps(compute_forward_return(raw, forward_periods=H)) == (H, H)

    def test_stride_equal_to_horizon_is_non_overlapping(self, raw, grid):
        panel = compute_forward_return(
            raw, forward_periods=H, dates=grid.gather_every(H)
        )
        assert _stamps(panel) == (H, 1)
        assert panel["date"].n_unique() == 24

    def test_stride_one_third_of_horizon_overlaps_three(self, raw, grid):
        panel = compute_forward_return(
            raw, forward_periods=H, dates=grid.gather_every(20)
        )
        assert _stamps(panel) == (H, 3)

    def test_uneven_grid_takes_the_maximum_not_a_typical_count(self, raw, grid):
        # Stride 30 everywhere (each row overlaps exactly one later row) plus
        # one extra date 5 periods after a kept one: that row overlaps two
        # later rows, so the max rule says 3 where a median would say 2.
        index = [*range(0, 1400, 30), 1385]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # uneven_evaluation_grid is expected
            panel = compute_forward_return(
                raw, forward_periods=H, dates=grid.gather(sorted(index))
            )
        assert _stamps(panel) == (H, 3)
        assert _overlap_on_grid(index, H) == 3

    def test_overlap_formula_on_small_index_sets(self):
        assert _overlap_on_grid(list(range(100)), 5) == 5
        assert _overlap_on_grid([0, 60, 120], 60) == 1
        assert _overlap_on_grid([0, 20, 40, 60], 60) == 3
        assert _overlap_on_grid([7], 60) == 1

    def test_vanished_period_still_counts_on_the_full_index(self, raw, grid):
        # Every asset's price is non-finite on one period strictly between
        # two kept dates, so that period vanishes from the OUTPUT panel. On
        # an index rebuilt from the output the kept dates would sit 59
        # apart and count as overlapping; on the full grid they are 60
        # apart and do not.
        vanish = grid[30]
        holed = raw.with_columns(
            pl.when(pl.col("date") == vanish)
            .then(float("nan"))
            .otherwise(pl.col("price"))
            .alias("price")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ragged_period_grid is expected
            panel = compute_forward_return(
                holed, forward_periods=H, dates=grid.gather_every(H)
            )
        assert vanish not in panel["date"].to_list()
        assert _stamps(panel) == (H, 1)

    def test_only_evaluation_dates_are_returned(self, raw, grid):
        kept = grid.gather_every(H)
        panel = compute_forward_return(raw, forward_periods=H, dates=kept)
        assert set(panel["date"].to_list()) <= set(kept.to_list())

    def test_accepts_a_plain_sequence(self, raw, grid):
        kept = grid.gather_every(H).to_list()
        panel = compute_forward_return(raw, forward_periods=H, dates=kept)
        assert _stamps(panel) == (H, 1)

    def test_return_values_match_the_full_grid(self, raw, grid):
        full = compute_forward_return(raw, forward_periods=H)
        kept = grid.gather_every(H)
        sampled = compute_forward_return(raw, forward_periods=H, dates=kept)
        expected = full.filter(pl.col("date").is_in(kept.implode())).drop(
            _FORWARD_PERIODS_COL, _OVERLAP_PERIODS_COL
        )
        got = sampled.drop(_FORWARD_PERIODS_COL, _OVERLAP_PERIODS_COL)
        assert got.sort("date", "asset_id").equals(expected.sort("date", "asset_id"))


class TestUnevenEvaluationGridWarning:
    """An uneven ``dates=`` grid is disclosed, a constant-stride one is not."""

    @staticmethod
    def _uneven_index(n: int = 1400) -> list[int]:
        """Period indices spaced (20, 20, 40) in a repeating cycle."""
        index, i = [0], 0
        for step in (20, 20, 40) * (n // 80 + 1):
            i += step
            if i >= n:
                break
            index.append(i)
        return index

    def test_uneven_grid_warns(self, raw, grid):
        dates = grid.gather(self._uneven_index())
        with pytest.warns(UserWarning, match="uneven_evaluation_grid"):
            compute_forward_return(raw, forward_periods=H, dates=dates)

    def test_constant_stride_grid_does_not_warn(self, raw, grid):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_forward_return(raw, forward_periods=H, dates=grid.gather_every(20))
        assert not [w for w in caught if "uneven_evaluation_grid" in str(w.message)]

    def test_full_grid_does_not_warn(self, raw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_forward_return(raw, forward_periods=H)
        assert not [w for w in caught if "uneven_evaluation_grid" in str(w.message)]

    def test_uneven_grid_code_is_documented(self):
        assert WarningCode.UNEVEN_EVALUATION_GRID.description


class TestEvaluationGridValidation:
    def test_date_off_the_grid_raises_without_snapping(self, raw, grid):
        # A value strictly between two grid periods, in the grid's own dtype.
        between = pl.Series([grid[0] + (grid[1] - grid[0]) / 2]).cast(grid.dtype)
        with pytest.raises(UserInputError, match="never snapped"):
            compute_forward_return(raw, forward_periods=H, dates=between)

    def test_python_datetimes_match_the_grid_exactly(self, raw, grid):
        # A plain list of Python datetimes arrives in microseconds; the unit
        # is re-expressed, membership is still exact.
        kept = grid.gather_every(H).to_list()
        assert _stamps(compute_forward_return(raw, forward_periods=H, dates=kept)) == (
            H,
            1,
        )
        shifted = [kept[0] + (grid[1] - grid[0]) / 2, *kept[1:]]
        with pytest.raises(UserInputError, match="never snapped"):
            compute_forward_return(raw, forward_periods=H, dates=shifted)

    def test_empty_dates_raises(self, raw):
        with pytest.raises(UserInputError, match="at least one evaluation date"):
            compute_forward_return(raw, forward_periods=H, dates=[])

    def test_scalar_dates_rejected(self, raw, grid):
        with pytest.raises(UserInputError, match="Series or a sequence"):
            compute_forward_return(raw, forward_periods=H, dates=str(grid[0]))

    def test_mismatched_dtype_rejected(self, raw, grid):
        with pytest.raises(UserInputError, match="date dtype"):
            compute_forward_return(
                raw, forward_periods=H, dates=[str(d) for d in grid.gather_every(H)]
            )

    def test_all_dates_in_the_tail_raises(self, raw, grid):
        with pytest.raises(UserInputError, match="horizon's tail"):
            compute_forward_return(raw, forward_periods=H, dates=[grid[-1]])


class TestEvaluateOnCoarseGrid:
    def test_quarterly_sampled_panel_runs_instead_of_short_circuiting(self, raw, grid):
        panel = compute_forward_return(
            raw, forward_periods=H, dates=grid.gather_every(H)
        )
        er = fx.evaluate(
            panel,
            metrics={"ic": ic(), "spread": quantile_spread()},
            factor_cols=["factor"],
        )["factor"]
        assert (er.forward_periods, er.overlap_periods) == (H, 1)
        assert er.metrics["ic"].reason is None
        assert er.metrics["spread"].reason is None
        assert er.metrics["ic"].n_obs == 24
        assert er.metrics["spread"].n_obs == 24
        assert er.metrics["ic"].metadata["overlap_periods"] == 1
        assert er.to_frame()["overlap_periods"].unique().to_list() == [1]

    def test_hand_sub_sampled_panel_short_circuits_and_points_at_dates(self, raw, grid):
        # The stale-stamp case the derivation exists for: sub-sampling AFTER
        # compute_forward_return keeps overlap_periods = 60, so the
        # stride-scaled floor (50 x 60) rejects a 24-period panel.
        full = compute_forward_return(raw, forward_periods=H)
        stale = full.filter(pl.col("date").is_in(grid.gather_every(H).implode()))
        with pytest.raises(
            fx.InsufficientSampleError, match=r"dates=<evaluation dates>"
        ):
            fx.evaluate(stale, metrics={"ic": ic()}, factor_cols=["factor"])
        er = fx.evaluate(
            stale, metrics={"ic": ic()}, factor_cols=["factor"], strict=False
        )["factor"]
        assert er.metrics["ic"].reason == "insufficient_ic_periods"
        assert "dates=<evaluation dates>" in er.metrics["ic"].metadata["hint"]
        [w] = [
            w
            for w in er.warnings
            if w.source == "ic" and w.message.startswith("insufficient_ic_periods")
        ]
        assert w.code is fx.WarningCode.METRIC_UNAVAILABLE
        assert "compute_forward_return(..., dates=" in w.message

    def test_evaluate_horizons_passes_dates_through(self, raw, grid):
        results = fx.evaluate_horizons(
            raw,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=[20, H],
            dates=grid.gather_every(20),
        )
        assert [(r.forward_periods, r.overlap_periods) for r in results] == [
            (20, 1),
            (H, 3),
        ]


class TestStaleStampHintIsTimeAxisOnly:
    """The "rebuild with dates=" sentence belongs to time-axis shortfalls.

    A stale ``overlap_periods`` stamp can only shrink a floor counted in
    periods (or event periods). A cross-section shortfall — too few assets to
    fill the requested buckets — is unrelated to how the panel was sampled in
    time, so the hint must not be attached to it however large the stamp is.
    """

    def test_period_axis_short_circuit_carries_the_hint(self, raw, grid):
        full = compute_forward_return(raw, forward_periods=H)
        stale = full.filter(pl.col("date").is_in(grid.gather_every(H).implode()))
        out = fx.evaluate(
            stale, metrics={"ic": ic()}, factor_cols=["factor"], strict=False
        )["factor"].metrics["ic"]
        assert out.n_obs_axis == "periods"
        assert out.metadata["reason"] == "insufficient_ic_periods"
        assert "compute_forward_return(..., dates=" in out.metadata["hint"]

    def test_period_axis_hint_reaches_the_strict_error(self, raw, grid):
        full = compute_forward_return(raw, forward_periods=H)
        stale = full.filter(pl.col("date").is_in(grid.gather_every(H).implode()))
        with pytest.raises(
            fx.InsufficientSampleError, match=r"compute_forward_return\(\.\.\., dates="
        ):
            fx.evaluate(stale, metrics={"ic": ic()}, factor_cols=["factor"])

    def test_hand_written_period_axis_short_circuit_carries_the_hint(self):
        """A metric that short-circuits outside the shared floor helpers still
        declares its axis, so the hint follows the axis rather than the call
        site."""
        dates = pl.date_range(
            pl.date(2024, 1, 1), pl.date(2024, 1, 6), "1d", eager=True
        )
        panel = pl.DataFrame(
            {
                "date": [d for d in dates for _ in range(4)],
                "asset_id": [a for _ in dates for a in "ABCD"],
                "factor": [
                    float((i + t) % 4) for t in range(len(dates)) for i in range(4)
                ],
            }
        )
        out = rank_turnover(panel, overlap_periods=5)
        assert out.n_obs_axis == "periods"
        assert out.metadata["reason"] == "insufficient_dates"
        assert "compute_forward_return(..., dates=" in out.metadata["hint"]

    def test_asset_axis_short_circuit_does_not_carry_the_hint(self):
        dates = pl.date_range(
            pl.date(2024, 1, 1), pl.date(2024, 1, 20), "1d", eager=True
        )
        panel = pl.DataFrame(
            {
                "date": [d for d in dates for _ in range(4)],
                "asset_id": [a for _ in dates for a in "ABCD"],
                "factor": [
                    float((i + t) % 4) for t in range(len(dates)) for i in range(4)
                ],
            }
        )
        out = notional_turnover(panel, n_groups=10, overlap_periods=5)
        assert out.n_obs_axis == "assets"
        assert out.metadata["reason"] == "insufficient_assets_for_quantile_groups"
        assert out.metadata["overlap_periods"] == 5
        assert "hint" not in out.metadata


class TestIdentityIsTheHorizon:
    def test_same_horizon_on_two_grids_is_one_hypothesis(self):
        # Quarterly (overlap 1) and monthly (overlap 3) evaluations of the
        # same 60-period return are two estimates of one hypothesis, so they
        # collide on the (factor, forward_periods) identity rather than
        # forming two BHY hypotheses.
        make_spec("ic")
        results = [
            make_result(
                factor="f", p=0.01, metric="ic", forward_periods=H, overlap_periods=1
            ),
            make_result(
                factor="f", p=0.02, metric="ic", forward_periods=H, overlap_periods=3
            ),
        ]
        with pytest.raises(UserInputError, match="unique \\(factor, forward_periods"):
            bhy(results, metrics=["ic"], q=0.05)

    def test_mixed_overlap_at_one_horizon_does_not_warn(self):
        make_spec("ic")
        results = [
            make_result(
                factor="f1", p=0.01, metric="ic", forward_periods=H, overlap_periods=1
            ),
            make_result(
                factor="f2", p=0.02, metric="ic", forward_periods=H, overlap_periods=3
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            bhy(results, metrics=["ic"], q=0.05)
