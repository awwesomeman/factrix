"""``fx.evaluate_horizons`` — thin sweep over overlap horizons that flattens
to ``list[EvaluationResult]`` and feeds ``compare`` / ``bhy`` directly."""

from __future__ import annotations

import warnings

import factrix as fx
import pytest
from factrix._errors import UserInputError
from factrix.metrics import ic


def _raw(n_assets: int = 20, n_dates: int = 300):
    # Sized so the default NonOverlapping IC floor (10·h) clears at h<=20.
    return fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates)


class TestSweepShape:
    def test_returns_flat_list_one_entry_per_horizon(self):
        results = fx.evaluate_horizons(
            _raw(),
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=[5, 10, 20],
        )
        assert isinstance(results, list)
        assert [r.forward_periods for r in results] == [5, 10, 20]
        assert all(r.factor == "factor" for r in results)

    def test_each_result_stamps_its_own_horizon(self):
        results = fx.evaluate_horizons(
            _raw(),
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=[5, 20],
        )
        # forward_periods is injected into the metric metadata per horizon.
        seen = {
            r.forward_periods: r.metrics["ic"].metadata["forward_periods"]
            for r in results
        }
        assert seen == {5: 5, 20: 20}

    def test_grouped_by_horizon_then_factor_order(self):
        panel = fx.datasets.make_multi_factor_panel(
            n_factors=2, n_assets=20, n_dates=300
        )
        cols = ["factor_0000", "factor_0001"]
        results = fx.evaluate_horizons(
            panel,
            metrics={"ic": ic()},
            factor_cols=cols,
            forward_periods=[5, 10],
        )
        identity = [(r.forward_periods, r.factor) for r in results]
        assert identity == [
            (5, "factor_0000"),
            (5, "factor_0001"),
            (10, "factor_0000"),
            (10, "factor_0001"),
        ]

    def test_single_horizon_matches_evaluate(self):
        raw = _raw()
        swept = fx.evaluate_horizons(
            raw,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=[5],
        )
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        direct = fx.evaluate(panel, metrics={"ic": ic()}, factor_cols=["factor"])
        assert len(swept) == 1
        assert swept[0].metrics["ic"].value == direct["factor"].metrics["ic"].value


class TestFeedsAggregationLayer:
    def test_compare_one_row_per_factor_horizon(self):
        panel = fx.datasets.make_multi_factor_panel(
            n_factors=2, n_assets=20, n_dates=300
        )
        cols = ["factor_0000", "factor_0001"]
        results = fx.evaluate_horizons(
            panel,
            metrics={"ic": ic()},
            factor_cols=cols,
            forward_periods=[5, 10],
        )
        board = fx.compare(results, metrics=["ic"])
        assert board.height == 4  # 2 factors x 2 horizons
        assert set(board["forward_periods"].to_list()) == {5, 10}

    def test_bhy_partitions_per_horizon(self):
        # Two factors per horizon so each bucket has n=2 (no singleton warning).
        panel = fx.datasets.make_multi_factor_panel(
            n_factors=2, n_assets=20, n_dates=300
        )
        cols = ["factor_0000", "factor_0001"]
        results = fx.evaluate_horizons(
            panel,
            metrics={"ic": ic()},
            factor_cols=cols,
            forward_periods=[5, 10],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no mixed-horizon / singleton warning
            screens = fx.multi_factor.bhy(
                results, metrics=["ic"], expand_over=("forward_periods",)
            )
        assert "ic" in screens

    def test_flatten_without_expand_over_raises_duplicate_identity(self):
        # Same factor at two horizons -> duplicate (factor,) identity in bhy.
        results = fx.evaluate_horizons(
            _raw(),
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=[5, 10],
        )
        with pytest.raises(UserInputError):
            fx.multi_factor.bhy(results, metrics=["ic"])


class TestStrictForwarded:
    def test_strict_false_keeps_inapplicable_as_nan(self):
        # h=60 blows the IC floor on this short panel; strict=False keeps it.
        results = fx.evaluate_horizons(
            _raw(n_dates=160),
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=[5, 60],
            strict=False,
        )
        by_h = {r.forward_periods: r for r in results}
        assert by_h[60].metrics["ic"].metadata["reason"] == "insufficient_ic_periods"

    def test_strict_true_raises_on_inapplicable(self):
        with pytest.raises(UserInputError):
            fx.evaluate_horizons(
                _raw(n_dates=160),
                metrics={"ic": ic()},
                factor_cols=["factor"],
                forward_periods=[5, 60],
                strict=True,
            )


class TestForwardPeriodsValidation:
    @pytest.mark.parametrize(
        "bad", [[], 5, (5, 10), [5, 5], [5, -1], [5, 0], [5.0], ["a"]]
    )
    def test_rejects_bad_forward_periods(self, bad):
        with pytest.raises(UserInputError):
            fx.evaluate_horizons(
                _raw(),
                metrics={"ic": ic()},
                factor_cols=["factor"],
                forward_periods=bad,
            )

    def test_rejects_already_attached_panel(self):
        panel = fx.preprocess.compute_forward_return(_raw(), forward_periods=5)
        with pytest.raises(UserInputError):
            fx.evaluate_horizons(
                panel,
                metrics={"ic": ic()},
                factor_cols=["factor"],
                forward_periods=[5],
            )


def test_exported():
    assert hasattr(fx, "evaluate_horizons")
    assert "evaluate_horizons" in fx.__all__
