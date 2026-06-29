"""Tests for small-N pairwise ordering accuracy."""

from __future__ import annotations

import math
from datetime import date, timedelta

import factrix as fx
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._types import MIN_PAIR_ACCURACY_PAIRS_HARD, MIN_PAIR_ACCURACY_PAIRS_WARN
from factrix.metrics.directional_pair_accuracy import directional_pair_accuracy


def _panel(rows_by_date: list[list[tuple[float | None, float | None]]]) -> pl.DataFrame:
    rows = []
    for di, values in enumerate(rows_by_date):
        d = date(2024, 1, 1) + timedelta(days=di)
        for ai, (factor, ret) in enumerate(values):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{ai}",
                    "factor": factor,
                    "forward_return": ret,
                }
            )
    return pl.DataFrame(rows)


class TestDirectionalPairAccuracy:
    def test_perfect_cross_sectional_ordering(self):
        data = _panel(
            [
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
                [(4.0, 4.0), (3.0, 3.0), (2.0, 2.0), (1.0, 1.0)],
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
                [(4.0, 4.0), (3.0, 3.0), (2.0, 2.0), (1.0, 1.0)],
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
            ]
        )
        result = directional_pair_accuracy(data, forward_periods=1)
        assert result.value == pytest.approx(1.0)
        assert result.p_value is None
        assert result.stat is None
        assert result.n_obs_axis == "periods"
        assert result.n_obs == 5
        assert result.metadata["n_pairs"] == 30
        assert result.metadata["n_correct_pairs"] == 30
        assert result.metadata["pooled_accuracy"] == pytest.approx(1.0)

    def test_reversed_ordering_scores_zero(self):
        data = _panel(
            [
                [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)],
                [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)],
                [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)],
                [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)],
                [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)],
            ]
        )
        result = directional_pair_accuracy(data, forward_periods=1)
        assert result.value == pytest.approx(0.0)
        assert result.metadata["n_incorrect_pairs"] == 30

    def test_ties_and_nulls_are_excluded_and_counted(self):
        data = _panel(
            [
                [
                    (1.0, 1.0),
                    (1.0, 2.0),  # factor tie with A0
                    (2.0, 2.0),  # return tie with A1
                    (3.0, 4.0),
                    (None, 5.0),
                ],
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)],
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)],
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)],
            ]
        )
        result = directional_pair_accuracy(data, forward_periods=1)
        assert result.value == pytest.approx(1.0)
        assert result.metadata["n_raw_pairs"] == 36
        assert result.metadata["n_pairs"] == 34
        assert result.metadata["factor_tie_pairs"] == 1
        assert result.metadata["return_tie_pairs"] == 1
        assert result.metadata["dropped_pairs"] == 2
        assert result.metadata["dropped_rows_null"] == 1

    def test_insufficient_comparable_pairs_short_circuits_on_pairs_axis(self):
        data = _panel([[(1.0, 1.0), (2.0, 2.0)]])
        result = directional_pair_accuracy(data, forward_periods=1)
        assert math.isnan(result.value)
        assert result.p_value is None
        assert result.n_obs_axis == "pairs"
        assert result.metadata["reason"] == "insufficient_ordering_pairs"
        assert result.metadata["min_required"] == MIN_PAIR_ACCURACY_PAIRS_HARD

    def test_warns_on_thin_pair_count(self):
        data = _panel(
            [
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            ]
        )
        assert MIN_PAIR_ACCURACY_PAIRS_HARD <= 12 < MIN_PAIR_ACCURACY_PAIRS_WARN
        with pytest.warns(UserWarning, match="MIN_PAIR_ACCURACY_PAIRS_WARN"):
            result = directional_pair_accuracy(data, forward_periods=1)
        assert result.metadata["n_pairs"] == 12
        assert WarningCode.FEW_ORDERING_PAIRS.value in result.warning_codes

    def test_dispatch_runs_on_individual_dense_panel(self):
        raw = fx.datasets.make_cs_panel(n_assets=8, n_dates=80)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        results = fx.evaluate(
            panel,
            metrics={"ordering": directional_pair_accuracy()},
            factor_cols=["factor"],
        )
        out = results["factor"].metrics["ordering"]
        assert out.name == "ordering"
        assert math.isnan(out.value) or 0.0 <= out.value <= 1.0

    def test_inspect_data_does_not_preflight_row_pairs_as_ordering_pairs(self):
        # DataProperties.n_pairs is row count, not within-date pair
        # combinations. A one-date, five-asset panel has only five rows but ten
        # ordering pairs, so the metric's pair floor must stay a runtime guard.
        data = _panel([[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]])
        info = fx.inspect_data(data)
        verdict = next(m for m in info.metrics if m.name == "directional_pair_accuracy")
        assert info.properties.n_pairs == 5
        assert verdict.blockers == []

        with pytest.warns(UserWarning, match="MIN_PAIR_ACCURACY_PAIRS_WARN"):
            result = directional_pair_accuracy(data, forward_periods=1)
        assert result.metadata["n_pairs"] == MIN_PAIR_ACCURACY_PAIRS_HARD
        assert result.value == pytest.approx(1.0)
