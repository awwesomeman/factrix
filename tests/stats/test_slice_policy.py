"""Tests for ``_detect_strict_subsets`` / ``_downscale_n_groups``."""

from __future__ import annotations

import polars as pl
import pytest
from factrix._stats.slice_policy import (
    _detect_strict_subsets,
    _downscale_n_groups,
)


def _slice(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["date", "asset"], orient="row")


class TestDetectStrictSubsets:
    def test_disjoint_slices_no_pair(self):
        slices = {
            "a": _slice([("2020-01-01", "X"), ("2020-01-02", "X")]),
            "b": _slice([("2020-01-03", "Y"), ("2020-01-04", "Y")]),
        }
        assert _detect_strict_subsets(slices) == []

    def test_partial_overlap_no_pair(self):
        # A and B share some keys but neither is a subset → no warn.
        slices = {
            "a": _slice([("d1", "x"), ("d2", "x"), ("d3", "x")]),
            "b": _slice([("d2", "x"), ("d3", "x"), ("d4", "x")]),
        }
        assert _detect_strict_subsets(slices) == []

    def test_strict_subset_emits_pair(self):
        # Top10 ⊊ Top50 (hierarchical universe pattern).
        slices = {
            "top10": _slice([("d1", "a"), ("d1", "b")]),
            "top50": _slice([("d1", "a"), ("d1", "b"), ("d1", "c"), ("d1", "d")]),
        }
        pairs = _detect_strict_subsets(slices)
        assert pairs == [("top10", "top50")]

    def test_subset_in_either_direction(self):
        # B contains A; verify flag goes (subset, superset) regardless
        # of dict insertion order.
        slices = {
            "big": _slice([("d1", "a"), ("d1", "b"), ("d1", "c")]),
            "small": _slice([("d1", "a")]),
        }
        pairs = _detect_strict_subsets(slices)
        assert pairs == [("small", "big")]

    def test_three_way_chain(self):
        # A ⊊ B ⊊ C → three pairs reported (every nested pair).
        slices = {
            "A": _slice([("d1", "x")]),
            "B": _slice([("d1", "x"), ("d1", "y")]),
            "C": _slice([("d1", "x"), ("d1", "y"), ("d1", "z")]),
        }
        pairs = _detect_strict_subsets(slices)
        assert sorted(pairs) == [("A", "B"), ("A", "C"), ("B", "C")]

    def test_equal_keysets_not_flagged(self):
        # Identical key-sets are not strict subsets → not flagged.
        slices = {
            "a": _slice([("d1", "x"), ("d2", "y")]),
            "b": _slice([("d2", "y"), ("d1", "x")]),  # same set, different order
        }
        assert _detect_strict_subsets(slices) == []

    def test_custom_key_cols(self):
        df_a = pl.DataFrame({"event_id": [1, 2], "extra": ["p", "q"]})
        df_b = pl.DataFrame({"event_id": [1, 2, 3], "extra": ["p", "q", "r"]})
        pairs = _detect_strict_subsets({"a": df_a, "b": df_b}, key_cols=("event_id",))
        assert pairs == [("a", "b")]

    def test_missing_key_column_raises(self):
        df_a = pl.DataFrame({"date": ["d1"], "asset": ["x"]})
        df_b = pl.DataFrame({"date": ["d1"]})  # missing 'asset'
        with pytest.raises(ValueError, match="missing key columns"):
            _detect_strict_subsets({"a": df_a, "b": df_b})

    def test_empty_slices(self):
        assert _detect_strict_subsets({}) == []


class TestDownscaleNGroups:
    def test_none_passes_through(self):
        # min_assets_per_group=None → metric doesn't bucket; return base.
        assert _downscale_n_groups(5, n_assets=20, min_assets_per_group=None) == 5

    def test_capacity_caps_groups(self):
        # n_assets=60, min_assets_per_group=30 → capacity=2; min(5, 2) = 2.
        assert _downscale_n_groups(5, n_assets=60, min_assets_per_group=30) == 2

    def test_capacity_above_base_returns_base(self):
        # n_assets=300, min=30 → capacity=10; base=5 → returns 5.
        assert _downscale_n_groups(5, n_assets=300, min_assets_per_group=30) == 5

    def test_floor_at_2_when_assets_below_min(self):
        # n_assets=15, min=30 → 15//30 = 0; floored to 2.
        assert _downscale_n_groups(5, n_assets=15, min_assets_per_group=30) == 2

    def test_zero_assets_floored_at_2(self):
        assert _downscale_n_groups(5, n_assets=0, min_assets_per_group=30) == 2

    def test_invalid_min_raises(self):
        with pytest.raises(ValueError, match="min_assets_per_group must be >= 1"):
            _downscale_n_groups(5, n_assets=100, min_assets_per_group=0)
