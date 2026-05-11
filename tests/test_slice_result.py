"""Tests for the SliceResult container returned by by_slice (#212)."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import pytest
from factrix import SliceResult
from factrix._types import MetricOutput


def _make_outputs() -> dict[str, MetricOutput]:
    return {
        "bull": MetricOutput(
            name="ic",
            value=0.07,
            stat=2.31,
            significance="*",
            metadata={"p_value": 0.024, "method": "non-overlapping t-test"},
        ),
        "bear": MetricOutput(
            name="ic",
            value=-0.02,
            stat=-0.41,
            significance="",
            metadata={"p_value": 0.683, "method": "non-overlapping t-test"},
        ),
    }


class TestMappingBehaviour:
    def test_basic_wiring(self):
        r = SliceResult(_make_outputs())
        assert isinstance(r, Mapping)
        assert r["bull"].value == pytest.approx(0.07)
        assert list(r) == ["bull", "bear"]
        assert len(r) == 2

    def test_iteration_order_preserved(self):
        # Insertion order, not lexicographic — matches polars partition_by behaviour.
        r = SliceResult({"z": _make_outputs()["bull"], "a": _make_outputs()["bear"]})
        assert list(r) == ["z", "a"]


class TestToFrame:
    def test_default_schema(self):
        df = SliceResult(_make_outputs()).to_frame()
        assert df.columns == ["slice", "name", "value", "stat", "p_value"]
        assert df.schema["slice"] == pl.Utf8
        assert df.schema["name"] == pl.Utf8
        assert df.schema["value"] == pl.Float64
        assert df.schema["stat"] == pl.Float64
        assert df.schema["p_value"] == pl.Float64

    def test_row_count_and_order(self):
        df = SliceResult(_make_outputs()).to_frame()
        assert df.height == 2
        assert df["slice"].to_list() == ["bull", "bear"]
        assert df["value"].to_list() == pytest.approx([0.07, -0.02])
        assert df["stat"].to_list() == pytest.approx([2.31, -0.41])
        assert df["p_value"].to_list() == pytest.approx([0.024, 0.683])

    def test_custom_slice_col(self):
        df = SliceResult(_make_outputs()).to_frame(slice_col="regime")
        assert df.columns == ["regime", "name", "value", "stat", "p_value"]
        assert df["regime"].to_list() == ["bull", "bear"]

    def test_missing_stat_and_p_value_become_null(self):
        # Descriptive metric without stat or p_value (e.g. event count summary).
        outputs = {
            "g1": MetricOutput(name="counts", value=42.0),
            "g2": MetricOutput(name="counts", value=17.0, metadata={"note": "x"}),
        }
        df = SliceResult(outputs).to_frame()
        assert df["stat"].to_list() == [None, None]
        assert df["p_value"].to_list() == [None, None]
        assert df["value"].to_list() == pytest.approx([42.0, 17.0])

    def test_empty_result(self):
        df = SliceResult({}).to_frame()
        assert df.columns == ["slice", "name", "value", "stat", "p_value"]
        assert df.height == 0


class TestReprHtml:
    def test_repr_html_non_empty(self):
        html = SliceResult(_make_outputs())._repr_html_()
        assert isinstance(html, str)
        assert html  # non-empty
        # Delegates to polars DataFrame._repr_html_, which renders <table>.
        assert "<table" in html
