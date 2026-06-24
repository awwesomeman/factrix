"""``EvaluationResult`` / ``MetricResult`` dataclasses + serialisation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

import polars as pl
import pytest
from factrix import (
    EvaluationResult,
    MetricResult,
    Warning,
    WarningCode,
)
from factrix._axis import DataStructure, FactorDensity, FactorScope


def _sample_group() -> Mapping[str, MetricResult]:
    ic_out = MetricResult(
        value=0.05,
        p_value=0.012,
        n_obs=100,
        stat=2.5,
        metadata={"p_value": 0.012},
        name="ic",
    )
    ic_ir_out = MetricResult(
        value=0.42,
        n_obs=100,
        name="ic_ir",
    )
    return MappingProxyType({"ic": ic_out, "ic_ir": ic_ir_out})


def _sample_result(
    group: Mapping[str, MetricResult], warnings=None, plan: str = "1. ic [per-factor]"
) -> EvaluationResult:
    return EvaluationResult(
        factor="mom_12_1",
        cell=(FactorScope.INDIVIDUAL, FactorDensity.DENSE, DataStructure.PANEL),
        forward_periods=5,
        n_periods=100,
        n_pairs=2500,
        n_assets=25,
        metrics=group,
        plan=plan,
        warnings=warnings or [],
    )


class TestMetricsMapping:
    def test_dict_like_access(self):
        g = _sample_group()
        assert "ic" in g
        assert "missing" not in g
        assert g["ic"].value == 0.05
        assert g.get("ic").value == 0.05
        assert g.get("missing") is None
        assert len(g) == 2
        assert set(g.keys()) == {"ic", "ic_ir"}
        assert {o.name for o in g.values()} == {"ic", "ic_ir"}
        assert {k for k, _ in g.items()} == {"ic", "ic_ir"}
        assert list(iter(g)) == ["ic", "ic_ir"]

    def test_mapping_is_read_only(self):
        g = _sample_result(_sample_group()).metrics
        with pytest.raises(TypeError):
            g["ic"] = MetricResult(value=0.0, name="ic")


class TestEvaluationResultToFrame:
    def test_schema_and_dtypes(self):
        r = _sample_result(_sample_group())
        df = r.to_frame()
        assert df.columns == [
            "factor",
            "n_assets",
            "metric_name",
            "value",
            "p_value",
            "stat",
            "n_obs",
            "n_obs_axis",
            "is_applicable",
            "reason",
            "warning_codes",
        ]
        assert df.schema["value"] == pl.Float64
        assert df.schema["p_value"] == pl.Float64
        assert df.schema["n_obs"] == pl.Int64
        assert df.schema["n_obs_axis"] == pl.Utf8
        assert df.schema["is_applicable"] == pl.Boolean
        assert df.schema["reason"] == pl.Utf8
        assert df.schema["warning_codes"] == pl.List(pl.Utf8)
        assert df.height == 2

    def test_n_obs_carries_per_metric_sample_size(self):
        ic_out = MetricResult(value=0.05, n_obs=114, name="ic")
        spread_out = MetricResult(value=0.01, n_obs=23, name="spread")
        g = MappingProxyType({"ic": ic_out, "spread": spread_out})
        df = _sample_result(g).to_frame()
        assert df.filter(pl.col("metric_name") == "ic")["n_obs"][0] == 114
        assert df.filter(pl.col("metric_name") == "spread")["n_obs"][0] == 23

    def test_short_circuit_row_is_null(self):
        bad = MetricResult(value=float("nan"), name="ic")
        g = MappingProxyType({"ic": bad})
        df = _sample_result(g).to_frame()
        row = df.row(0, named=True)
        assert row["value"] is None
        assert row["p_value"] is None

    def test_short_circuit_marks_metric_inapplicable(self):
        bad = MetricResult(
            value=float("nan"),
            metadata={"reason": "insufficient_ic_periods"},
            name="ic",
        )
        g = MappingProxyType({"ic": bad})
        r = _sample_result(g)
        row = r.to_frame().row(0, named=True)

        assert bad.is_applicable is False
        assert bad.reason == "insufficient_ic_periods"
        assert row["is_applicable"] is False
        assert row["reason"] == "insufficient_ic_periods"
        assert r.to_dict()["metrics"]["ic"]["is_applicable"] is False
        assert r.to_dict()["metrics"]["ic"]["reason"] == "insufficient_ic_periods"

    def test_warning_codes_filter_by_source(self):
        warnings = [
            Warning(code=WarningCode.FEW_ASSETS, source="ic", message="thin"),
            Warning(
                code=WarningCode.SERIAL_CORRELATION_DETECTED,
                source=None,
                message="bundle",
            ),
        ]
        r = _sample_result(_sample_group(), warnings=warnings)
        df = r.to_frame()
        ic_row = df.filter(pl.col("metric_name") == "ic").row(0, named=True)
        ic_ir_row = df.filter(pl.col("metric_name") == "ic_ir").row(0, named=True)
        assert ic_row["warning_codes"] == [WarningCode.FEW_ASSETS.value]
        assert ic_ir_row["warning_codes"] == []

    def test_metric_name_from_name_field(self):
        out = MetricResult(value=0.01, name="fm_beta")
        g = MappingProxyType({"fm_beta": out})
        df = _sample_result(g).to_frame()
        assert df.row(0, named=True)["metric_name"] == "fm_beta"


class TestEvaluationResultToDict:
    def test_round_trips_through_json(self):
        warnings = [
            Warning(code=WarningCode.FEW_ASSETS, source="ic", message="thin"),
        ]
        r = _sample_result(_sample_group(), warnings=warnings)
        d = r.to_dict()
        encoded = json.dumps(d)
        back = json.loads(encoded)
        assert back["factor"] == "mom_12_1"
        assert back["cell"]["scope"] == "individual"
        assert back["cell"]["density"] == "dense"
        assert back["cell"]["structure"] == "panel"
        assert back["n_periods"] == 100
        assert back["n_pairs"] == 2500
        assert "n_obs" not in back
        assert "metrics_partition" not in back
        assert back["metrics"]["ic"]["p_value"] == 0.012
        assert back["metrics"]["ic"]["is_applicable"] is True
        assert back["metrics"]["ic"]["reason"] is None
        assert back["warnings"][0]["code"] == WarningCode.FEW_ASSETS.value
        assert back["plan"] == "1. ic [per-factor]"

    def test_nonfinite_floats_become_null(self):
        bad = MetricResult(
            value=float("nan"),
            p_value=float("nan"),
            stat=float("inf"),
            metadata={"p_value": float("nan")},
            name="ic",
        )
        g = MappingProxyType({"ic": bad})
        d = _sample_result(g).to_dict()
        assert d["metrics"]["ic"]["value"] is None
        assert d["metrics"]["ic"]["stat"] is None
        assert d["metrics"]["ic"]["p_value"] is None
        json.dumps(d)


class TestReprHtml:
    def test_group_renders(self):
        r = _sample_result(_sample_group())
        html_out = r._repr_html_()
        assert "EvaluationResult" in html_out
        assert "mom_12_1" in html_out
        assert "ic" in html_out

    def test_no_role_column_in_html(self):
        r = _sample_result(_sample_group())
        assert "primary" not in r._repr_html_()
        assert "diagnostic" not in r._repr_html_()

    def test_renders_warnings_when_present(self):
        warnings = [Warning(code=WarningCode.FEW_ASSETS, source="ic", message="thin")]
        r = _sample_result(_sample_group(), warnings=warnings)
        html_out = r._repr_html_()
        assert "warnings" in html_out
        assert WarningCode.FEW_ASSETS.value in html_out

    def test_no_warnings_block_when_empty(self):
        r = _sample_result(_sample_group())
        assert "summary>warnings" not in r._repr_html_()


class TestMetricResultNameField:
    def test_default_empty(self):
        out = MetricResult(value=1.0)
        assert out.name == ""

    def test_carries_name(self):
        out = MetricResult(value=0.1, name="ic")
        assert out.name == "ic"
