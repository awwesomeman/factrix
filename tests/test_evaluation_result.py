"""``EvaluationResult`` / ``MetricResultGroup`` dataclasses + serialisation (#441)."""

from __future__ import annotations

import json

import polars as pl
from factrix import (
    EvaluationResult,
    MetricResult,
    MetricResultGroup,
    Warning,
    WarningCode,
)
from factrix._axis import DataStructure, FactorDensity, FactorScope


def _sample_group() -> MetricResultGroup:
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
    return MetricResultGroup(
        applicable=["ic", "ic_ir"],
        primary=["ic"],
        diagnostic=["ic_ir"],
        outputs={"ic": ic_out, "ic_ir": ic_ir_out},
    )


def _sample_result(
    group: MetricResultGroup, warnings=None, plan: str = "1. ic [per-factor]"
) -> EvaluationResult:
    return EvaluationResult(
        factor="mom_12_1",
        cell=(FactorScope.INDIVIDUAL, FactorDensity.DENSE, DataStructure.PANEL),
        forward_periods=5,
        n_obs=100,
        n_assets=25,
        metrics=group,
        plan=plan,
        warnings=warnings or [],
    )


class TestMetricResult:
    def test_dict_like_access(self):
        g = _sample_group()
        assert "ic" in g
        assert "missing" not in g
        assert g["ic"].value == 0.05
        assert len(g) == 2
        assert set(g.keys()) == {"ic", "ic_ir"}
        assert {o.name for o in g.values()} == {"ic", "ic_ir"}
        assert {k for k, _ in g.items()} == {"ic", "ic_ir"}
        assert list(iter(g)) == ["ic", "ic_ir"]

    def test_partition_lists_are_names(self):
        g = _sample_group()
        assert g.primary == ["ic"]
        assert g.diagnostic == ["ic_ir"]
        assert g.applicable == ["ic", "ic_ir"]


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
            "warning_codes",
        ]
        assert df.schema["value"] == pl.Float64
        assert df.schema["p_value"] == pl.Float64
        assert df.schema["n_obs"] == pl.Int64
        assert df.schema["warning_codes"] == pl.List(pl.Utf8)
        assert df.height == 2

    def test_short_circuit_row_is_null(self):
        bad = MetricResult(value=float("nan"), name="ic")
        g = MetricResultGroup(
            applicable=["ic"], primary=["ic"], diagnostic=[], outputs={"ic": bad}
        )
        df = _sample_result(g).to_frame()
        row = df.row(0, named=True)
        assert row["value"] is None
        assert row["p_value"] is None

    def test_warning_codes_filter_by_source(self):
        warnings = [
            Warning(
                code=WarningCode.SMALL_CROSS_SECTION_N, source="ic", message="thin"
            ),
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
        assert ic_row["warning_codes"] == [WarningCode.SMALL_CROSS_SECTION_N.value]
        assert ic_ir_row["warning_codes"] == []

    def test_metric_name_from_name_field(self):
        out = MetricResult(value=0.01, name="fm_beta")
        g = MetricResultGroup(
            applicable=["fm_beta"],
            primary=["fm_beta"],
            diagnostic=[],
            outputs={"fm_beta": out},
        )
        df = _sample_result(g).to_frame()
        assert df.row(0, named=True)["metric_name"] == "fm_beta"


class TestEvaluationResultToDict:
    def test_round_trips_through_json(self):
        warnings = [
            Warning(
                code=WarningCode.SMALL_CROSS_SECTION_N, source="ic", message="thin"
            ),
        ]
        r = _sample_result(_sample_group(), warnings=warnings)
        d = r.to_dict()
        encoded = json.dumps(d)
        back = json.loads(encoded)
        assert back["factor"] == "mom_12_1"
        assert back["cell"]["scope"] == "individual"
        assert back["cell"]["density"] == "dense"
        assert back["cell"]["structure"] == "panel"
        assert back["n_obs"] == 100
        assert back["metrics"]["ic"]["p_value"] == 0.012
        assert back["metrics_partition"]["primary"] == ["ic"]
        assert back["metrics_partition"]["diagnostic"] == ["ic_ir"]
        assert back["warnings"][0]["code"] == WarningCode.SMALL_CROSS_SECTION_N.value
        assert back["plan"] == "1. ic [per-factor]"

    def test_nonfinite_floats_become_null(self):
        bad = MetricResult(
            value=float("nan"),
            p_value=float("nan"),
            stat=float("inf"),
            metadata={"p_value": float("nan")},
            name="ic",
        )
        g = MetricResultGroup(
            applicable=["ic"], primary=["ic"], diagnostic=[], outputs={"ic": bad}
        )
        d = _sample_result(g).to_dict()
        assert d["metrics"]["ic"]["value"] is None
        assert d["metrics"]["ic"]["stat"] is None
        assert d["metrics"]["ic"]["p_value"] is None
        json.dumps(d)


class TestReprHtml:
    def test_group_renders(self):
        # MetricResultGroup itself ships only dict-like access; HTML
        # lives on the bundle. Smoke-test the bundle render.
        r = _sample_result(_sample_group())
        html_out = r._repr_html_()
        assert "EvaluationResult" in html_out
        assert "mom_12_1" in html_out
        assert "ic" in html_out

    def test_renders_warnings_when_present(self):
        warnings = [
            Warning(code=WarningCode.SMALL_CROSS_SECTION_N, source="ic", message="thin")
        ]
        r = _sample_result(_sample_group(), warnings=warnings)
        html_out = r._repr_html_()
        assert "warnings" in html_out
        assert WarningCode.SMALL_CROSS_SECTION_N.value in html_out

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
