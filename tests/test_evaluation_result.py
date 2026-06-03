"""``EvaluationResult`` / ``MetricResultGroup`` dataclasses + serialisation (#441)."""

from __future__ import annotations

import json

import polars as pl
import pytest
from factrix import (
    EvaluationResult,
    MetricResult,
    MetricResultGroup,
    Warning,
    WarningCode,
)
from factrix._axis import DataStructure, FactorDensity, FactorScope
from factrix._metric_index import spec_by_name


@pytest.fixture
def ic_spec():
    return spec_by_name()["ic"]


@pytest.fixture
def ic_ir_spec():
    return spec_by_name()["ic_ir"]


def _sample_group(ic_spec, ic_ir_spec) -> MetricResultGroup:
    ic_out = MetricResult(
        value=0.05,
        p=0.012,
        n_obs=100,
        stat=2.5,
        metadata={"p_value": 0.012},
        spec=ic_spec,
    )
    ic_ir_out = MetricResult(
        value=0.42,
        n_obs=100,
        spec=ic_ir_spec,
    )
    return MetricResultGroup(
        applicable=[ic_spec, ic_ir_spec],
        primary=[ic_spec],
        diagnostic=[ic_ir_spec],
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
    def test_dict_like_access(self, ic_spec, ic_ir_spec):
        g = _sample_group(ic_spec, ic_ir_spec)
        assert "ic" in g
        assert "missing" not in g
        assert g["ic"].value == 0.05
        assert len(g) == 2
        assert set(g.keys()) == {"ic", "ic_ir"}
        assert {o.spec.name for o in g.values()} == {"ic", "ic_ir"}
        assert {k for k, _ in g.items()} == {"ic", "ic_ir"}
        assert list(iter(g)) == ["ic", "ic_ir"]

    def test_partition_lists_carry_specs(self, ic_spec, ic_ir_spec):
        g = _sample_group(ic_spec, ic_ir_spec)
        assert g.primary == [ic_spec]
        assert g.diagnostic == [ic_ir_spec]
        assert g.applicable == [ic_spec, ic_ir_spec]


class TestEvaluationResultToFrame:
    def test_schema_and_dtypes(self, ic_spec, ic_ir_spec):
        r = _sample_result(_sample_group(ic_spec, ic_ir_spec))
        df = r.to_frame()
        assert df.columns == [
            "factor",
            "n_assets",
            "metric_name",
            "value",
            "p",
            "stat",
            "n_obs",
            "warning_codes",
        ]
        assert df.schema["value"] == pl.Float64
        assert df.schema["p"] == pl.Float64
        assert df.schema["n_obs"] == pl.Int64
        assert df.schema["warning_codes"] == pl.List(pl.Utf8)
        assert df.height == 2

    def test_short_circuit_row_is_null(self, ic_spec, ic_ir_spec):
        bad = MetricResult(value=float("nan"), spec=ic_spec)
        g = MetricResultGroup(
            applicable=[ic_spec], primary=[ic_spec], diagnostic=[], outputs={"ic": bad}
        )
        df = _sample_result(g).to_frame()
        row = df.row(0, named=True)
        assert row["value"] is None
        assert row["p"] is None

    def test_warning_codes_filter_by_source(self, ic_spec, ic_ir_spec):
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
        r = _sample_result(_sample_group(ic_spec, ic_ir_spec), warnings=warnings)
        df = r.to_frame()
        ic_row = df.filter(pl.col("metric_name") == "ic").row(0, named=True)
        ic_ir_row = df.filter(pl.col("metric_name") == "ic_ir").row(0, named=True)
        assert ic_row["warning_codes"] == [WarningCode.SMALL_CROSS_SECTION_N.value]
        assert ic_ir_row["warning_codes"] == []

    def test_metric_name_uses_spec_name(self):
        fm_spec = spec_by_name()["fm_beta"]
        out = MetricResult(value=0.01, spec=fm_spec)
        g = MetricResultGroup(
            applicable=[fm_spec],
            primary=[fm_spec],
            diagnostic=[],
            outputs={fm_spec.name: out},
        )
        df = _sample_result(g).to_frame()
        assert df.row(0, named=True)["metric_name"] == fm_spec.name


class TestEvaluationResultToDict:
    def test_round_trips_through_json(self, ic_spec, ic_ir_spec):
        warnings = [
            Warning(
                code=WarningCode.SMALL_CROSS_SECTION_N, source="ic", message="thin"
            ),
        ]
        r = _sample_result(_sample_group(ic_spec, ic_ir_spec), warnings=warnings)
        d = r.to_dict()
        encoded = json.dumps(d)
        back = json.loads(encoded)
        assert back["factor"] == "mom_12_1"
        assert back["cell"]["scope"] == "individual"
        assert back["cell"]["density"] == "dense"
        assert back["cell"]["structure"] == "panel"
        assert back["n_obs"] == 100
        assert back["metrics"]["ic"]["p"] == 0.012
        assert back["metrics_partition"]["primary"] == ["ic"]
        assert back["metrics_partition"]["diagnostic"] == ["ic_ir"]
        assert back["warnings"][0]["code"] == WarningCode.SMALL_CROSS_SECTION_N.value
        assert back["plan"] == "1. ic [per-factor]"

    def test_nonfinite_floats_become_null(self, ic_spec, ic_ir_spec):
        bad = MetricResult(
            value=float("nan"),
            p=float("nan"),
            stat=float("inf"),
            metadata={"p_value": float("nan")},
            spec=ic_spec,
        )
        g = MetricResultGroup(
            applicable=[ic_spec], primary=[ic_spec], diagnostic=[], outputs={"ic": bad}
        )
        d = _sample_result(g).to_dict()
        assert d["metrics"]["ic"]["value"] is None
        assert d["metrics"]["ic"]["stat"] is None
        assert d["metrics"]["ic"]["p"] is None
        json.dumps(d)


class TestReprHtml:
    def test_group_renders(self, ic_spec, ic_ir_spec):
        # MetricResultGroup itself ships only dict-like access; HTML
        # lives on the bundle. Smoke-test the bundle render.
        r = _sample_result(_sample_group(ic_spec, ic_ir_spec))
        html_out = r._repr_html_()
        assert "EvaluationResult" in html_out
        assert "mom_12_1" in html_out
        assert "ic" in html_out

    def test_renders_warnings_when_present(self, ic_spec, ic_ir_spec):
        warnings = [
            Warning(code=WarningCode.SMALL_CROSS_SECTION_N, source="ic", message="thin")
        ]
        r = _sample_result(_sample_group(ic_spec, ic_ir_spec), warnings=warnings)
        html_out = r._repr_html_()
        assert "warnings" in html_out
        assert WarningCode.SMALL_CROSS_SECTION_N.value in html_out

    def test_no_warnings_block_when_empty(self, ic_spec, ic_ir_spec):
        r = _sample_result(_sample_group(ic_spec, ic_ir_spec))
        assert "summary>warnings" not in r._repr_html_()


class TestMetricResultSpecBackref:
    def test_default_none(self):
        out = MetricResult(value=1.0)
        assert out.spec is None

    def test_carries_spec(self, ic_spec):
        out = MetricResult(value=0.1, spec=ic_spec)
        assert out.spec is ic_spec
