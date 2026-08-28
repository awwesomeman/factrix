"""``EvaluationResult`` / ``MetricResult`` dataclasses + serialisation."""

from __future__ import annotations

import dataclasses
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
        alternative="two-sided",
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
        overlap_periods=5,
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


class TestMetricAccessor:
    def test_metric_returns_result(self):
        r = _sample_result(_sample_group())
        assert r.metric("ic").value == 0.05
        assert r.metric("ic") is r.metrics["ic"]

    def test_metric_miss_lists_available_labels(self):
        r = _sample_result(_sample_group())
        with pytest.raises(KeyError, match=r"no metric 'sharpe'.*available: ic, ic_ir"):
            r.metric("sharpe")


class TestEvaluationResultToFrame:
    def test_schema_and_dtypes(self):
        r = _sample_result(_sample_group())
        df = r.to_frame()
        assert df.columns == [
            "factor",
            "forward_periods",
            "overlap_periods",
            "n_assets",
            "metric_name",
            "value",
            "p_value",
            "alternative",
            "stat",
            "n_obs",
            "n_obs_axis",
            "is_applicable",
            "reason",
            "warning_codes",
        ]
        assert df.schema["value"] == pl.Float64
        assert df.schema["p_value"] == pl.Float64
        assert df.schema["alternative"] == pl.Utf8
        assert df.schema["n_obs"] == pl.Int64
        assert df.schema["n_obs_axis"] == pl.Utf8
        assert df.schema["is_applicable"] == pl.Boolean
        assert df.schema["reason"] == pl.Utf8
        assert df.schema["warning_codes"] == pl.List(pl.Utf8)
        assert df.schema["forward_periods"] == pl.Int64
        assert df.height == 2

    def test_carries_hypothesis_identity(self):
        """(factor, overlap_periods, *params) — the same tuple to_dict and
        compare carry. Without it an evaluate_horizons stack is unreadable."""
        r = dataclasses.replace(
            _sample_result(_sample_group()), params={"universe": "tw50", "k": 3}
        )
        df = r.to_frame()
        assert df.columns[:4] == ["factor", "forward_periods", "universe", "k"]
        row = df.row(0, named=True)
        assert row["forward_periods"] == 5
        assert (row["universe"], row["k"]) == ("tw50", 3)

    def test_horizon_stack_is_distinguishable(self):
        frames = [
            dataclasses.replace(
                _sample_result(_sample_group()), forward_periods=h
            ).to_frame()
            for h in (1, 5, 20)
        ]
        stacked = pl.concat(frames)
        assert sorted(set(stacked["forward_periods"])) == [1, 5, 20]

    def test_params_key_colliding_with_fixed_column_raises(self):
        r = dataclasses.replace(
            _sample_result(_sample_group()), params={"value": 1, "n_obs": 2}
        )
        with pytest.raises(ValueError, match=r"collide with fixed column"):
            r.to_frame()

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
        assert row["alternative"] is None

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
        assert back["metrics"]["ic"]["alternative"] == "two-sided"
        assert back["metrics"]["ic"]["is_applicable"] is True
        assert back["metrics"]["ic"]["reason"] is None
        assert back["warnings"][0]["code"] == WarningCode.FEW_ASSETS.value
        assert back["plan"] == "1. ic [per-factor]"

    def test_nonfinite_floats_become_null(self):
        # `stat` may be NaN ("ran, could not form the statistic") but never
        # ±Inf — see TestMetricResultFieldContract.
        bad = MetricResult(
            value=float("nan"),
            stat=float("nan"),
            metadata={"p_value": float("nan"), "se": float("inf")},
            name="ic",
        )
        g = MappingProxyType({"ic": bad})
        d = _sample_result(g).to_dict()
        assert d["metrics"]["ic"]["value"] is None
        assert d["metrics"]["ic"]["stat"] is None
        assert d["metrics"]["ic"]["p_value"] is None
        assert d["metrics"]["ic"]["metadata"]["se"] is None
        json.dumps(d)


class TestMetricResultPValueContract:
    @pytest.mark.parametrize("p_value", [float("nan"), float("inf"), -0.1, 1.1])
    def test_rejects_invalid_p_value(self, p_value: float):
        with pytest.raises(ValueError, match=r"finite.*\[0, 1\]"):
            MetricResult(
                value=0.0,
                p_value=p_value,
                alternative="two-sided",
            )

    def test_rejects_p_value_without_alternative(self):
        with pytest.raises(ValueError, match="both be provided"):
            MetricResult(value=0.0, p_value=0.5)

    def test_rejects_alternative_without_p_value(self):
        with pytest.raises(ValueError, match="both be provided"):
            MetricResult(value=0.0, alternative="greater")


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


class TestMetricResultFieldContract:
    """``__post_init__`` guards on fields other than ``p_value``."""

    def test_rejects_unknown_alternative(self):
        with pytest.raises(ValueError, match="alternative must be one of"):
            MetricResult(value=0.0, p_value=0.5, alternative="bigger")  # type: ignore[arg-type]

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_accepts_declared_alternatives(self, alternative):
        out = MetricResult(value=0.0, p_value=0.5, alternative=alternative)
        assert out.alternative == alternative

    def test_rejects_negative_n_obs(self):
        with pytest.raises(ValueError, match="n_obs must be non-negative"):
            MetricResult(value=0.0, n_obs=-1)

    def test_accepts_zero_n_obs(self):
        assert MetricResult(value=float("nan"), n_obs=0).n_obs == 0

    @pytest.mark.parametrize("p_value", [True, False])
    def test_rejects_bool_p_value(self, p_value: bool):
        with pytest.raises(ValueError, match="not a bool"):
            MetricResult(value=0.0, p_value=p_value, alternative="two-sided")

    @pytest.mark.parametrize("stat", [float("inf"), float("-inf")])
    def test_rejects_infinite_stat(self, stat: float):
        with pytest.raises(ValueError, match="stat must be finite"):
            MetricResult(value=0.0, stat=stat)

    def test_allows_nan_stat(self):
        """A NaN stat is the "ran but could not form the statistic" marker."""
        import math

        out = MetricResult(value=0.0, stat=float("nan"))
        assert math.isnan(out.stat)


class TestReprHtmlValueTolerance:
    def test_non_float_value_does_not_raise(self):
        """``value`` is annotated float but structural metrics carry payloads;
        ``math.isnan`` on a non-float used to blow up the whole repr."""
        g = MappingProxyType(
            {
                "structural": MetricResult(value="selected", name="structural"),  # type: ignore[arg-type]
                "missing": MetricResult(value=None, name="missing"),  # type: ignore[arg-type]
            }
        )
        out = _sample_result(g)._repr_html_()
        assert "selected" in out
        assert "null" in out

    def test_nan_value_renders_null(self):
        g = MappingProxyType({"ic": MetricResult(value=float("nan"), name="ic")})
        assert "null" in _sample_result(g)._repr_html_()


class TestToFrameMetadataExport:
    """#883 — ``to_frame(metadata=)`` carries named estimator scalars along.

    The fixed schema stays the default; a requested key becomes one column
    named after it, ``null`` on metrics that do not carry it, and nested
    values / reserved names are refused rather than silently flattened.
    """

    @staticmethod
    def _result():
        spread = MetricResult(
            value=0.001,
            n_obs=80,
            name="quantile_spread",
            metadata={"n_groups": 2, "mean_tail_size": 3.0, "legs": [0.1, -0.2]},
        )
        turnover = MetricResult(
            value=0.25,
            n_obs=79,
            name="notional_turnover",
            metadata={"n_groups": 2, "rebalance_lag": 1, "mean_tail_size": 3.0},
        )
        return _sample_result(
            MappingProxyType({"spread": spread, "turnover": turnover})
        )

    def test_default_schema_is_unchanged(self):
        frame = self._result().to_frame()
        assert "n_groups" not in frame.columns
        assert frame.columns[-1] == "warning_codes"

    def test_requested_keys_become_trailing_columns_with_nulls_where_absent(self):
        frame = self._result().to_frame(
            metadata=("n_groups", "rebalance_lag", "mean_tail_size")
        )
        assert frame.columns[-3:] == ["n_groups", "rebalance_lag", "mean_tail_size"]
        rows = {r["metric_name"]: r for r in frame.to_dicts()}
        assert rows["quantile_spread"]["n_groups"] == 2
        assert rows["notional_turnover"]["n_groups"] == 2
        assert rows["quantile_spread"]["rebalance_lag"] is None
        assert rows["notional_turnover"]["rebalance_lag"] == 1
        assert rows["notional_turnover"]["mean_tail_size"] == pytest.approx(3.0)
        # Still pairs with the identity block, so stacked frames stay auditable.
        assert set(frame["factor"].to_list()) == {"mom_12_1"}

    def test_nested_values_are_refused_not_flattened(self):
        with pytest.raises(ValueError, match=r"metadata\['legs'\].*quantile_spread"):
            self._result().to_frame(metadata=("legs",))

    def test_reserved_and_repeated_keys_are_refused(self):
        with pytest.raises(ValueError, match="collide"):
            self._result().to_frame(metadata=("value",))
        with pytest.raises(ValueError, match="collide"):
            self._result().to_frame(metadata=("forward_periods",))
        with pytest.raises(ValueError, match="repeated"):
            self._result().to_frame(metadata=("n_groups", "n_groups"))
        with pytest.raises(ValueError, match="sequence"):
            self._result().to_frame(metadata="n_groups")  # type: ignore[arg-type]

    def test_params_keys_are_reserved_too(self):
        res = dataclasses.replace(self._result(), params={"n_groups": 2})
        with pytest.raises(ValueError, match="collide"):
            res.to_frame(metadata=("n_groups",))

    def test_end_to_end_tradability_audit(self):
        """The reporter's case: spread and turnover n_groups checkable from CSV."""
        import factrix as fx
        from factrix.metrics import notional_turnover, quantile_spread
        from factrix.preprocess import compute_forward_return

        raw = fx.datasets.make_cs_panel(n_assets=6, n_dates=120, seed=0)
        panel = compute_forward_return(raw, forward_periods=5)
        out = fx.evaluate(
            panel,
            metrics={
                "spread": quantile_spread(n_groups=2),
                "turnover": notional_turnover(n_groups=2, rebalance_lag=1),
            },
            factor_cols=["factor"],
        )["factor"]
        frame = out.to_frame(metadata=("n_groups", "rebalance_lag", "mean_tail_size"))
        # ``metric_name`` is the caller's label inside ``evaluate``.
        rows = {r["metric_name"]: r for r in frame.to_dicts()}
        assert rows["spread"]["n_groups"] == 2
        assert rows["turnover"]["n_groups"] == 2
        assert rows["turnover"]["rebalance_lag"] == 1
        assert rows["spread"]["rebalance_lag"] is None
        assert rows["turnover"]["mean_tail_size"] == pytest.approx(3.0)
        # The exported cells are scalars, so the frame is CSV-safe once the
        # (pre-existing) list-typed ``warning_codes`` column is set aside.
        text = frame.drop("warning_codes").write_csv()
        assert "turnover" in text and "n_groups" in text.splitlines()[0]
