"""``fx.evaluate`` dict[str, Metric] API -- labels, strict, data-stamped
forward_periods + by-value DAG dedup."""

from __future__ import annotations

import math

import factrix as fx
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix.metrics import (
    caar,
    clustering_hhi,
    ic,
    ic_ir,
    ic_trend,
    oos_decay,
    positive_rate,
)


def _panel(n_assets: int = 20, n_dates: int = 80, *, with_price: bool = True):
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
    if not with_price and "price" in panel.columns:
        panel = panel.drop("price")
    return panel


def _eval(metrics, panel=None, **kw):
    panel = panel if panel is not None else _panel()
    return fx.evaluate(
        panel, metrics=metrics, factor_cols=["factor"], forward_periods=5, **kw
    )


class TestLabelKeying:
    def test_results_key_by_factor_name(self):
        results = _eval({"my_ic": ic(), "my_ir": ic_ir()})
        assert "factor" in results
        er = results["factor"]
        assert list(er.metrics) == ["my_ic", "my_ir"]
        assert er.metrics["my_ic"].name == "my_ic"

    def test_to_frame_uses_labels(self):
        results = _eval({"my_ic": ic()})
        assert results["factor"].to_frame()["metric_name"].to_list() == ["my_ic"]

    def test_returns_dict_keyed_by_factor(self):
        panel = _panel()
        panel = panel.with_columns((pl.col("factor") * -1).alias("factor_b"))
        results = fx.evaluate(
            panel,
            metrics={"ic": ic()},
            factor_cols=["factor", "factor_b"],
            forward_periods=5,
        )
        assert set(results.keys()) == {"factor", "factor_b"}

    def test_dict_insertion_order_matches_factor_cols(self):
        panel = _panel()
        panel = panel.with_columns((pl.col("factor") * -1).alias("factor_b"))
        results = fx.evaluate(
            panel,
            metrics={"ic": ic()},
            factor_cols=["factor", "factor_b"],
            forward_periods=5,
        )
        assert list(results.keys()) == ["factor", "factor_b"]


class TestPanelStats:
    def test_n_periods_and_n_pairs_populated(self):
        er = _eval({"ic": ic()})["factor"]
        assert er.n_periods > 0
        assert er.n_pairs > 0
        assert er.n_pairs >= er.n_periods

    def test_no_n_obs_on_bundle(self):
        er = _eval({"ic": ic()})["factor"]
        assert not hasattr(er, "n_obs")


class TestMetricsValidation:
    def test_bare_class_gives_targeted_error(self):
        with pytest.raises(UserInputError, match="call it: ic"):
            _eval({"a": ic})

    def test_list_is_rejected(self):
        with pytest.raises(UserInputError, match="dict\\[str, Metric\\]"):
            _eval([ic()])

    def test_spec_value_rejected(self):
        from factrix._metric_index import spec_by_name

        with pytest.raises(UserInputError, match="metric instance"):
            _eval({"a": spec_by_name()["ic"]})

    def test_overview_rejected_with_guidance(self):
        with pytest.raises(UserInputError, match="overview catalog"):
            _eval(fx.list_metrics())

    def test_non_numeric_factor_column_rejected_at_entry(self):
        panel = _panel().with_columns(pl.lit("buy").alias("factor_text"))
        with pytest.raises(UserInputError, match="factor columns to be numeric"):
            fx.evaluate(
                panel,
                metrics={"ic": ic()},
                factor_cols=["factor_text"],
                forward_periods=5,
            )


class TestByValueDedup:
    def test_same_config_alias_is_one_node_two_labels(self):
        er = _eval({"a": ic_ir(), "b": ic_ir()})["factor"]
        assert set(er.metrics) == {"a", "b"}

    def test_different_config_coexists(self):
        from factrix.metrics import quantile_spread

        er = _eval(
            {"a": quantile_spread(n_groups=5), "b": quantile_spread(n_groups=10)}
        )["factor"]
        assert set(er.metrics) == {"a", "b"}

    def test_horizon_injected_from_data_into_every_metric(self):
        # forward_periods is the data's stamped overlap horizon (5 here),
        # injected into every metric -- there is no per-metric override.
        er = fx.evaluate(
            _panel(),
            metrics={
                "t": ic(),
                "nw": ic(inference=fx.inference.NEWEY_WEST),
            },
            factor_cols=["factor"],
        )["factor"]
        assert er.metrics["t"].metadata["forward_periods"] == 5
        assert er.metrics["nw"].metadata["forward_periods"] == 5
        assert er.forward_periods == 5

    def test_shared_producer_runs_once_across_configs(self):
        # Two ic configs (different inference) share one compute_ic producer.
        er = fx.evaluate(
            _panel(),
            metrics={
                "t": ic(),
                "nw": ic(inference=fx.inference.NEWEY_WEST),
            },
            factor_cols=["factor"],
        )["factor"]
        assert er.plan.count("compute_ic [batchable]") == 1

    def test_metric_level_forward_periods_is_rejected(self):
        # The mixed-horizon usage {"ic_5d": ic(), "ic_20d": ic(forward_periods=20)}
        # is gone: forward_periods is not a metric parameter.
        with pytest.raises(fx.UserInputError):
            ic(forward_periods=20)


class TestForwardPeriodsContract:
    """The overlap horizon is a property of the data (stamp), declared once for
    a self-attached panel (path B), never a per-metric knob."""

    def test_horizon_read_from_stamp_when_omitted(self):
        er = fx.evaluate(_panel(), metrics={"ic": ic()}, factor_cols=["factor"])[
            "factor"
        ]
        assert er.forward_periods == 5  # read from the compute_forward_return stamp

    def test_self_attached_panel_without_declaration_raises(self):
        from factrix._data_input import _FORWARD_PERIODS_COL

        unstamped = _panel().drop(_FORWARD_PERIODS_COL)
        with pytest.raises(UserInputError, match="overlap horizon"):
            fx.evaluate(unstamped, metrics={"ic": ic()}, factor_cols=["factor"])

    def test_self_attached_panel_with_declaration_runs(self):
        from factrix._data_input import _FORWARD_PERIODS_COL

        unstamped = _panel().drop(_FORWARD_PERIODS_COL)
        er = fx.evaluate(
            unstamped, metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
        )["factor"]
        assert er.forward_periods == 5

    def test_declaration_conflicting_with_stamp_raises(self):
        with pytest.raises(UserInputError, match="stamp"):
            fx.evaluate(
                _panel(),
                metrics={"ic": ic()},
                factor_cols=["factor"],
                forward_periods=20,
            )

    def test_stamp_column_never_leaks_into_outputs(self):
        from factrix._data_input import _FORWARD_PERIODS_COL

        er = fx.evaluate(_panel(), metrics={"ic": ic()}, factor_cols=["factor"])[
            "factor"
        ]
        assert _FORWARD_PERIODS_COL not in er.to_frame().columns
        assert _FORWARD_PERIODS_COL not in er.metrics["ic"].metadata


class TestStrict:
    def test_strict_true_raises_on_inapplicable(self):
        thin = _panel(n_dates=12)
        with pytest.raises(UserInputError) as exc:
            fx.evaluate(
                thin, metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
            )
        msg = str(exc.value)
        assert "inapplicable" in msg
        assert "strict=False" in msg
        assert "is_applicable/reason" in msg

    def test_strict_false_keeps_nan(self):
        thin = _panel(n_dates=12)
        er = fx.evaluate(
            thin,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )["factor"]
        assert er.metrics["ic"].value != er.metrics["ic"].value  # NaN
        status = er.to_frame().row(0, named=True)
        assert status["is_applicable"] is False
        assert status["reason"] == "insufficient_ic_periods"

    def test_strict_true_keeps_not_applicable_without_aborting(self):
        # A discrete +/-k signal makes a continuous-magnitude metric
        # not_applicable; under default strict=True the battery must still
        # return the applicable metric rather than raising on the inapplicable
        # one (type-routing verdict, not a failure).
        from factrix.metrics import event_ic
        from factrix.metrics.caar import caar

        panel = _panel(n_assets=30, n_dates=120).with_columns(
            (pl.col("factor") > 0.5).cast(pl.Float64).alias("factor")
        )
        with pytest.warns(UserWarning):  # caar's thin-sample advisory
            er = fx.evaluate(
                panel,
                metrics={"caar": caar(), "event_ic": event_ic()},
                factor_cols=["factor"],
                forward_periods=5,
            )["factor"]
        assert er.metrics["caar"].is_applicable
        assert not math.isnan(er.metrics["caar"].value)
        assert er.metrics["event_ic"].is_applicable is False
        assert er.metrics["event_ic"].reason == "not_applicable_discrete_signal"


class TestStrictCellSoftening:
    def _single_asset_panel(self):
        panel = _panel(n_assets=20, n_dates=80)
        first = panel["asset_id"][0]
        return panel.filter(pl.col("asset_id") == first)

    def test_strict_true_raises_on_cell_mismatch(self):
        with pytest.raises(fx.IncompatibleAxisError, match=r"cell=.*PANEL"):
            fx.evaluate(
                self._single_asset_panel(),
                metrics={"ic": ic()},
                factor_cols=["factor"],
                forward_periods=5,
            )

    def test_strict_false_softens_to_nan(self):
        er = fx.evaluate(
            self._single_asset_panel(),
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )["factor"]
        assert math.isnan(er.metrics["ic"].value)
        assert er.metrics["ic"].metadata["reason"] == "structure_mismatch"

    def test_strict_false_short_circuits_without_executing_metric(self):
        # #631: clustering_hhi does not self-guard structure on TIMESERIES
        # data it would compute a numerically real (but invalid) value. The
        # pre-flight gate must intercept it under strict=False: NaN value,
        # structure_mismatch reason (legacy code proving the metric never ran),
        # and a matching warning. p_value is None (no test was run).
        er = fx.evaluate(
            self._single_asset_panel(),
            metrics={"hhi": clustering_hhi()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )["factor"]
        m = er.metrics["hhi"]
        assert math.isnan(m.value)
        assert m.metadata["reason"] == "structure_mismatch"
        assert m.metadata["declared_cell"] == "(*, SPARSE, PANEL)"
        assert m.metadata["detected_cell"] == "(common, dense, timeseries)"
        assert m.metadata["declared_data_structure"] == "panel"
        assert m.metadata["data_structure"] == "timeseries"
        assert m.p_value is None
        assert (
            next(w for w in er.warnings if w.source == "hhi").code.value
            == "structure_mismatch"
        )

    def test_strict_false_mismatch_does_not_block_applicable_metric(self):
        # A cell-mismatched label is dropped from the DAG; sibling
        # applicable metrics in the same call still run, and the returned
        # dict preserves the caller's request order across both.
        panel = _panel(n_assets=20, n_dates=80)  # PANEL data
        single = panel.filter(pl.col("asset_id") == panel["asset_id"][0])
        er = fx.evaluate(
            panel,
            metrics={"hhi": clustering_hhi(), "ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )["factor"]
        assert list(er.metrics.keys()) == ["hhi", "ic"]
        # On PANEL data neither is mismatched, so ic computes a real value.
        assert not math.isnan(er.metrics["ic"].value)
        # And on single-asset data both PANEL metrics are mismatched.
        er2 = fx.evaluate(
            single,
            metrics={"hhi": clustering_hhi(), "ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )["factor"]
        assert er2.metrics["hhi"].metadata["reason"] == "structure_mismatch"
        assert er2.metrics["ic"].metadata["reason"] == "structure_mismatch"

    def test_series_diagnostics_do_not_run_on_single_asset_dense_panel(self):
        single = self._single_asset_panel()
        info = fx.inspect_data(single, factor_cols=["factor"])
        verdicts = {m.name: m for m in info.metrics}
        for name in ("positive_rate", "oos_decay", "ic_trend"):
            assert verdicts[name].usable is False
            assert any("cell mismatch" in b for b in verdicts[name].blockers)

        er = fx.evaluate(
            single,
            metrics={
                "positive_rate": positive_rate(),
                "oos_decay": oos_decay(),
                "ic_trend": ic_trend(),
            },
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )["factor"]

        for name in ("positive_rate", "oos_decay", "ic_trend"):
            out = er.metrics[name]
            assert math.isnan(out.value)
            assert out.metadata["reason"] == "structure_mismatch"

    def test_series_diagnostics_run_on_panel_ic_series(self):
        er = fx.evaluate(
            _panel(n_assets=20, n_dates=120),
            metrics={
                "positive_rate": positive_rate(),
                "oos_decay": oos_decay(),
                "ic_trend": ic_trend(),
            },
            factor_cols=["factor"],
            forward_periods=5,
        )["factor"]

        assert not math.isnan(er.metrics["positive_rate"].value)
        assert not math.isnan(er.metrics["oos_decay"].value)
        assert not math.isnan(er.metrics["ic_trend"].value)

    def test_dense_factor_does_not_run_sparse_event_metric(self):
        panel = _panel(n_assets=20, n_dates=120)
        with pytest.raises(fx.IncompatibleAxisError) as exc:
            fx.evaluate(
                panel,
                metrics={"caar": caar()},
                factor_cols=["factor"],
                forward_periods=5,
            )
        msg = str(exc.value)
        assert "SPARSE" in msg
        assert "zero non-event" in msg

        er = fx.evaluate(
            panel,
            metrics={"caar": caar()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )["factor"]
        assert math.isnan(er.metrics["caar"].value)
        assert er.metrics["caar"].metadata["reason"] == "structure_mismatch"
        assert er.metrics["caar"].metadata["data_density"] == "dense"
        assert "zero non-event" in er.metrics["caar"].metadata["guidance"]
        assert "zero non-event" in next(
            w.message for w in er.warnings if w.source == "caar"
        )
        assert er.metrics["caar"].p_value is None

    def test_explicit_sparse_metric_runs_on_low_zero_ratio_event_signal(self):
        panel = _panel(n_assets=20, n_dates=220).with_columns(
            pl.when(pl.int_range(0, pl.len()) % 5 < 2)
            .then(0.0)
            .otherwise(1.0)
            .alias("factor")
        )

        er = fx.evaluate(
            panel,
            metrics={"caar": caar()},
            factor_cols=["factor"],
            forward_periods=5,
        )["factor"]

        assert er.cell[1] is fx.FactorDensity.DENSE
        assert not math.isnan(er.metrics["caar"].value)
        warn = next(
            w for w in er.warnings if w.code is fx.WarningCode.FREQUENT_EVENT_SIGNAL
        )
        assert warn.source == "caar"
        assert "sparse_ratio=0.40" in warn.message

    def test_sparse_factor_does_not_run_dense_ic_metric(self):
        raw = fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        er = fx.evaluate(
            panel,
            metrics={"ic": ic(), "caar": caar()},
            factor_cols=["factor"],
            strict=False,
        )["factor"]
        assert er.cell[1] is fx.FactorDensity.SPARSE
        assert er.metrics["ic"].metadata["reason"] == "structure_mismatch"
        assert er.metrics["caar"].reason is None
        assert not math.isnan(er.metrics["caar"].value)

    def test_mixed_factor_cells_route_in_separate_groups(self):
        panel = _panel(n_assets=20, n_dates=120).with_columns(
            pl.when(pl.int_range(0, pl.len()) % 20 == 0)
            .then(1.0)
            .otherwise(0.0)
            .alias("event_factor")
        )
        results = fx.evaluate(
            panel,
            metrics={"ic": ic(), "caar": caar()},
            factor_cols=["factor", "event_factor"],
            forward_periods=5,
            strict=False,
        )
        dense = results["factor"]
        sparse = results["event_factor"]

        assert dense.cell[1] is fx.FactorDensity.DENSE
        assert sparse.cell[1] is fx.FactorDensity.SPARSE
        assert dense.metrics["caar"].metadata["reason"] == "structure_mismatch"
        assert not math.isnan(dense.metrics["ic"].value)
        assert sparse.metrics["ic"].metadata["reason"] == "structure_mismatch"
        assert not math.isnan(sparse.metrics["caar"].value)


class TestEntryConsistency:
    """A metric's sample gating must be identical whether it is called
    directly or routed through ``evaluate()`` -- the declare-once-enforce
    contract. The shared helpers read the same declared floor on both paths.
    ``ic_ir`` exercises all three tiers (short-circuit / warn / clean) off a
    single periods floor, with ``evaluate`` computing the same upstream IC.
    """

    @staticmethod
    def _ic_df(panel):
        from factrix.metrics.ic import compute_ic

        return compute_ic(panel)["factor"]

    def _direct_and_via(self, panel):
        direct = ic_ir(self._ic_df(panel))
        via = _eval({"ir": ic_ir()}, panel, strict=False)["factor"].metrics["ir"]
        return direct, via

    def test_short_circuit_matches(self):
        # ~15 IC rows < MIN_PERIODS_HARD=20: both entry points short-circuit
        # to NaN on the same floor with the same reason.
        direct, via = self._direct_and_via(_panel(n_assets=20, n_dates=20))
        assert math.isnan(direct.value) and math.isnan(via.value)
        assert direct.metadata["reason"] == via.metadata["reason"]
        assert direct.metadata["reason"] == "insufficient_ic_periods"

    def test_warn_code_matches(self):
        # ~25 IC rows: clears the min floor (20) but below the warn floor
        # (30) -> both paths attach the same degraded-tier code.
        from factrix._codes import WarningCode

        direct, via = self._direct_and_via(_panel(n_assets=20, n_dates=30))
        assert not math.isnan(direct.value)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in direct.warning_codes
        assert direct.warning_codes == via.warning_codes

    def test_clean_tier_matches(self):
        # ~75 IC rows >= warn floor: both paths return the same value, no code.
        direct, via = self._direct_and_via(_panel(n_assets=20, n_dates=80))
        assert not math.isnan(direct.value)
        assert direct.value == via.value
        assert direct.warning_codes == via.warning_codes == ()
