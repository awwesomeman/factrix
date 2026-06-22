"""``fx.evaluate`` dict[str, Metric] API — labels, strict, data-stamped
forward_periods + by-value DAG dedup."""

from __future__ import annotations

import math

import factrix as fx
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._errors import UserInputError
from factrix.metrics import clustering_hhi, ic, ic_ir


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
        # injected into every metric — there is no per-metric override.
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
        with pytest.raises(UserInputError, match="inapplicable"):
            fx.evaluate(
                thin, metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
            )

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


class TestStrictStructureSoftening:
    def _single_asset_panel(self):
        panel = _panel(n_assets=20, n_dates=80)
        first = panel["asset_id"][0]
        return panel.filter(pl.col("asset_id") == first)

    def test_strict_true_raises_on_structure_mismatch(self):
        with pytest.raises(UserInputError, match="structure"):
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

    def test_strict_false_short_circuits_without_executing_metric(self):
        # #631: clustering_hhi does not self-guard structure — on TIMESERIES
        # data it would compute a numerically real (but invalid) value. The
        # pre-flight gate must intercept it under strict=False: NaN value,
        # structure_mismatch reason (proves the metric never ran), and a
        # STRUCTURE_MISMATCH warning. p_value is None (no test was run).
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
        assert m.metadata["cell_structure"] == "panel"
        assert m.metadata["data_structure"] == "timeseries"
        assert m.p_value is None
        assert (WarningCode.STRUCTURE_MISMATCH, "hhi") in [
            (w.code, w.source) for w in er.warnings
        ]

    def test_strict_false_mismatch_does_not_block_applicable_metric(self):
        # A structure-mismatched label is dropped from the DAG; sibling
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
        # And on single-asset data only hhi is mismatched.
        er2 = fx.evaluate(
            single,
            metrics={"hhi": clustering_hhi(), "ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )["factor"]
        assert er2.metrics["hhi"].metadata["reason"] == "structure_mismatch"


class TestEntryConsistency:
    """A metric's sample gating must be identical whether it is called
    directly or routed through ``evaluate()`` — the declare-once-enforce
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
