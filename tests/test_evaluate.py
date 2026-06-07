"""``fx.evaluate`` dict[str, Metric] API — labels, strict, per-instance
forward_periods + by-value DAG dedup."""

from __future__ import annotations

import math

import factrix as fx
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix.metrics import ic, ic_ir, ic_newey_west


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
        assert list(er.metrics.outputs) == ["my_ic", "my_ir"]
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
        assert set(er.metrics.outputs) == {"a", "b"}

    def test_different_config_coexists(self):
        from factrix.metrics import quantile_spread

        er = _eval(
            {"a": quantile_spread(n_groups=5), "b": quantile_spread(n_groups=10)}
        )["factor"]
        assert set(er.metrics.outputs) == {"a", "b"}

    def test_per_instance_forward_periods_override(self):
        er = fx.evaluate(
            _panel(),
            metrics={"fp5": ic_newey_west(), "fp20": ic_newey_west(forward_periods=20)},
            factor_cols=["factor"],
        )["factor"]
        assert er.metrics["fp5"].metadata["forward_periods"] == 5
        assert er.metrics["fp20"].metadata["forward_periods"] == 20

    def test_shared_producer_runs_once_across_configs(self):
        er = fx.evaluate(
            _panel(),
            metrics={"fp5": ic_newey_west(), "fp20": ic_newey_west(forward_periods=20)},
            factor_cols=["factor"],
        )["factor"]
        assert er.plan.count("compute_ic [batchable]") == 1

    def test_top_level_forward_periods_is_default_fallback(self):
        er = fx.evaluate(
            _panel(),
            metrics={
                "dflt": ic_newey_west(),
                "expl": ic_newey_west(forward_periods=10),
            },
            factor_cols=["factor"],
            forward_periods=20,
        )["factor"]
        assert er.metrics["dflt"].metadata["forward_periods"] == 20
        assert er.metrics["expl"].metadata["forward_periods"] == 10
        assert er.forward_periods == 20  # top-level fp stamped on bundle


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
