"""``fx.evaluate`` dict[str, Metric] API — labels, primary, strict, per-instance
forward_periods + by-value DAG dedup (#497 / #494)."""

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
        # forward_return is already attached; dropping price now makes
        # event-study metrics (caar) short-circuit at evaluate time.
        panel = panel.drop("price")
    return panel


def _eval(metrics, panel=None, **kw):
    panel = panel if panel is not None else _panel()
    return fx.evaluate(
        panel, metrics=metrics, factor_cols=["factor"], forward_periods=5, **kw
    )


class TestLabelKeying:
    def test_results_key_by_user_label(self):
        [er] = _eval({"my_ic": ic(), "my_ir": ic_ir()})
        assert list(er.metrics.outputs) == ["my_ic", "my_ir"]
        assert er.metrics["my_ic"].name == "my_ic"

    def test_to_frame_uses_labels(self):
        [er] = _eval({"my_ic": ic()})
        assert er.to_frame()["metric_name"].to_list() == ["my_ic"]


class TestPrimary:
    def test_default_primary_is_first_key(self):
        [er] = _eval({"a": ic(), "b": ic_ir()})
        assert er.metrics.primary == ["a"]
        assert er.metrics.diagnostic == ["b"]

    def test_explicit_primary(self):
        [er] = _eval({"a": ic(), "b": ic_ir()}, primary="b")
        assert er.metrics.primary == ["b"]
        assert er.metrics.diagnostic == ["a"]

    def test_unknown_primary_raises(self):
        with pytest.raises(UserInputError, match="one of the metrics labels"):
            _eval({"a": ic()}, primary="zzz")


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
        [er] = _eval({"a": ic_ir(), "b": ic_ir()})
        assert set(er.metrics.outputs) == {"a", "b"}

    def test_different_config_coexists(self):
        # #494: one class under two labels with different config now runs both.
        from factrix.metrics import quantile_spread

        [er] = _eval(
            {"a": quantile_spread(n_groups=5), "b": quantile_spread(n_groups=10)}
        )
        assert set(er.metrics.outputs) == {"a", "b"}

    def test_per_instance_forward_periods_override(self):
        # Distinct horizons coexist; each carries its own resolved fp.
        [er] = fx.evaluate(
            _panel(),
            metrics={"fp5": ic_newey_west(), "fp20": ic_newey_west(forward_periods=20)},
            factor_cols=["factor"],
        )
        assert er.metrics["fp5"].metadata["forward_periods"] == 5
        assert er.metrics["fp20"].metadata["forward_periods"] == 20

    def test_shared_producer_runs_once_across_configs(self):
        # Both ic horizons require compute_ic, which is fp-independent — so the
        # single batchable producer is computed once, not per config.
        [er] = fx.evaluate(
            _panel(),
            metrics={"fp5": ic_newey_west(), "fp20": ic_newey_west(forward_periods=20)},
            factor_cols=["factor"],
        )
        assert er.plan.count("compute_ic [batchable]") == 1

    def test_top_level_forward_periods_is_default_fallback(self):
        # An instance left at its signature default inherits the top-level fp;
        # an explicit per-instance value still wins.
        [er] = fx.evaluate(
            _panel(),
            metrics={
                "dflt": ic_newey_west(),
                "expl": ic_newey_west(forward_periods=10),
            },
            factor_cols=["factor"],
            forward_periods=20,
        )
        assert er.metrics["dflt"].metadata["forward_periods"] == 20
        assert er.metrics["expl"].metadata["forward_periods"] == 10
        assert er.forward_periods == 20  # primary = first key ("dflt")


class TestStrict:
    # A 12-date panel is below IC's period floor, so ``ic`` short-circuits to
    # NaN with reason ``insufficient_ic_periods`` — an apply-time failure.
    def test_strict_true_raises_on_inapplicable(self):
        thin = _panel(n_dates=12)
        with pytest.raises(UserInputError, match="inapplicable"):
            fx.evaluate(
                thin, metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
            )

    def test_strict_false_keeps_nan(self):
        thin = _panel(n_dates=12)
        [er] = fx.evaluate(
            thin,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )
        assert er.metrics["ic"].value != er.metrics["ic"].value  # NaN


class TestStrictStructureSoftening:
    # #494 / #497 carry-over: a PANEL primary on a single-asset TIMESERIES panel
    # is a structure mismatch — an apply-time failure under #476 §4.
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
        [er] = fx.evaluate(
            self._single_asset_panel(),
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )
        assert math.isnan(er.metrics["ic"].value)
