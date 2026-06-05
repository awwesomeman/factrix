"""``fx.evaluate`` dict[str, Metric] API — labels, primary, strict (#497)."""

from __future__ import annotations

import factrix as fx
import pytest
from factrix._errors import UserInputError
from factrix.metrics import ic, ic_ir


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


class TestDuplicateClassGuard:
    def test_same_config_alias_is_allowed(self):
        [er] = _eval({"a": ic_ir(), "b": ic_ir()})
        assert set(er.metrics.outputs) == {"a", "b"}

    def test_different_config_via_params_raises(self):
        from factrix.metrics import quantile_spread

        with pytest.raises(UserInputError, match="different config"):
            _eval({"a": quantile_spread(n_groups=5), "b": quantile_spread(n_groups=10)})


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
