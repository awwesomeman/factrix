"""The ``expect_few_assets`` study-level declaration.

A by-design few-asset study (single asset, pairs, a hand-picked handful
of names) declares its thin cross-section once on
``evaluate(..., expect_few_assets=True)``: the ``FEW_ASSETS`` warning and
its ``UserWarning`` echo stop firing, while every inference consequence
(the block-bootstrap switch, the per-date asset floors) stays readable
from ``MetricResult.metadata``, now joined by a ``few_assets_expected``
acknowledgment marker. Undeclared behavior must stay bit-for-bit
unchanged.
"""

from __future__ import annotations

import warnings

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._errors import UserInputError
from factrix.metrics import ic, quantile_spread


def _thin_panel(n_assets: int = 6, n_dates: int = 220) -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, seed=11)
    return fx.preprocess.compute_forward_return(raw, forward_periods=5)


def _wide_panel(n_assets: int = 60, n_dates: int = 220) -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, seed=11)
    return fx.preprocess.compute_forward_return(raw, forward_periods=5)


def _few_assets_warnings(result) -> list:
    return [w for w in result.warnings if w.code is WarningCode.FEW_ASSETS]


class TestUndeclaredBehaviorUnchanged:
    """Without the declaration, FEW_ASSETS keeps firing exactly as before."""

    def test_thin_panel_still_emits_few_assets(self):
        panel = _thin_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = fx.evaluate(
                panel,
                metrics={"ic": ic(), "spread": quantile_spread(n_groups=2)},
                factor_cols=["factor"],
            )
        res = results["factor"]
        assert _few_assets_warnings(res)
        assert "few_assets" in res.metrics["spread"].warning_codes
        assert "few_assets_expected" not in res.metrics["spread"].metadata

    def test_thin_panel_still_echoes_userwarning(self):
        panel = _thin_panel()
        with pytest.warns(UserWarning, match="min_assets_per_period"):
            fx.evaluate(panel, metrics={"ic": ic()}, factor_cols=["factor"])


class TestDeclaredStudy:
    """expect_few_assets=True: no warning, switch stays readable."""

    def test_no_few_assets_warning_on_declared_thin_panel(self):
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"ic": ic(), "spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expect_few_assets=True,
        )
        res = results["factor"]
        assert not _few_assets_warnings(res)
        for out in res.metrics.values():
            assert "few_assets" not in out.warning_codes

    def test_no_userwarning_echo_on_declared_thin_panel(self):
        panel = _thin_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fx.evaluate(
                panel,
                metrics={"ic": ic()},
                factor_cols=["factor"],
                expect_few_assets=True,
            )

    def test_inference_switch_stays_readable(self):
        """The declaration acknowledges the switch; it must not hide it."""
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expect_few_assets=True,
        )
        meta = results["factor"].metrics["spread"].metadata
        assert meta["method"] == "block-bootstrap CI"
        assert "p_value_t" in meta
        assert meta["few_assets_expected"] is True

    def test_ic_floors_stay_stamped_with_marker(self):
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            expect_few_assets=True,
        )
        meta = results["factor"].metrics["ic"].metadata
        assert meta["min_assets_per_period"] < meta["warn_assets_per_period"]
        assert meta["few_assets_expected"] is True

    def test_p_value_identical_to_undeclared_run(self):
        """The declaration changes reporting only — never the inference."""
        panel = _thin_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            undeclared = fx.evaluate(
                panel,
                metrics={"spread": quantile_spread(n_groups=2)},
                factor_cols=["factor"],
            )
        declared = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expect_few_assets=True,
        )
        p_undeclared = undeclared["factor"].metrics["spread"].p_value
        p_declared = declared["factor"].metrics["spread"].p_value
        assert np.isclose(p_undeclared, p_declared, rtol=0.0, atol=0.0)

    def test_wide_panel_declaration_is_a_noop(self):
        panel = _wide_panel()
        results = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=5)},
            factor_cols=["factor"],
            expect_few_assets=True,
        )
        meta = results["factor"].metrics["spread"].metadata
        assert meta["method"] == "non-overlapping t-test"
        assert "few_assets_expected" not in meta


class TestDeclarationSurface:
    """The declaration is study-level, not a per-metric knob."""

    def test_metric_constructor_rejects_expect_few_assets(self):
        with pytest.raises(UserInputError, match="study-level declaration"):
            quantile_spread(n_groups=2, expect_few_assets=True)

    def test_non_bool_declaration_rejected(self):
        panel = _thin_panel()
        with pytest.raises(UserInputError, match="expect_few_assets"):
            fx.evaluate(
                panel,
                metrics={"ic": ic()},
                factor_cols=["factor"],
                expect_few_assets="yes",
            )

    def test_evaluate_horizons_forwards_declaration(self):
        raw = fx.datasets.make_cs_panel(n_assets=6, n_dates=260, seed=11)
        results = fx.evaluate_horizons(
            raw,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=[5, 10],
            expect_few_assets=True,
        )
        assert results
        for res in results:
            assert not _few_assets_warnings(res)
