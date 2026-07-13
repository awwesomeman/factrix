"""The ``expected_warnings`` study-level declaration: mark, never drop.

A by-design few-asset study (single asset, pairs, a hand-picked handful
of names) declares its regime once on
``evaluate(..., expected_warnings=("few_assets",))``. Declared codes are
marked ``expected=True`` on their :class:`Warning` records — the audit
trail stays complete — while the human channels go quiet: the per-run
``UserWarning`` echoes stop and ``result.unexpected_warnings`` reads
empty. Inference is never touched, and undeclared behavior must stay
bit-for-bit unchanged.
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

    def test_thin_panel_still_emits_few_assets_unexpected(self):
        panel = _thin_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = fx.evaluate(
                panel,
                metrics={"ic": ic(), "spread": quantile_spread(n_groups=2)},
                factor_cols=["factor"],
            )
        res = results["factor"]
        few = _few_assets_warnings(res)
        assert few
        assert all(not w.expected for w in few)
        assert few[0] in res.unexpected_warnings
        assert "few_assets" in res.metrics["spread"].warning_codes

    def test_thin_panel_still_echoes_userwarning(self):
        panel = _thin_panel()
        with pytest.warns(UserWarning, match="min_assets_per_period"):
            fx.evaluate(panel, metrics={"ic": ic()}, factor_cols=["factor"])


class TestDeclaredStudy:
    """expected_warnings=("few_assets",): marked record, quiet channels."""

    def test_record_kept_and_marked_expected(self):
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"ic": ic(), "spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expected_warnings=("few_assets",),
        )
        res = results["factor"]
        few = _few_assets_warnings(res)
        assert few, "the record must never be dropped"
        assert all(w.expected for w in few)
        assert not [
            w for w in res.unexpected_warnings if w.code is WarningCode.FEW_ASSETS
        ]

    def test_metric_level_codes_stay_complete(self):
        """MetricResult.warning_codes is the record — the declaration never edits it."""
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expected_warnings=("few_assets",),
        )
        assert "few_assets" in results["factor"].metrics["spread"].warning_codes

    def test_no_userwarning_echo_on_declared_thin_panel(self):
        panel = _thin_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fx.evaluate(
                panel,
                metrics={"ic": ic()},
                factor_cols=["factor"],
                expected_warnings=("few_assets",),
            )

    def test_undeclared_codes_stay_unexpected(self):
        """Marking is per-code: only the declared code goes quiet."""
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expected_warnings=("few_assets",),
        )
        res = results["factor"]
        other = [w for w in res.warnings if w.code is not WarningCode.FEW_ASSETS]
        assert all(not w.expected for w in other)

    def test_inference_switch_stays_readable(self):
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expected_warnings=("few_assets",),
        )
        meta = results["factor"].metrics["spread"].metadata
        assert meta["method"] == "block-bootstrap CI"
        assert "p_value_t" in meta

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
            expected_warnings=("few_assets",),
        )
        p_undeclared = undeclared["factor"].metrics["spread"].p_value
        p_declared = declared["factor"].metrics["spread"].p_value
        assert np.isclose(p_undeclared, p_declared, rtol=0.0, atol=0.0)

    def test_wide_panel_declaration_marks_nothing(self):
        panel = _wide_panel()
        results = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=5)},
            factor_cols=["factor"],
            expected_warnings=("few_assets",),
        )
        res = results["factor"]
        assert not _few_assets_warnings(res)
        assert res.metrics["spread"].metadata["method"] == "non-overlapping t-test"

    def test_to_dict_serializes_expected_flag(self):
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expected_warnings=("few_assets",),
        )
        records = results["factor"].to_dict()["warnings"]
        few = [r for r in records if r["code"] == "few_assets"]
        assert few and all(r["expected"] for r in few)


class TestDeclarationSurface:
    """The declaration is study-level with a strict typo guard."""

    def test_metric_constructor_rejects_expected_warnings(self):
        with pytest.raises(UserInputError, match="study-level declaration"):
            quantile_spread(n_groups=2, expected_warnings=("few_assets",))

    def test_bare_string_rejected(self):
        panel = _thin_panel()
        with pytest.raises(UserInputError, match="tuple of WarningCode"):
            fx.evaluate(
                panel,
                metrics={"ic": ic()},
                factor_cols=["factor"],
                expected_warnings="few_assets",
            )

    def test_unknown_code_rejected(self):
        panel = _thin_panel()
        with pytest.raises(UserInputError, match="unknown codes are rejected"):
            fx.evaluate(
                panel,
                metrics={"ic": ic()},
                factor_cols=["factor"],
                expected_warnings=("few_asset",),
            )

    def test_enum_member_accepted(self):
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            expected_warnings=(WarningCode.FEW_ASSETS,),
        )
        few = _few_assets_warnings(results["factor"])
        assert few and all(w.expected for w in few)

    def test_evaluate_horizons_forwards_declaration(self):
        raw = fx.datasets.make_cs_panel(n_assets=6, n_dates=260, seed=11)
        results = fx.evaluate_horizons(
            raw,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=[5, 10],
            expected_warnings=("few_assets",),
        )
        assert results
        for res in results:
            few = _few_assets_warnings(res)
            assert few and all(w.expected for w in few)
