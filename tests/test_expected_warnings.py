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
from datetime import datetime, timedelta

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._errors import UserInputError
from factrix.metrics import ic, ic_ir, quantile_spread


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

    def test_thin_cross_section_keeps_the_t_test(self):
        """Declaring the regime expected does not change the test that ran."""
        panel = _thin_panel()
        results = fx.evaluate(
            panel,
            metrics={"spread": quantile_spread(n_groups=2)},
            factor_cols=["factor"],
            expected_warnings=("few_assets",),
        )
        meta = results["factor"].metrics["spread"].metadata
        assert meta["method"] == "non-overlapping t-test"
        assert "p_value_t" not in meta

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


class TestEchoStopsForEveryMetric:
    """The declaration is study-level, so it has to reach every warn path, not
    only the two that gated on it by hand (``ic`` / ``fm_beta``). The matrix
    below drives one metric per shared helper: ``_warn_below_floor``,
    ``_warn_below_scaled_floor`` and the drop-rate helper behind
    ``_surface_null_drop`` / ``_surface_drop_stats``.
    """

    @staticmethod
    def _single_asset_panel(n_dates: int, rate: float, seed: int) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        flag = (rng.random(n_dates) < rate).astype(float)
        rets = rng.normal(0.0, 0.01, n_dates)
        raw = pl.DataFrame(
            {
                "date": [
                    datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)
                ],
                "asset_id": ["A"] * n_dates,
                "factor": flag,
                "price": 100.0 * np.cumprod(1.0 + rets),
            }
        )
        return fx.preprocess.compute_forward_return(raw, forward_periods=5)

    @staticmethod
    def _short_dense_panel(n_dates: int = 40) -> pl.DataFrame:
        rng = np.random.default_rng(5)
        raw = pl.DataFrame(
            {
                "date": [
                    datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)
                ],
                "asset_id": ["A"] * n_dates,
                "factor": rng.normal(size=n_dates),
                "price": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n_dates)),
            }
        )
        return fx.preprocess.compute_forward_return(raw, forward_periods=5)

    def _cases(self):
        thin_events = self._single_asset_panel(600, 0.05, seed=71)
        return [
            (
                "predictive_beta",
                self._short_dense_panel(),
                {"pb": fx.metrics.predictive_beta()},
                "unreliable_se_short_periods",
            ),
            (
                "caar",
                thin_events,
                {"caar": fx.metrics.caar()},
                "few_events",
            ),
            (
                "corrado_rank",
                thin_events,
                {"c": fx.metrics.corrado_rank()},
                "few_events",
            ),
            (
                "event_hit_rate",
                thin_events,
                {"h": fx.metrics.event_hit_rate()},
                "few_events",
            ),
            (
                "bmp_z",
                thin_events,
                {"b": fx.metrics.bmp_z()},
                "few_events",
            ),
        ]

    def test_declaration_removes_the_echo_and_keeps_the_record(self):
        # Run each case twice — undeclared, then declared — and compare. The
        # undeclared run must echo the code (otherwise the case is vacuous);
        # the declared run must echo strictly less while keeping an
        # expected=True record.
        for label, panel, metrics, code in self._cases():
            with warnings.catch_warnings(record=True) as undeclared:
                warnings.simplefilter("always")
                plain = fx.evaluate(
                    panel, metrics=metrics, factor_cols=["factor"], strict=False
                )["factor"]
            with warnings.catch_warnings(record=True) as declared:
                warnings.simplefilter("always")
                quiet = fx.evaluate(
                    panel,
                    metrics=metrics,
                    factor_cols=["factor"],
                    strict=False,
                    expected_warnings=(code,),
                )["factor"]

            n_before = len([w for w in undeclared if f"{label}:" in str(w.message)])
            n_after = len([w for w in declared if f"{label}:" in str(w.message)])
            assert n_before > 0, f"{label} never echoed {code} — case is vacuous"
            assert n_after < n_before, f"{label} still echoed {code}"

            assert any(w.code.value == code for w in plain.warnings), label
            records = [w for w in quiet.warnings if w.code.value == code]
            assert records, f"{label} lost the {code} record"
            assert all(w.expected for w in records), label
            assert all(w.code.value != code for w in quiet.unexpected_warnings), label
            # Inference is untouched by the declaration.
            for key in metrics:
                assert quiet.metrics[key].p_value == plain.metrics[key].p_value, label


def _drop_panel(
    n_periods: int = 80, n_assets: int = 6, dropped: int = 16
) -> pl.DataFrame:
    """Thin cross-section whose first ``dropped`` periods carry no factor.

    ``compute_ic`` drops those periods, so every consumer of the shared
    primitive surfaces ``excessive_period_drops`` alongside ``few_assets``.
    """
    rng = np.random.default_rng(3)
    factor = rng.normal(size=(n_periods, n_assets))
    ret = 0.3 * factor + rng.normal(size=(n_periods, n_assets))
    rows = [
        {
            "date": datetime(2020, 1, 1) + timedelta(days=period),
            "asset_id": str(asset),
            "factor": None if period < dropped else float(factor[period, asset]),
            "forward_return": float(ret[period, asset]),
        }
        for period in range(n_periods)
        for asset in range(n_assets)
    ]
    return pl.DataFrame(rows)


_DROP_METRICS = {
    "ic": ic(inference=fx.inference.NEWEY_WEST),
    "ic_ir": ic_ir(),
}


class TestDropEchoFollowsDeclaration:
    """Drop-rate echoes from a shared primitive obey the same contract."""

    def test_undeclared_drop_rate_still_echoes(self):
        with pytest.warns(UserWarning, match="of periods dropped"):
            results = fx.evaluate(
                _drop_panel(),
                metrics=_DROP_METRICS,
                factor_cols=["factor"],
                forward_periods=1,
                strict=False,
            )
        res = results["factor"]
        codes = {w.code for w in res.warnings}
        assert WarningCode.EXCESSIVE_PERIOD_DROPS in codes
        assert all(not w.expected for w in res.warnings)

    def test_mixed_declaration_silences_every_declared_echo(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = fx.evaluate(
                _drop_panel(),
                metrics=_DROP_METRICS,
                factor_cols=["factor"],
                forward_periods=1,
                strict=False,
                expected_warnings=("few_assets", "excessive_period_drops"),
            )
        assert not [w for w in caught if issubclass(w.category, UserWarning)], [
            str(w.message) for w in caught
        ]
        res = results["factor"]
        for name in _DROP_METRICS:
            assert "excessive_period_drops" in res.metrics[name].warning_codes
            assert "few_assets" in res.metrics[name].warning_codes
        declared = [
            w
            for w in res.warnings
            if w.code in (WarningCode.EXCESSIVE_PERIOD_DROPS, WarningCode.FEW_ASSETS)
        ]
        assert {w.source for w in declared} >= set(_DROP_METRICS)
        assert all(w.expected for w in declared)
        assert not [
            w
            for w in res.unexpected_warnings
            if w.code in (WarningCode.EXCESSIVE_PERIOD_DROPS, WarningCode.FEW_ASSETS)
        ]
