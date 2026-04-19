"""Top-level profile-era API: evaluate, evaluate_batch, list_factor_types."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import factorlib as fl
from factorlib.evaluation.profiles import (
    CrossSectionalProfile,
    EventProfile,
)


def _panel_with_price(n_dates: int, n_assets: int, signal: float, seed: int):
    """Build a raw panel and preprocess it with the default CS config.

    All callers in this module use default CrossSectionalConfig (no
    forward_periods override), so preprocessing in the fixture matches
    what they'd otherwise do with fl.preprocess at the call site.
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    prices = {f"a{i}": 100.0 for i in range(n_assets)}
    rows = []
    for d in dates:
        f_vec = rng.standard_normal(n_assets)
        ret_vec = signal * f_vec * 0.01 + (1 - abs(signal)) * 0.01 * rng.standard_normal(n_assets)
        for i in range(n_assets):
            prices[f"a{i}"] *= (1 + ret_vec[i])
            rows.append({
                "date": d, "asset_id": f"a{i}",
                "factor": float(f_vec[i]), "price": float(prices[f"a{i}"]),
            })
    raw = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    return fl.preprocess(raw, config=fl.CrossSectionalConfig())


class TestListFactorTypes:
    def test_returns_list(self):
        types = fl.list_factor_types()
        assert isinstance(types, list)
        assert "cross_sectional" in types
        assert "event_signal" in types

    def test_all_registered_appear(self):
        from factorlib.evaluation.profiles import _PROFILE_REGISTRY
        types = set(fl.list_factor_types())
        assert types == {ft.value for ft in _PROFILE_REGISTRY}


class TestEvaluate:
    def test_returns_profile(self):
        df = _panel_with_price(80, 35, signal=0.4, seed=100)
        profile = fl.evaluate(df, "mom", factor_type="cross_sectional")
        assert isinstance(profile, CrossSectionalProfile)
        assert profile.factor_name == "mom"

    def test_respects_config_override(self):
        df = _panel_with_price(80, 35, signal=0.4, seed=101)
        profile = fl.evaluate(
            df, "mom", factor_type="cross_sectional", n_groups=5,
        )
        assert isinstance(profile, CrossSectionalProfile)

    def test_invalid_factor_type_lists_valid(self):
        df = _panel_with_price(40, 20, 0.3, 102)
        with pytest.raises(ValueError, match="Unknown factor_type"):
            fl.evaluate(df, "x", factor_type="not_real_type")

    def test_cannot_pass_both_config_and_overrides(self):
        df = _panel_with_price(40, 20, 0.3, 103)
        cfg = fl.CrossSectionalConfig()
        with pytest.raises(TypeError, match="Pick one"):
            fl.evaluate(df, "x", config=cfg, n_groups=5)

    def test_raw_panel_raises_strict_gate(self):
        # Build a raw panel (no forward_return) — strict gate must refuse
        # rather than silently auto-preprocess.
        rng = np.random.default_rng(777)
        raw = pl.DataFrame({
            "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
            "asset_id": ["a0"] * 10,
            "factor": rng.standard_normal(10),
            "price": 100.0 + rng.standard_normal(10),
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        with pytest.raises(ValueError, match="preprocessed panel"):
            fl.evaluate(raw, "x", factor_type="cross_sectional")

    def test_return_artifacts_returns_tuple(self):
        from factorlib.evaluation._protocol import Artifacts

        df = _panel_with_price(60, 30, 0.3, 104)
        result = fl.evaluate(
            df, "x", factor_type="cross_sectional", return_artifacts=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        profile, artifacts = result
        assert isinstance(profile, CrossSectionalProfile)
        assert isinstance(artifacts, Artifacts)
        assert "factor" in artifacts.prepared.columns
        assert artifacts.factor_name == "x"


class TestEvaluateBatch:
    def test_returns_profile_set(self):
        factors = {
            f"f_{k}": _panel_with_price(60, 30, signal=s, seed=200 + k)
            for k, s in enumerate([0.4, 0.3, 0.2])
        }
        ps = fl.evaluate_batch(factors, factor_type="cross_sectional")
        assert isinstance(ps, fl.ProfileSet)
        assert len(ps) == 3
        assert ps.profile_cls is CrossSectionalProfile

    def test_stop_on_error_false_skips_bad_factors(self):
        # Pass a DataFrame missing required columns to force an error.
        good = _panel_with_price(60, 30, signal=0.3, seed=210)
        bad = pl.DataFrame({"date": [], "asset_id": [], "factor": []})
        captured: list[tuple[str, BaseException]] = []
        ps = fl.evaluate_batch(
            {"good": good, "bad": bad},
            factor_type="cross_sectional",
            stop_on_error=False,
            on_error=lambda name, exc: captured.append((name, exc)),
        )
        assert len(ps) == 1  # bad was skipped
        assert captured and captured[0][0] == "bad"

    def test_stop_on_error_true_raises(self):
        bad = pl.DataFrame({"date": [], "asset_id": [], "factor": []})
        with pytest.raises(Exception):  # narrow type varies by failure mode
            fl.evaluate_batch(
                {"bad": bad},
                factor_type="cross_sectional",
                stop_on_error=True,
            )

    def test_keep_artifacts_returns_tuple_with_dict(self):
        from factorlib.evaluation._protocol import Artifacts

        factors = {
            "a": _panel_with_price(60, 30, 0.3, 230),
            "b": _panel_with_price(60, 30, 0.3, 231),
        }
        result = fl.evaluate_batch(
            factors, factor_type="cross_sectional", keep_artifacts=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        ps, arts = result
        assert isinstance(ps, fl.ProfileSet)
        assert set(arts) == {"a", "b"}
        assert all(isinstance(a, Artifacts) for a in arts.values())
        # prepared survives when compact=False
        assert "factor" in arts["a"].prepared.columns

    def test_compact_drops_prepared_panel(self):
        factors = {"a": _panel_with_price(60, 30, 0.3, 232)}
        ps, arts = fl.evaluate_batch(
            factors,
            factor_type="cross_sectional",
            keep_artifacts=True,
            compact=True,
        )
        assert arts["a"].compact is True
        with pytest.raises(RuntimeError, match="compact mode"):
            _ = arts["a"].prepared.columns

    def test_compact_without_keep_artifacts_raises(self):
        factors = {"a": _panel_with_price(40, 20, 0.3, 233)}
        with pytest.raises(ValueError, match="requires keep_artifacts=True"):
            fl.evaluate_batch(
                factors, factor_type="cross_sectional", compact=True,
            )

    def test_on_result_called_per_success(self):
        factors = {
            "a": _panel_with_price(60, 30, 0.3, 220),
            "b": _panel_with_price(60, 30, 0.3, 221),
        }
        seen: list[str] = []
        fl.evaluate_batch(
            factors,
            factor_type="cross_sectional",
            on_result=lambda name, _: seen.append(name),
        )
        assert set(seen) == {"a", "b"}

    def test_on_result_false_stops_early(self):
        # 3 factors, stop after 2 — 3rd must not appear.
        factors = {
            name: _panel_with_price(60, 30, 0.3, 240 + k)
            for k, name in enumerate(["a", "b", "c"])
        }
        seen: list[str] = []

        def cb(name: str, _p: object) -> bool:
            seen.append(name)
            return len(seen) < 2

        ps = fl.evaluate_batch(
            factors, factor_type="cross_sectional", on_result=cb,
        )
        assert seen == ["a", "b"]
        assert len(ps) == 2  # third was never evaluated

    def test_on_result_true_continues(self):
        factors = {
            "a": _panel_with_price(60, 30, 0.3, 243),
            "b": _panel_with_price(60, 30, 0.3, 244),
        }
        seen: list[str] = []
        fl.evaluate_batch(
            factors,
            factor_type="cross_sectional",
            on_result=lambda name, _: seen.append(name) or True,
        )
        assert set(seen) == {"a", "b"}


class TestDescribeProfile:
    def test_runs_without_raising(self, capsys):
        fl.describe_profile("cross_sectional")
        captured = capsys.readouterr()
        assert "CrossSectionalProfile" in captured.out
        assert "CANONICAL_P_FIELD" in captured.out

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown factor_type"):
            fl.describe_profile("not_a_type")  # type: ignore[arg-type]

    def test_accepts_enum(self):
        fl.describe_profile(fl.FactorType.EVENT_SIGNAL)


