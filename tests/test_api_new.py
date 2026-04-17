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
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


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


class TestBackwardsCompat:
    def test_old_quick_check_still_works(self):
        df = _panel_with_price(60, 30, 0.3, 230)
        res = fl.quick_check(df, "mom", factor_type="cross_sectional")
        assert type(res).__name__ == "EvaluationResult"
