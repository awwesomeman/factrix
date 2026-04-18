"""MacroCommonProfile: schema + from_artifacts smoke + single-asset degeneracy."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib._types import FactorType
from factorlib.config import MacroCommonConfig
from factorlib.evaluation.pipeline import build_artifacts
from factorlib.evaluation.profiles import MacroCommonProfile
from factorlib.evaluation.profiles._base import _PROFILE_REGISTRY


def _macro_common(n_dates: int, n_assets: int, signal: float, seed: int) -> pl.DataFrame:
    """A single common factor series replicated across assets."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    common_factor = rng.standard_normal(n_dates)
    rows = []
    for d_i, d in enumerate(dates):
        for a_i in range(n_assets):
            r = signal * common_factor[d_i] + (1 - abs(signal)) * rng.standard_normal()
            rows.append({
                "date": d, "asset_id": f"a{a_i}",
                "factor": float(common_factor[d_i]),
                "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.fixture
def macro_common_profile() -> MacroCommonProfile:
    df = _macro_common(n_dates=120, n_assets=10, signal=0.4, seed=701)
    art = build_artifacts(df, MacroCommonConfig(ts_window=40))
    art.factor_name = "mc_strong"
    return MacroCommonProfile.from_artifacts(art)


class TestSchema:
    def test_registered(self):
        assert _PROFILE_REGISTRY[FactorType.MACRO_COMMON] is MacroCommonProfile

    def test_canonical_is_ts_beta_p(self):
        assert MacroCommonProfile.CANONICAL_P_FIELD == "ts_beta_p"

    def test_canonical_in_whitelist(self):
        assert MacroCommonProfile.CANONICAL_P_FIELD in MacroCommonProfile.P_VALUE_FIELDS

    def test_whitelist_subset_of_fields(self):
        names = {f.name for f in dc_fields(MacroCommonProfile)}
        assert MacroCommonProfile.P_VALUE_FIELDS.issubset(names)


class TestFromArtifacts:
    def test_constructs(self, macro_common_profile):
        p = macro_common_profile
        assert isinstance(p, MacroCommonProfile)
        assert p.factor_name == "mc_strong"
        assert p.n_assets > 1
        assert p.n_periods > 0

    def test_canonical_p_is_ts_beta_p(self, macro_common_profile):
        assert macro_common_profile.canonical_p == macro_common_profile.ts_beta_p

    def test_single_asset_is_degenerate(self):
        # N=1 case: the cross-sectional t-test on per-asset β cannot be
        # computed. Profile should fall back to single-asset t-stat with
        # p_value=1.0 (suppressed from BHY).
        df = _macro_common(n_dates=120, n_assets=1, signal=0.5, seed=999)
        art = build_artifacts(df, MacroCommonConfig(ts_window=40))
        art.factor_name = "single"
        p = MacroCommonProfile.from_artifacts(art)
        assert p.n_assets == 1
        assert p.ts_beta_p == 1.0
        assert p.verdict() == "FAILED"
        # single_asset rule should fire
        codes = {d.code for d in p.diagnose()}
        assert "macro_common.single_asset" in codes

    def test_wrong_config_raises(self):
        from factorlib.config import CrossSectionalConfig
        from factorlib.evaluation._protocol import Artifacts
        bad = Artifacts(prepared=pl.DataFrame(), config=CrossSectionalConfig())
        with pytest.raises(TypeError, match="expects MacroCommonConfig"):
            MacroCommonProfile.from_artifacts(bad)
