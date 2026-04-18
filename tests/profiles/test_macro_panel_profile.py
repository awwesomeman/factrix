"""MacroPanelProfile: schema + from_artifacts smoke test."""

from __future__ import annotations

from dataclasses import fields as dc_fields

import polars as pl
import pytest

from factorlib._types import FactorType
from factorlib.config import MacroPanelConfig
from factorlib.evaluation.pipeline import build_artifacts
from factorlib.evaluation.profiles import MacroPanelProfile
from factorlib.evaluation.profiles._base import _PROFILE_REGISTRY

from tests.conftest import make_macro_panel


@pytest.fixture
def macro_panel_strong() -> MacroPanelProfile:
    df = make_macro_panel(n_dates=60, n_countries=12, signal=0.5, seed=601)
    art = build_artifacts(df, MacroPanelConfig())
    art.factor_name = "mp_strong"
    return MacroPanelProfile.from_artifacts(art)


class TestSchema:
    def test_registered(self):
        assert _PROFILE_REGISTRY[FactorType.MACRO_PANEL] is MacroPanelProfile

    def test_canonical_is_fm_beta_p(self):
        assert MacroPanelProfile.CANONICAL_P_FIELD == "fm_beta_p"

    def test_canonical_in_whitelist(self):
        assert MacroPanelProfile.CANONICAL_P_FIELD in MacroPanelProfile.P_VALUE_FIELDS

    def test_whitelist_fields_all_exist(self):
        names = {f.name for f in dc_fields(MacroPanelProfile)}
        assert MacroPanelProfile.P_VALUE_FIELDS.issubset(names)


class TestFromArtifacts:
    def test_constructs(self, macro_panel_strong):
        p = macro_panel_strong
        assert isinstance(p, MacroPanelProfile)
        assert p.factor_name == "mp_strong"
        assert p.n_periods > 0
        assert p.median_cross_section_n > 0

    def test_canonical_p_is_fm_beta_p(self, macro_panel_strong):
        assert macro_panel_strong.canonical_p == macro_panel_strong.fm_beta_p

    def test_pooled_beta_populated(self, macro_panel_strong):
        # pooled_ols may have n_obs below the MIN threshold; field should
        # still be a float even if it's 0.0
        assert isinstance(macro_panel_strong.pooled_beta, float)

    def test_wrong_config_raises(self):
        from factorlib.config import CrossSectionalConfig
        from factorlib.evaluation._protocol import Artifacts
        bad = Artifacts(prepared=pl.DataFrame(), config=CrossSectionalConfig())
        with pytest.raises(TypeError, match="expects MacroPanelConfig"):
            MacroPanelProfile.from_artifacts(bad)
