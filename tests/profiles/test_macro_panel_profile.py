"""MacroPanelProfile: schema + from_artifacts smoke test."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from datetime import datetime, timedelta

import polars as pl
import pytest

from factrix._types import FactorType
from factrix.config import MacroPanelConfig
from factrix.evaluation.pipeline import build_artifacts
from factrix.evaluation.profiles import MacroPanelProfile
from factrix.evaluation.profiles._base import _PROFILE_REGISTRY

from tests.conftest import make_macro_panel


@pytest.fixture
def macro_panel_strong() -> MacroPanelProfile:
    df = make_macro_panel(n_dates=60, n_countries=12, signal=0.5, seed=601)
    art = build_artifacts(df, MacroPanelConfig())
    art.factor_name = "mp_strong"
    profile, _ = MacroPanelProfile.from_artifacts(art)
    return profile


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
        from factrix.config import CrossSectionalConfig
        from factrix.evaluation._protocol import Artifacts
        bad = Artifacts(prepared=pl.DataFrame(), config=CrossSectionalConfig())
        with pytest.raises(TypeError, match="expects MacroPanelConfig"):
            MacroPanelProfile.from_artifacts(bad)


class TestDiagnose:
    def test_high_notional_turnover_fires_on_iid_factor(self, macro_panel_strong):
        # The macro_panel_strong fixture re-draws factor iid each date,
        # so top/bot tercile membership rotates fully every rebalance and
        # notional_turnover > 0.5 by construction. The rule must fire.
        codes = {d.code for d in macro_panel_strong.diagnose()}
        assert "macro_panel.high_notional_turnover" in codes
        assert macro_panel_strong.notional_turnover > 0.5

    def test_high_notional_turnover_quiet_on_persistent_factor(self):
        # Persistent factor (each country fixed across dates) → tail sets
        # stable → notional_turnover ≈ 0 → rule must NOT fire. Pins the
        # negative half so a threshold drop (e.g. 0.5 → 0.0) gets caught.
        # NB: this does not catch a `>` → `<` operator flip — for that
        # you'd need a moderate-churn case in (0, 0.5).
        n_dates, n_countries = 40, 12
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            for i in range(n_countries):
                rows.append({
                    "date": d, "asset_id": f"c{i}",
                    "factor": float(i),
                    "forward_return": float(i) * 0.01,
                })
        df = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        art = build_artifacts(df, MacroPanelConfig())
        art.factor_name = "mp_persistent"
        profile, _ = MacroPanelProfile.from_artifacts(art)

        codes = {d.code for d in profile.diagnose()}
        assert "macro_panel.high_notional_turnover" not in codes
        assert profile.notional_turnover < 0.5
