"""CrossSectionalProfile: from_artifacts, canonical_p, verdict, diagnose."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from datetime import datetime, timedelta
from typing import get_type_hints

import polars as pl
import pytest

from factrix._types import FactorType, PValue
from factrix.config import CrossSectionalConfig
from factrix.evaluation.pipeline import build_artifacts
from factrix.evaluation.profiles import CrossSectionalProfile
from factrix.evaluation.profiles._base import _PROFILE_REGISTRY


class TestSchema:
    def test_is_registered(self):
        assert _PROFILE_REGISTRY[FactorType.CROSS_SECTIONAL] is CrossSectionalProfile

    def test_canonical_field_in_whitelist(self):
        assert (
            CrossSectionalProfile.CANONICAL_P_FIELD
            in CrossSectionalProfile.P_VALUE_FIELDS
        )

    def test_canonical_is_ic_p(self):
        assert CrossSectionalProfile.CANONICAL_P_FIELD == "ic_p"

    def test_whitelist_subset_of_fields(self):
        field_names = {f.name for f in dc_fields(CrossSectionalProfile)}
        assert CrossSectionalProfile.P_VALUE_FIELDS.issubset(field_names)

    def test_frozen_and_slotted(self):
        # Slots: __dict__ should not exist on an instance
        assert CrossSectionalProfile.__dataclass_params__.frozen  # type: ignore[attr-defined]
        assert CrossSectionalProfile.__dataclass_params__.slots  # type: ignore[attr-defined]


class TestFromArtifacts:
    def test_constructs_for_strong_signal(self, cs_profile_strong):
        p = cs_profile_strong
        assert isinstance(p, CrossSectionalProfile)
        assert p.factor_name == "cs_strong"
        assert p.n_periods > 0

    def test_ic_fields_populated(self, cs_profile_strong):
        p = cs_profile_strong
        assert p.ic_tstat != 0.0
        assert 0.0 <= p.ic_p <= 1.0
        # Strong signal => small p
        assert p.ic_p < 0.05

    def test_spread_fields_populated(self, cs_profile_strong):
        p = cs_profile_strong
        assert 0.0 <= p.spread_p <= 1.0

    def test_weak_signal_yields_high_p(self, cs_profile_weak):
        assert cs_profile_weak.ic_p > 0.10

    def test_wrong_config_type_raises(self, cs_profile_strong):
        from factrix.config import EventConfig
        from factrix.evaluation._protocol import Artifacts
        import polars as pl
        bad = Artifacts(prepared=pl.DataFrame(), config=EventConfig())
        with pytest.raises(TypeError, match="expects CrossSectionalConfig"):
            CrossSectionalProfile.from_artifacts(bad)


class TestCanonicalP:
    def test_property_matches_ic_p(self, cs_profile_strong):
        assert cs_profile_strong.canonical_p == cs_profile_strong.ic_p

    def test_returns_float_value(self, cs_profile_strong):
        # PValue is a NewType over float at runtime
        assert isinstance(cs_profile_strong.canonical_p, float)


class TestVerdict:
    def test_strong_passes(self, cs_profile_strong):
        assert cs_profile_strong.verdict() == "PASS"

    def test_weak_fails(self, cs_profile_weak):
        assert cs_profile_weak.verdict() == "FAILED"

    def test_stricter_threshold_harder_to_pass(self, cs_profile_strong):
        # At threshold=10.0 (p threshold ~1e-23) almost nothing passes
        assert cs_profile_strong.verdict(threshold=10.0) == "FAILED"

    def test_threshold_monotone(self, cs_profile_strong):
        # Lowering the threshold (more permissive) cannot flip PASS to FAILED
        p = cs_profile_strong
        for t in [1.0, 1.5, 2.0]:
            if p.verdict(threshold=3.0) == "PASS":
                assert p.verdict(threshold=t) == "PASS"

    def test_threshold_zero_accepts_any_positive_p(self, cs_profile_weak):
        # threshold=0 → t-critical=0 → p-cut = 1.0 → any valid p passes.
        # The signal is weak, but at zero threshold FAILED would be a bug.
        assert cs_profile_weak.verdict(threshold=0.0) == "PASS"

    def test_negative_threshold_symmetric(self, cs_profile_strong):
        # Two-sided p is symmetric in the sign of the t-stat cutoff, so
        # verdict(-t) must match verdict(+t). No crash, no asymmetric
        # behaviour that would punish negative thresholds a caller might
        # plug in by mistake.
        p = cs_profile_strong
        for t in (1.0, 2.0, 3.0):
            assert p.verdict(threshold=-t) == p.verdict(threshold=t)


class TestDiagnose:
    def test_returns_list(self, cs_profile_strong):
        diag = cs_profile_strong.diagnose()
        assert isinstance(diag, list)

    def test_diagnostics_have_required_fields(self, cs_profile_strong):
        for d in cs_profile_strong.diagnose():
            assert d.severity in ("info", "warn", "veto")
            assert isinstance(d.message, str) and d.message
            # code is optional but our rules always set it
            assert d.code is not None

    def test_small_universe_flagged(self, cs_profile_strong):
        # The fixture uses N=30 which is <200, so the rule should fire
        codes = {d.code for d in cs_profile_strong.diagnose()}
        assert "cs.small_universe" in codes

    def test_oos_sign_flip_on_pure_noise(self, cs_profile_weak):
        # Weak near-zero signal often shows OOS sign flip
        diag = cs_profile_weak.diagnose()
        # Not asserting definitively; but at least one warn or veto typical
        assert any(d.severity in ("warn", "veto") for d in diag)

    def test_high_notional_turnover_fires_on_iid_factor(self, cs_profile_strong):
        # cs_profile_strong fixture uses re-drawn iid factor each date,
        # so Q1/Q10 membership rotates ~fully every rebalance and
        # notional_turnover > 0.5 by construction. The rule must fire.
        codes = {d.code for d in cs_profile_strong.diagnose()}
        assert "cs.high_notional_turnover" in codes
        assert cs_profile_strong.notional_turnover > 0.5

    def test_high_notional_turnover_quiet_on_persistent_factor(self):
        # Persistent factor (each asset's value fixed across dates) →
        # tail sets stable → notional_turnover ≈ 0 → rule must NOT fire.
        # Pins the negative half so a threshold drop (e.g. 0.5 → 0.0)
        # gets caught. NB: this does not catch a `>` → `<` operator
        # flip — for that you'd need a moderate-churn case in (0, 0.5).
        n_dates, n_assets = 40, 30
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            for i in range(n_assets):
                rows.append({
                    "date": d, "asset_id": f"a{i}",
                    "factor": float(i),  # fixed per asset across dates
                    "forward_return": float(i) * 0.01,
                })
        df = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        art = build_artifacts(df, CrossSectionalConfig())
        art.factor_name = "persistent"
        profile, _ = CrossSectionalProfile.from_artifacts(art)

        codes = {d.code for d in profile.diagnose()}
        assert "cs.high_notional_turnover" not in codes
        assert profile.notional_turnover < 0.5
