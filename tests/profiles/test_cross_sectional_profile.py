"""CrossSectionalProfile: from_artifacts, canonical_p, verdict, diagnose."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import get_type_hints

import pytest

from factorlib._types import FactorType, PValue
from factorlib.evaluation.profiles import CrossSectionalProfile
from factorlib.evaluation.profiles._base import _PROFILE_REGISTRY


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
        from factorlib.config import EventConfig
        from factorlib.evaluation._protocol import Artifacts
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
