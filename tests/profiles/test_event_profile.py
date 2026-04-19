"""EventProfile: schema + from_artifacts smoke test."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib._types import FactorType
from factorlib.config import EventConfig
from factorlib.evaluation.pipeline import build_artifacts
from factorlib.evaluation.profiles import EventProfile
from factorlib.evaluation.profiles._base import _PROFILE_REGISTRY


def _event_panel(n_dates: int, n_assets: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for i in range(n_assets):
            # ~1/5 of rows are events with discrete {-1, +1}; the rest 0.
            f = int(rng.choice([-1, 0, 0, 0, 0, 1]))
            r = 0.5 * f + 0.5 * rng.standard_normal()
            rows.append({
                "date": d, "asset_id": f"ev{i}",
                "factor": float(f), "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.fixture
def event_profile_strong() -> EventProfile:
    df = _event_panel(n_dates=120, n_assets=15, seed=501)
    art = build_artifacts(df, EventConfig())
    art.factor_name = "ev_strong"
    profile, _ = EventProfile.from_artifacts(art)
    return profile


class TestSchema:
    def test_registered(self):
        assert _PROFILE_REGISTRY[FactorType.EVENT_SIGNAL] is EventProfile

    def test_canonical_is_caar_p(self):
        assert EventProfile.CANONICAL_P_FIELD == "caar_p"

    def test_canonical_in_whitelist(self):
        assert EventProfile.CANONICAL_P_FIELD in EventProfile.P_VALUE_FIELDS

    def test_event_ic_p_excluded_from_whitelist(self):
        # event_ic_p can be None when the signal has no magnitude variance,
        # so it cannot participate in a BHY batch that requires a valid p
        # for every factor. Verify the whitelist excludes it.
        assert "event_ic_p" not in EventProfile.P_VALUE_FIELDS

    def test_whitelist_fields_all_exist(self):
        names = {f.name for f in dc_fields(EventProfile)}
        assert EventProfile.P_VALUE_FIELDS.issubset(names)


class TestFromArtifacts:
    def test_constructs(self, event_profile_strong):
        p = event_profile_strong
        assert isinstance(p, EventProfile)
        assert p.factor_name == "ev_strong"
        assert p.n_periods > 0
        assert p.n_events > 0

    def test_canonical_p_is_caar_p(self, event_profile_strong):
        assert event_profile_strong.canonical_p == event_profile_strong.caar_p

    def test_verdict_callable(self, event_profile_strong):
        assert event_profile_strong.verdict() in ("PASS", "FAILED")

    def test_diagnose_is_list(self, event_profile_strong):
        assert isinstance(event_profile_strong.diagnose(), list)

    def test_wrong_config_raises(self):
        from factorlib.config import CrossSectionalConfig
        from factorlib.evaluation._protocol import Artifacts
        bad = Artifacts(prepared=pl.DataFrame(), config=CrossSectionalConfig())
        with pytest.raises(TypeError, match="expects EventConfig"):
            EventProfile.from_artifacts(bad)
