"""EventProfile: schema + from_artifacts smoke test."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factrix._types import FactorType
from factrix.config import EventConfig
from factrix.evaluation.pipeline import build_artifacts
from factrix.evaluation.profiles import EventProfile
from factrix.evaluation.profiles._base import _PROFILE_REGISTRY


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
        from factrix.config import CrossSectionalConfig
        from factrix.evaluation._protocol import Artifacts
        bad = Artifacts(prepared=pl.DataFrame(), config=CrossSectionalConfig())
        with pytest.raises(TypeError, match="expects EventConfig"):
            EventProfile.from_artifacts(bad)


class TestSingleAssetFallback:
    """N=1 event panels are a legitimate workflow (firm-specific event
    studies). CAAR degrades from cross-sectional to time-average; clustering
    diagnostic is disabled. The ``event.single_asset`` rule must fire so
    users see the semantic change in diagnose output."""

    def test_single_asset_fires_rule(self):
        df = _event_panel(n_dates=120, n_assets=1, seed=901)
        art = build_artifacts(df, EventConfig())
        art.factor_name = "ev_n1"
        profile, _ = EventProfile.from_artifacts(art)
        # N=1 → clustering disabled, Profile field is None
        assert profile.clustering_hhi is None
        assert profile.clustering_hhi_normalized is None
        # Rule fires with info severity
        codes = {d.code: d.severity for d in profile.diagnose()}
        assert "event.single_asset" in codes
        assert codes["event.single_asset"] == "info"

    def test_multi_asset_does_not_fire_rule(self, event_profile_strong):
        codes = {d.code for d in event_profile_strong.diagnose()}
        assert "event.single_asset" not in codes

    def test_single_asset_caar_reflects_injected_signal(self):
        """CAAR must not only be non-NaN at N=1 — it must point in the
        direction of the injected signal. ``_event_panel`` uses
        ``r = 0.5 * f + 0.5 * noise``, so mean(signed_return | factor≠0)
        should be clearly positive. A silent-degeneracy failure mode would
        pass a plain ``not isnan`` check but give caar_mean ≈ 0.
        """
        df = _event_panel(n_dates=200, n_assets=1, seed=902)
        art = build_artifacts(df, EventConfig())
        art.factor_name = "ev_n1"
        profile, _ = EventProfile.from_artifacts(art)
        assert profile.n_events >= 10
        assert not np.isnan(profile.caar_mean)
        # With a 0.5 signal coefficient, CAAR should be comfortably
        # positive — not just non-zero by chance.
        assert profile.caar_mean > 0.1, (
            f"caar_mean={profile.caar_mean:.4f} is suspicious for N=1 panel "
            f"with injected signal_coef=0.5; expected > 0.1"
        )


class TestCoverageSummary:
    """Silent per-date / per-asset drops are recorded in
    ``artifacts.intermediates['coverage']`` so users can inspect how much
    of their panel actually reached the canonical test. Not pushed as a
    warning; pulled via the Artifacts object.
    """

    def test_cs_coverage_records_drops(self):
        """CS: compute_ic drops dates with per-date N < MIN_IC_PERIODS."""
        from factrix.config import CrossSectionalConfig

        # 20 dates where every date has only 5 assets (below MIN_IC_PERIODS=10
        # for IC) but 6 assets global; max_per_date=5 >= 2 passes N guard
        # but compute_ic would drop every date. Use ≥2 per date to clear
        # guard, then rely on compute_ic's internal N>=10 filter to skip.
        rng = np.random.default_rng(701)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        rows = []
        for d_i, d in enumerate(dates):
            # Rotate assets so per-date N=5 (below MIN_IC_PERIODS=10).
            asset_start = d_i % 6
            for k in range(5):
                asset = f"a{(asset_start + k) % 6}"
                rows.append({
                    "date": d, "asset_id": asset,
                    "price": 100.0 + float(rng.standard_normal()),
                    "factor": float(rng.standard_normal()),
                })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        from factrix.preprocess.pipeline import preprocess_cs_factor
        prepared = preprocess_cs_factor(df, config=CrossSectionalConfig())
        art = build_artifacts(prepared, CrossSectionalConfig())
        cov = art.get("coverage")
        assert cov.height == 1
        row = cov.row(0, named=True)
        # All dates with per-date N=5 < MIN_IC_PERIODS=10 → compute_ic
        # filters every date out.
        assert row["axis"] == "dates"
        assert row["n_total"] > 0
        assert row["n_kept"] == 0
        assert row["n_dropped"] == row["n_total"]
        assert "min_ic_periods" in row["drop_reason"]

    def test_macro_common_coverage_records_skipped_assets(self):
        """MC: compute_ts_betas skips assets with T < MIN_TS_OBS=20."""
        from factrix.config import MacroCommonConfig

        rng = np.random.default_rng(702)
        # 5 assets with full T=40, 3 assets with T=10 (below MIN_TS_OBS)
        rows = []
        # Long-history assets (40 dates)
        for a in range(5):
            common = rng.standard_normal(40)
            for d_i in range(40):
                d = datetime(2024, 1, 1) + timedelta(days=d_i)
                rows.append({
                    "date": d, "asset_id": f"long{a}",
                    "factor": float(common[d_i]),
                    "forward_return": float(rng.standard_normal()),
                })
        # Short-history assets (10 dates)
        for a in range(3):
            for d_i in range(10):
                d = datetime(2024, 1, 1) + timedelta(days=d_i)
                rows.append({
                    "date": d, "asset_id": f"short{a}",
                    "factor": float(rng.standard_normal()),
                    "forward_return": float(rng.standard_normal()),
                })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        art = build_artifacts(df, MacroCommonConfig())
        cov = art.get("coverage")
        assert cov.height == 1
        row = cov.row(0, named=True)
        assert row["axis"] == "assets"
        assert row["n_total"] == 8
        assert row["n_kept"] == 5
        assert row["n_dropped"] == 3
        assert "min_ts_obs" in row["drop_reason"]
