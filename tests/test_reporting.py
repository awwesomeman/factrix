"""Smoke + contract tests for ``factorlib.reporting.describe_profile_values``.

The renderer is Profile-driven: no ``Artifacts`` handle required, and
every non-None dataclass field appears as a row. L2 opt-in summary
fields appear only when the corresponding config was enabled.
"""

from __future__ import annotations

from dataclasses import fields

import polars as pl
import pytest

import factorlib as fl
from factorlib.evaluation.profiles import (
    CrossSectionalProfile,
    EventProfile,
    MacroCommonProfile,
    MacroPanelProfile,
)
from factorlib.reporting import describe_profile_values
from factorlib.reporting import _SKIP_FIELDS  # internal, used to assert contract

from tests.conftest import _cs_panel


def _panel(n_dates: int = 100, n_assets: int = 30, seed: int = 11) -> pl.DataFrame:
    return _cs_panel(
        n_dates=n_dates, n_assets=n_assets,
        signal_coef=0.25, seed=seed, include_price=True,
    )


class TestProfileRendering:
    def test_header_and_scalar_fields_present(self, capsys):
        df = _panel()
        profile = fl.evaluate(df, "cs_smoke")
        describe_profile_values(profile)
        out = capsys.readouterr().out
        assert "cs_smoke" in out
        assert "CrossSectionalProfile" in out
        # Canonical scalar fields
        assert "ic_mean" in out
        assert "ic_p" in out
        assert "quantile_spread" in out
        assert "top_concentration" in out

    def test_accepts_only_profile_no_artifacts(self, capsys):
        # describe_profile_values must work on the vanilla fl.evaluate
        # return value — no return_artifacts=True needed.
        df = _panel()
        profile = fl.evaluate(df, "cs_bare")
        describe_profile_values(profile)  # must not raise
        out = capsys.readouterr().out
        assert "Values:" in out

    def test_l2_summary_rows_only_when_configured(self, capsys):
        df = _panel()
        profile_no_l2 = fl.evaluate(df, "cs_no_l2")
        describe_profile_values(profile_no_l2)
        out_no = capsys.readouterr().out
        assert "regime_ic_min_tstat" not in out_no  # None → skipped

        dates = df["date"].unique().sort()
        regime_df = pl.DataFrame({
            "date": dates,
            "regime": ["bull" if i % 2 == 0 else "bear" for i in range(len(dates))],
        })
        cfg = fl.CrossSectionalConfig(regime_labels=regime_df)
        profile_l2 = fl.evaluate(df, "cs_with_l2", config=cfg)
        describe_profile_values(profile_l2)
        out_yes = capsys.readouterr().out
        assert "regime_ic_min_tstat" in out_yes
        assert "regime_ic_consistent" in out_yes

    def test_every_non_none_field_renders(self, capsys):
        # Regression guard: if someone accidentally adds a scalar field
        # name to _SKIP_FIELDS, that field silently disappears from output.
        # Every non-None, non-skipped field MUST appear as a row.
        df = _panel()
        profile = fl.evaluate(df, "cs_completeness")
        describe_profile_values(profile)
        out = capsys.readouterr().out
        missing = []
        for f in fields(profile):
            if f.name in _SKIP_FIELDS:
                continue
            value = getattr(profile, f.name)
            if value is None:
                continue
            if f.name not in out:
                missing.append(f.name)
        assert not missing, (
            f"Fields with non-None values missing from rendered output: "
            f"{missing}. Contract violation — either the renderer skipped "
            f"them or _SKIP_FIELDS has drifted."
        )


# ---------------------------------------------------------------------------
# Cross-factor-type coverage (guard against profile-subclass drift)
# ---------------------------------------------------------------------------

class TestAllProfileTypes:
    def test_cross_sectional_renders(self, cs_profile_strong, capsys):
        describe_profile_values(cs_profile_strong)
        out = capsys.readouterr().out
        assert "CrossSectionalProfile" in out and "Values:" in out

    def test_event_renders(self, capsys):
        from tests.profiles.test_event_profile import _event_panel
        from factorlib.config import EventConfig
        from factorlib.evaluation.pipeline import build_artifacts
        df = _event_panel(n_dates=100, n_assets=25, seed=13)
        art = build_artifacts(df, EventConfig())
        art.factor_name = "ev_render"
        profile, _ = EventProfile.from_artifacts(art)
        describe_profile_values(profile)
        out = capsys.readouterr().out
        assert "EventProfile" in out and "Values:" in out

    def test_macro_panel_renders(self, capsys):
        from tests.conftest import make_macro_panel
        from factorlib.config import MacroPanelConfig
        from factorlib.evaluation.pipeline import build_artifacts
        df = make_macro_panel(n_dates=100, n_countries=20, signal=0.3, seed=17)
        art = build_artifacts(df, MacroPanelConfig())
        art.factor_name = "mp_render"
        profile, _ = MacroPanelProfile.from_artifacts(art)
        describe_profile_values(profile)
        out = capsys.readouterr().out
        assert "MacroPanelProfile" in out and "Values:" in out

    def test_macro_common_renders(self, capsys):
        from tests.profiles.test_macro_common_profile import _macro_common
        from factorlib.config import MacroCommonConfig
        from factorlib.evaluation.pipeline import build_artifacts
        df = _macro_common(n_dates=100, n_assets=5, signal=0.3, seed=19)
        art = build_artifacts(df, MacroCommonConfig())
        art.factor_name = "mc_render"
        profile, _ = MacroCommonProfile.from_artifacts(art)
        describe_profile_values(profile)
        out = capsys.readouterr().out
        assert "MacroCommonProfile" in out and "Values:" in out
