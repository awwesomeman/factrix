"""Smoke + contract tests for ``factorlib.reporting.describe_profile_values``.

The detail view auto-discovers opt-in metrics (regime / multi-horizon /
spanning) from ``artifacts.metric_outputs``. These tests pin:

- Scalar table always prints (smoke)
- ``include_detail=False`` suppresses the detail sections
- When an opt-in metric is present, its section appears in output
- When absent, the section is silently omitted (no raise)
- Artifacts without ``metric_outputs`` populated raises ``ValueError``
  pointing at ``return_artifacts=True`` as the fix
"""

from __future__ import annotations

import polars as pl
import pytest

import factorlib as fl
from factorlib.evaluation._protocol import Artifacts
from factorlib.reporting import describe_profile_values

from tests.conftest import _cs_panel


def _panel(n_dates: int = 100, n_assets: int = 30, seed: int = 11) -> pl.DataFrame:
    return _cs_panel(
        n_dates=n_dates, n_assets=n_assets,
        signal_coef=0.25, seed=seed, include_price=True,
    )


# ---------------------------------------------------------------------------
# Scalar smoke
# ---------------------------------------------------------------------------

class TestScalarTable:
    def test_prints_header_and_scalar_rows(self, capsys):
        df = _panel()
        profile, arts = fl.evaluate(
            df, "cs_smoke", return_artifacts=True,
        )
        describe_profile_values(profile, arts)
        out = capsys.readouterr().out
        # Header
        assert "cs_smoke" in out
        assert "CrossSectionalProfile" in out
        # At least one canonical metric row
        assert "ic" in out
        assert "q1_q5_spread" in out

    def test_include_detail_false_omits_detail_section(self, capsys):
        df = _panel()
        dates = df["date"].unique().sort()
        regime_df = pl.DataFrame({
            "date": dates,
            "regime": ["bull" if i % 2 == 0 else "bear" for i in range(len(dates))],
        })
        cfg = fl.CrossSectionalConfig(regime_labels=regime_df)
        profile, arts = fl.evaluate(
            df, "cs_no_detail",
            config=cfg, return_artifacts=True,
        )
        describe_profile_values(profile, arts, include_detail=False)
        out = capsys.readouterr().out
        # Scalar table still there
        assert "Metrics:" in out
        # But no detail header
        assert "Detail:" not in out


# ---------------------------------------------------------------------------
# Detail auto-discovery
# ---------------------------------------------------------------------------

class TestDetailAutoDiscover:
    def test_regime_detail_renders_when_present(self, capsys):
        df = _panel()
        dates = df["date"].unique().sort()
        regime_df = pl.DataFrame({
            "date": dates,
            "regime": ["bull" if i % 2 == 0 else "bear" for i in range(len(dates))],
        })
        cfg = fl.CrossSectionalConfig(regime_labels=regime_df)
        profile, arts = fl.evaluate(
            df, "cs_regime_detail",
            config=cfg, return_artifacts=True,
        )
        describe_profile_values(profile, arts)
        out = capsys.readouterr().out
        assert "Detail:" in out
        assert "regime_ic" in out
        # Per-regime labels should appear
        assert "bull" in out
        assert "bear" in out

    def test_multi_horizon_detail_renders_when_present(self, capsys):
        df = _panel(n_dates=120)
        cfg = fl.CrossSectionalConfig(multi_horizon_periods=[1, 5, 10])
        profile, arts = fl.evaluate(
            df, "cs_mh_detail",
            config=cfg, return_artifacts=True,
        )
        describe_profile_values(profile, arts)
        out = capsys.readouterr().out
        assert "multi_horizon_ic" in out
        assert "horizon" in out

    def test_absent_metric_section_silently_omitted(self, capsys):
        # No opt-in config → none of the detail sections should appear,
        # and no exception should be raised.
        df = _panel()
        profile, arts = fl.evaluate(
            df, "cs_no_optin", return_artifacts=True,
        )
        describe_profile_values(profile, arts)  # must not raise
        out = capsys.readouterr().out
        assert "regime_ic" not in out
        assert "multi_horizon_ic" not in out
        assert "spanning_alpha" not in out
        # The scalar table alone: no "Detail:" header either
        assert "Detail:" not in out


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------

class TestArtifactsRequired:
    def test_empty_metric_outputs_raises_value_error(self):
        df = _panel()
        profile = fl.evaluate(df, "cs_missing_outputs")
        # Hand-built Artifacts with no metric_outputs populated — mimics a
        # user who constructed Artifacts manually rather than going through
        # fl.evaluate(..., return_artifacts=True).
        empty = Artifacts(prepared=pl.DataFrame(), config=fl.CrossSectionalConfig())
        with pytest.raises(ValueError, match="return_artifacts"):
            describe_profile_values(profile, empty)
