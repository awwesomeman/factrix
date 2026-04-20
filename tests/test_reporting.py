"""Smoke + contract tests for ``factorlib.reporting.describe_profile_values``.

The renderer is Profile-driven: no ``Artifacts`` handle required, and
every non-None dataclass field appears as a row. L2 opt-in summary
fields appear only when the corresponding config was enabled.
"""

from __future__ import annotations

import polars as pl

import factorlib as fl
from factorlib.reporting import describe_profile_values

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
