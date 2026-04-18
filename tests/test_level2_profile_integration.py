"""T3.S2 integration tests: Level 2 metrics surfaced on CS Profile.

Covers:
  - config.regime_labels → regime_ic_min_tstat / _consistent populated
  - price panel → multi_horizon retention / monotonic populated
  - config.spanning_base_spreads → spanning_alpha_t / _p populated
  - all three None-by-default when config inputs missing
  - diagnose rules fire on pathological cases
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import factorlib as fl
from factorlib.evaluation.diagnostics import clear_custom_rules


@pytest.fixture(autouse=True)
def _isolate_custom_rules():
    clear_custom_rules()
    yield
    clear_custom_rules()


def _cs_panel_with_price(
    n_dates: int = 80,
    n_assets: int = 40,
    signal: float = 0.3,
    seed: int = 101,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    prices = {f"a{i}": 100.0 for i in range(n_assets)}
    rows: list[dict] = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        ret = signal * f * 0.01 + (1 - abs(signal)) * 0.01 * rng.standard_normal(n_assets)
        for i in range(n_assets):
            prices[f"a{i}"] *= (1 + ret[i])
            rows.append({
                "date": d, "asset_id": f"a{i}",
                "factor": float(f[i]), "price": float(prices[f"a{i}"]),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestDefaultsOff:
    def test_no_config_inputs_leaves_fields_none(self):
        df = _cs_panel_with_price(seed=1)
        p = fl.evaluate(df, "x", factor_type="cross_sectional")
        # All three Level 2 metrics are opt-in; default config leaves
        # the Profile fields as None.
        assert p.regime_ic_min_tstat is None
        assert p.regime_ic_consistent is None
        assert p.multi_horizon_ic_retention is None
        assert p.multi_horizon_ic_monotonic is None
        assert p.spanning_alpha_t is None
        assert p.spanning_alpha_p is None


class TestRegimeIc:
    def test_fields_populated_when_labels_supplied(self):
        df = _cs_panel_with_price(seed=2)
        n = df.select(pl.col("date").n_unique()).item()
        dates = df.select(pl.col("date").unique().sort())["date"].to_list()
        regimes = pl.DataFrame({
            "date": dates,
            "regime": ["bull" if i < n // 2 else "bear" for i in range(n)],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        cfg = fl.CrossSectionalConfig(regime_labels=regimes)
        p = fl.evaluate(df, "x", config=cfg)
        assert p.regime_ic_min_tstat is not None
        assert isinstance(p.regime_ic_min_tstat, float)
        assert isinstance(p.regime_ic_consistent, bool)


class TestMultiHorizonIc:
    def test_retention_is_finite_number(self):
        df = _cs_panel_with_price(seed=3, signal=0.4)
        cfg = fl.CrossSectionalConfig(multi_horizon_periods=[1, 5, 10, 20])
        p = fl.evaluate(df, "x", config=cfg)
        assert p.multi_horizon_ic_retention is not None
        assert np.isfinite(p.multi_horizon_ic_retention)

    def test_custom_periods_config(self):
        df = _cs_panel_with_price(seed=4)
        cfg = fl.CrossSectionalConfig(multi_horizon_periods=[1, 3, 5])
        p = fl.evaluate(df, "x", config=cfg)
        assert p.multi_horizon_ic_retention is not None

    def test_skipped_when_no_price(self):
        rng = np.random.default_rng(5)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(80)]
        rows = [
            {"date": d, "asset_id": f"a{i}",
             "factor": float(rng.standard_normal()),
             "forward_return": float(rng.standard_normal() * 0.01)}
            for d in dates for i in range(30)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        cfg = fl.CrossSectionalConfig(multi_horizon_periods=[1, 5, 10, 20])
        p = fl.evaluate(df, "x", config=cfg, preprocess=False)
        assert p.multi_horizon_ic_retention is None
        assert p.multi_horizon_ic_monotonic is None


class TestSpanningAlpha:
    def test_fields_populated_when_base_spreads_supplied(self):
        df = _cs_panel_with_price(seed=6, signal=0.3)
        base_df = _cs_panel_with_price(seed=6, signal=0.3)

        base_profile, base_arts = fl.evaluate(
            base_df, "base", factor_type="cross_sectional",
            return_artifacts=True,
        )
        base_spread = base_arts.intermediates["spread_series"].select(
            "date", "spread",
        )

        cfg = fl.CrossSectionalConfig(
            spanning_base_spreads={"base": base_spread},
        )
        p = fl.evaluate(df, "candidate", config=cfg)
        assert p.spanning_alpha_t is not None
        assert p.spanning_alpha_p is not None
        assert 0.0 <= p.spanning_alpha_p <= 1.0


class TestDiagnoseRules:
    def test_multi_horizon_decay_fires_on_very_low_retention(self):
        df = _cs_panel_with_price(seed=7)
        p = fl.evaluate(df, "x", factor_type="cross_sectional")
        p2 = dataclasses.replace(p, multi_horizon_ic_retention=0.1)
        codes = [d.code for d in p2.diagnose()]
        assert "cs.multi_horizon_decay_fast" in codes

    def test_multi_horizon_decay_uses_abs_not_signed(self):
        # Sign-flip (retention=-0.6) must NOT fire the decay rule —
        # sign-flip is reported via monotonic=False, decay is about
        # magnitude loss only.
        df = _cs_panel_with_price(seed=7)
        p = fl.evaluate(df, "x", factor_type="cross_sectional")
        p2 = dataclasses.replace(p, multi_horizon_ic_retention=-0.6)
        codes = [d.code for d in p2.diagnose()]
        assert "cs.multi_horizon_decay_fast" not in codes

    def test_multi_horizon_decay_quiet_when_retention_none(self):
        df = _cs_panel_with_price(seed=8)
        p = fl.evaluate(df, "x", factor_type="cross_sectional")
        p2 = dataclasses.replace(p, multi_horizon_ic_retention=None)
        codes = [d.code for d in p2.diagnose()]
        assert "cs.multi_horizon_decay_fast" not in codes

    def test_regime_inconsistent_fires_on_false(self):
        df = _cs_panel_with_price(seed=9)
        p = fl.evaluate(df, "x", factor_type="cross_sectional")
        p2 = dataclasses.replace(p, regime_ic_consistent=False)
        codes = [d.code for d in p2.diagnose()]
        assert "cs.regime_ic_inconsistent" in codes

    def test_regime_inconsistent_quiet_on_none(self):
        df = _cs_panel_with_price(seed=10)
        p = fl.evaluate(df, "x", factor_type="cross_sectional")
        codes = [d.code for d in p.diagnose()]
        assert "cs.regime_ic_inconsistent" not in codes

    def test_spanning_absorbed_fires_on_high_p(self):
        df = _cs_panel_with_price(seed=11)
        p = fl.evaluate(df, "x", factor_type="cross_sectional")
        p2 = dataclasses.replace(p, spanning_alpha_p=0.8)
        codes = [d.code for d in p2.diagnose()]
        assert "cs.spanning_alpha_absorbed" in codes
