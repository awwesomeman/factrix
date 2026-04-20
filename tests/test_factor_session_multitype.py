"""fl.factor() cross-type coverage for EventFactor / MP / MC subclasses.

P0 of ``spike_factor_session.md`` only shipped ``CrossSectionalFactor``.
This file exercises the PR 2/3 follow-up (Event / MacroPanel /
MacroCommon) to enforce the cross-type usage consistency (U1) promised
by the demo's cheat-sheet:

- Factory dispatch picks the right subclass via ``_FACTOR_REGISTRY``
- Every method returns ``MetricOutput`` (uniform return contract, D1/D3)
- ``f.evaluate()`` is value-equal to ``fl.evaluate(df, name, config=cfg)``
- Repeat metric calls hit the cache
- Cross-type methods raise ``AttributeError`` (Python idiomatic, §3.10)
"""

from __future__ import annotations

import dataclasses as _dc
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import factorlib as fl
from factorlib._types import MetricOutput
from factorlib.config import (
    CrossSectionalConfig,
    EventConfig,
    MacroCommonConfig,
    MacroPanelConfig,
)
from factorlib.factor import (
    CrossSectionalFactor,
    EventFactor,
    MacroCommonFactor,
    MacroPanelFactor,
)


# ---------------------------------------------------------------------------
# Raw-panel fixtures (no reliance on conftest to keep this file self-contained)
# ---------------------------------------------------------------------------

@pytest.fixture
def event_panel() -> pl.DataFrame:
    """Pre-preprocessed event panel: discrete {-1, 0, +1} signal with
    ``forward_return`` + ``abnormal_return`` + ``price`` columns already
    materialized (matches what ``fl.preprocess`` produces, so ``fl.factor``
    passes the strict gate without re-running preprocess)."""
    rng = np.random.default_rng(42)
    n_assets, n_dates = 30, 200
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for a in range(n_assets):
        price = 100.0
        for d in dates:
            is_event = rng.random() < 0.04
            direction = float(rng.choice([-1.0, 1.0])) if is_event else 0.0
            ret = rng.normal(0.0, 0.015) + 0.02 * direction
            price *= (1 + ret)
            rows.append({
                "date": d, "asset_id": f"a{a}",
                "factor": direction,
                "forward_return": float(ret),
                "abnormal_return": float(ret),
                "price": float(price),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.fixture
def event_cfg() -> EventConfig:
    return EventConfig(forward_periods=5)


@pytest.fixture
def macro_panel() -> pl.DataFrame:
    """Pre-preprocessed macro-panel (small-N) with ``forward_return``."""
    rng = np.random.default_rng(7)
    n_countries, n_dates = 15, 240
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        fvals = rng.standard_normal(n_countries)
        for i in range(n_countries):
            r = 0.4 * fvals[i] + 0.6 * rng.standard_normal()
            rows.append({
                "date": d, "asset_id": f"c{i}",
                "factor": float(fvals[i]),
                "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.fixture
def macro_panel_cfg() -> MacroPanelConfig:
    return MacroPanelConfig()


@pytest.fixture
def macro_common_panel() -> pl.DataFrame:
    """Pre-preprocessed macro-common panel: shared factor broadcast across
    assets, each asset has its own noisy forward return."""
    rng = np.random.default_rng(13)
    n_assets, n_dates = 20, 300
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    shared_factor = rng.standard_normal(n_dates)
    rows = []
    for ai in range(n_assets):
        beta = rng.normal(0.5, 0.2)
        for di, d in enumerate(dates):
            r = beta * shared_factor[di] + rng.normal(0.0, 0.5)
            rows.append({
                "date": d, "asset_id": f"a{ai}",
                "factor": float(shared_factor[di]),
                "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.fixture
def macro_common_cfg() -> MacroCommonConfig:
    return MacroCommonConfig()


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

class TestFactoryDispatch:
    def test_event_signal_returns_event_factor(self, event_panel, event_cfg):
        f = fl.factor(event_panel, "GoldenCross", config=event_cfg)
        assert isinstance(f, EventFactor)

    def test_macro_panel_returns_macro_panel_factor(self, macro_panel, macro_panel_cfg):
        f = fl.factor(macro_panel, "MP", config=macro_panel_cfg)
        assert isinstance(f, MacroPanelFactor)

    def test_macro_common_returns_macro_common_factor(self, macro_common_panel, macro_common_cfg):
        f = fl.factor(macro_common_panel, "MC", config=macro_common_cfg)
        assert isinstance(f, MacroCommonFactor)


# ---------------------------------------------------------------------------
# Uniform MetricOutput return contract
# ---------------------------------------------------------------------------

class TestEventFactorMethods:
    @pytest.mark.parametrize("method_name", [
        "caar", "bmp_test", "event_hit_rate", "profit_factor",
        "event_skewness", "event_ic", "signal_density", "clustering_hhi",
        "corrado_rank_test", "caar_trend", "oos_decay",
        "mfe_mae_summary", "event_around_return", "multi_horizon_hit_rate",
    ])
    def test_method_returns_metric_output(self, event_panel, event_cfg, method_name):
        f = fl.factor(event_panel, "GoldenCross", config=event_cfg)
        result = getattr(f, method_name)()
        assert isinstance(result, MetricOutput), (
            f"{method_name} returned {type(result).__name__}"
        )


class TestMacroPanelFactorMethods:
    @pytest.mark.parametrize("method_name", [
        "fm_beta", "pooled_beta", "beta_sign_consistency",
        "quantile_spread", "turnover", "breakeven_cost", "net_spread",
        "beta_trend", "oos_decay",
    ])
    def test_method_returns_metric_output(self, macro_panel, macro_panel_cfg, method_name):
        f = fl.factor(macro_panel, "MP", config=macro_panel_cfg)
        result = getattr(f, method_name)()
        assert isinstance(result, MetricOutput)


class TestMacroCommonFactorMethods:
    @pytest.mark.parametrize("method_name", [
        "ts_beta", "mean_r_squared", "ts_beta_sign_consistency",
        "beta_trend", "oos_decay",
    ])
    def test_method_returns_metric_output(self, macro_common_panel, macro_common_cfg, method_name):
        f = fl.factor(macro_common_panel, "MC", config=macro_common_cfg)
        result = getattr(f, method_name)()
        assert isinstance(result, MetricOutput)


# ---------------------------------------------------------------------------
# fl.evaluate(df, name, cfg)  ≡  fl.factor(df, name, cfg).evaluate()
# ---------------------------------------------------------------------------

def _profile_dicts_equal(a: dict, b: dict) -> bool:
    """Compare two Profile dataclass dicts with tolerance on floats.

    Direct vs session paths can differ by last-bit floating-point noise
    depending on the order operations are dispatched (e.g. Polars reorders
    group-by chunks between runs). Keys / non-float values must match
    exactly; floats are compared with ``math.isclose`` at the default
    relative tolerance.
    """
    import math
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float):
            if not math.isclose(va, vb, rel_tol=1e-9, abs_tol=0.0):
                return False
        elif va != vb:
            return False
    return True


class TestEvaluateEquivalence:
    def test_event(self, event_panel, event_cfg):
        p_direct = fl.evaluate(event_panel, "E", config=event_cfg)
        p_session = fl.factor(event_panel, "E", config=event_cfg).evaluate()
        assert type(p_direct) is type(p_session)
        assert _profile_dicts_equal(_dc.asdict(p_direct), _dc.asdict(p_session))

    def test_macro_panel(self, macro_panel, macro_panel_cfg):
        p_direct = fl.evaluate(macro_panel, "MP", config=macro_panel_cfg)
        p_session = fl.factor(macro_panel, "MP", config=macro_panel_cfg).evaluate()
        assert _profile_dicts_equal(_dc.asdict(p_direct), _dc.asdict(p_session))

    def test_macro_common(self, macro_common_panel, macro_common_cfg):
        p_direct = fl.evaluate(macro_common_panel, "MC", config=macro_common_cfg)
        p_session = fl.factor(macro_common_panel, "MC", config=macro_common_cfg).evaluate()
        assert _profile_dicts_equal(_dc.asdict(p_direct), _dc.asdict(p_session))


# ---------------------------------------------------------------------------
# Cache behaviour: repeat standalone call returns cached MetricOutput
# ---------------------------------------------------------------------------

class TestCacheReuse:
    def test_event_repeat_call_returns_cached(self, event_panel, event_cfg):
        f = fl.factor(event_panel, "E", config=event_cfg)
        first = f.caar()
        second = f.caar()
        assert second is f.artifacts.metric_outputs["caar"]
        assert first.value == second.value

    def test_mp_repeat_call_returns_cached(self, macro_panel, macro_panel_cfg):
        f = fl.factor(macro_panel, "MP", config=macro_panel_cfg)
        first = f.fm_beta()
        second = f.fm_beta()
        assert second is f.artifacts.metric_outputs["fm_beta"]
        assert first.value == second.value

    def test_mc_repeat_call_returns_cached(self, macro_common_panel, macro_common_cfg):
        f = fl.factor(macro_common_panel, "MC", config=macro_common_cfg)
        first = f.ts_beta()
        second = f.ts_beta()
        assert second is f.artifacts.metric_outputs["ts_beta"]


# ---------------------------------------------------------------------------
# Cross-type AttributeError (Python-idiomatic — spike §3.10)
# ---------------------------------------------------------------------------

class TestCrossTypeAttributeError:
    def test_event_factor_has_no_ic(self, event_panel, event_cfg):
        f = fl.factor(event_panel, "E", config=event_cfg)
        with pytest.raises(AttributeError):
            f.ic()

    def test_mp_factor_has_no_caar(self, macro_panel, macro_panel_cfg):
        f = fl.factor(macro_panel, "MP", config=macro_panel_cfg)
        with pytest.raises(AttributeError):
            f.caar()

    def test_mc_factor_has_no_quantile_spread(self, macro_common_panel, macro_common_cfg):
        f = fl.factor(macro_common_panel, "MC", config=macro_common_cfg)
        with pytest.raises(AttributeError):
            f.quantile_spread()

    def test_cs_factor_has_no_fm_beta(self, noisy_panel):
        f = fl.factor(noisy_panel, "CS")
        with pytest.raises(AttributeError):
            f.fm_beta()


# ---------------------------------------------------------------------------
# Short-circuit paths (no price / single asset / no magnitude variance)
# ---------------------------------------------------------------------------

class TestMacroCommonN1Fallback:
    """Locks in the review-follow-up fix: MacroCommonFactor.ts_beta with
    N=1 must call the shared ``ts_beta_single_asset_fallback`` helper
    directly, *not* trigger a hidden ``self.evaluate()`` side-effect.
    The earlier implementation built the full Profile behind the user's
    back which (a) violated the "method does one thing" contract and (b)
    leaked unrelated cache entries on what should be a targeted call.
    """

    @pytest.fixture
    def single_asset_mc_panel(self) -> pl.DataFrame:
        rng = np.random.default_rng(999)
        n_dates = 120
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for i, d in enumerate(dates):
            fv = float(rng.standard_normal())
            rows.append({
                "date": d, "asset_id": "sole",
                "factor": fv,
                "forward_return": 0.3 * fv + 0.7 * float(rng.standard_normal()),
            })
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_ts_beta_returns_fallback_without_side_effects(
        self, single_asset_mc_panel, macro_common_cfg,
    ):
        f = fl.factor(single_asset_mc_panel, "SoleAsset", config=macro_common_cfg)
        result = f.ts_beta()
        # Shared fallback: p_value=1.0, named method string present.
        assert result.metadata["n_assets"] == 1
        assert result.metadata["p_value"] == 1.0
        assert "single-asset" in result.metadata["method"]
        # Unrelated metrics must NOT be in cache just because ts_beta was called.
        cached_keys = set(f.artifacts.metric_outputs.keys())
        assert cached_keys == {"ts_beta"}, (
            f"ts_beta() leaked cache entries: {cached_keys - {'ts_beta'}}"
        )


class TestEventShortCircuits:
    @pytest.fixture
    def event_panel_no_price(self) -> pl.DataFrame:
        rng = np.random.default_rng(101)
        n_assets, n_dates = 20, 150
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for a in range(n_assets):
            for d in dates:
                is_event = rng.random() < 0.04
                direction = float(rng.choice([-1.0, 1.0])) if is_event else 0.0
                ret = rng.normal(0.0, 0.015) + 0.02 * direction
                rows.append({
                    "date": d, "asset_id": f"a{a}",
                    "factor": direction,
                    "forward_return": float(ret),
                    "abnormal_return": float(ret),
                })
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_mfe_mae_short_circuits_without_price(self, event_panel_no_price):
        f = fl.factor(event_panel_no_price, "E", config=EventConfig(forward_periods=5))
        r = f.mfe_mae_summary()
        assert r.metadata["reason"] == "no_price_column"

    def test_event_around_short_circuits_without_price(self, event_panel_no_price):
        f = fl.factor(event_panel_no_price, "E", config=EventConfig(forward_periods=5))
        r = f.event_around_return()
        assert r.metadata["reason"] == "no_price_column"

    def test_multi_horizon_hit_rate_short_circuits_without_price(self, event_panel_no_price):
        f = fl.factor(event_panel_no_price, "E", config=EventConfig(forward_periods=5))
        r = f.multi_horizon_hit_rate()
        assert r.metadata["reason"] == "no_price_column"

    def test_event_ic_short_circuits_for_discrete_signal(self, event_panel, event_cfg):
        # Discrete {-1, 0, +1} signal has no magnitude variance →
        # event_ic is undefined and short-circuits.
        f = fl.factor(event_panel, "E", config=event_cfg)
        r = f.event_ic()
        assert r.metadata["reason"] == "no_magnitude_variance"
