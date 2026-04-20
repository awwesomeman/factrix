"""Tests for ``fl.factor()`` / ``Factor`` / ``CrossSectionalFactor``.

Covers the design contract from ``docs/spike_factor_session.md``:
  - Eager ``build_artifacts`` at factory time (§3.2)
  - Per-factor-type subclass with static methods (§3.3)
  - Per-call override bypasses AND doesn't pollute cache (§3.4)
  - ``fl.evaluate`` ≡ ``fl.factor().evaluate()`` value-equality (§3.5)
  - Factor method = thin adapter on primitive via cache (§3.6)
  - ``__post_init__`` validates factor_type + factor_name (§3.7)
  - Cross-type metric call → AttributeError (§3.10)
  - Strict preprocess gate mirrors ``fl.evaluate`` (§3.8)
"""

from __future__ import annotations

import dataclasses as _dc

import polars as pl
import pytest

import factorlib as fl
from factorlib._types import FactorType, MetricOutput
from factorlib.config import CrossSectionalConfig, EventConfig
from factorlib.evaluation._protocol import Artifacts
from factorlib.factor import CrossSectionalFactor, Factor, _FACTOR_REGISTRY
from factorlib.evaluation.pipeline import build_artifacts


# ---------------------------------------------------------------------------
# Factory + basic shape
# ---------------------------------------------------------------------------

class TestFactoryBasics:
    def test_returns_cs_factor_by_default(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        assert isinstance(f, CrossSectionalFactor)
        assert f.factor_name == "Mom_20D"
        assert f.config.factor_type == FactorType.CROSS_SECTIONAL

    def test_registry_dispatch(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D", factor_type="cross_sectional")
        assert type(f) is _FACTOR_REGISTRY[FactorType.CROSS_SECTIONAL]

    def test_explicit_config_overrides_factor_type(self, noisy_panel):
        cfg = CrossSectionalConfig(forward_periods=3, n_groups=3)
        f = fl.factor(noisy_panel, "Mom_20D", config=cfg)
        assert f.config.forward_periods == 3
        assert f.config.n_groups == 3

    def test_config_and_overrides_conflict_raises(self, noisy_panel):
        cfg = CrossSectionalConfig()
        with pytest.raises(TypeError, match="cannot pass both config="):
            fl.factor(noisy_panel, "Mom_20D", config=cfg, forward_periods=3)

    def test_artifacts_escape_hatch(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        assert isinstance(f.artifacts, Artifacts)
        assert f.artifacts.factor_name == "Mom_20D"
        assert "ic_series" in f.artifacts.intermediates
        assert "spread_series" in f.artifacts.intermediates


# ---------------------------------------------------------------------------
# Strict preprocess gate (§3.8)
# ---------------------------------------------------------------------------

class TestStrictGate:
    def test_missing_forward_return_raises(self, noisy_panel):
        no_fr = noisy_panel.drop("forward_return")
        with pytest.raises(ValueError, match="preprocessed panel"):
            fl.factor(no_fr, "Mom_20D")


# ---------------------------------------------------------------------------
# __post_init__ validation (§3.7)
# ---------------------------------------------------------------------------

class TestPostInitValidation:
    def test_wrong_factor_type_raises(self, noisy_panel):
        # Build CS artifacts, try to wrap in CrossSectionalFactor with a
        # config of the wrong type — requires manually mutating to cover
        # the case of direct instantiation with mismatched artifacts.
        cs_cfg = CrossSectionalConfig()
        arts = build_artifacts(noisy_panel, cs_cfg)
        arts.factor_name = "Mom_20D"
        # Swap the config to a different factor_type
        arts.config = EventConfig()
        with pytest.raises(TypeError, match="expects factor_type="):
            CrossSectionalFactor(artifacts=arts)

    def test_empty_factor_name_raises(self, noisy_panel):
        cfg = CrossSectionalConfig()
        arts = build_artifacts(noisy_panel, cfg)
        # factor_name left as default "" — direct instantiation should reject
        with pytest.raises(ValueError, match="factor_name is empty"):
            CrossSectionalFactor(artifacts=arts)


# ---------------------------------------------------------------------------
# Method set + return contract (§3.3 / §3.13)
# ---------------------------------------------------------------------------

class TestMethodSet:
    @pytest.mark.parametrize("method_name", [
        "ic", "ic_ir", "hit_rate", "ic_trend",
        "quantile_spread", "monotonicity", "top_concentration",
        "turnover", "breakeven_cost", "net_spread", "oos_decay",
        "regime_ic", "multi_horizon_ic", "spanning_alpha",
    ])
    def test_all_methods_return_metric_output(self, noisy_panel, method_name):
        f = fl.factor(noisy_panel, "Mom_20D")
        result = getattr(f, method_name)()
        assert isinstance(result, MetricOutput), (
            f"{method_name} returned {type(result).__name__}, not MetricOutput"
        )

    def test_cs_factor_method_count(self):
        """CrossSectionalFactor exposes exactly 14 public metric methods."""
        methods = [
            m for m in dir(CrossSectionalFactor)
            if not m.startswith("_") and callable(getattr(CrossSectionalFactor, m))
            and m not in {"evaluate", "config", "factor_name"}
        ]
        assert len(methods) == 14, f"got methods={methods}"


# ---------------------------------------------------------------------------
# Cache behaviour (§3.4 / §3.5 / §3.6 / §3.12)
# ---------------------------------------------------------------------------

class TestCacheBehavior:
    def test_repeat_call_returns_same_cached_object(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        first = f.ic()
        second = f.ic()
        # Second call returns the stored (proxy-wrapped) MetricOutput.
        assert second is f.artifacts.metric_outputs["ic"]
        # First call also returns the stored version (written during call).
        assert first is f.artifacts.metric_outputs["ic"]
        assert first.value == second.value

    def test_override_bypasses_cache(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D", n_groups=5)
        base = f.quantile_spread()
        override_result = f.quantile_spread(n_groups=3)
        cached = f.artifacts.metric_outputs["quantile_spread"]
        # Cache reflects config-bound n_groups, not the override.
        assert cached.value == base.value
        assert cached is not override_result

    def test_override_returns_fresh_result(self, noisy_panel):
        """Override path should return a freshly-constructed MetricOutput
        (not the cached proxy-wrapped one)."""
        f = fl.factor(noisy_panel, "Mom_20D", n_groups=5)
        f.quantile_spread()  # populate cache
        cached = f.artifacts.metric_outputs["quantile_spread"]
        override = f.quantile_spread(n_groups=3)
        # Override did not return the cached object.
        assert override is not cached

    def test_evaluate_reuses_cached_standalone_call(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        ic_first = f.ic()
        # Now run full evaluate — it should reuse the cached ic entry.
        f.evaluate()
        ic_after = f.artifacts.metric_outputs["ic"]
        # Same cached object — the value propagated through from_artifacts
        # rather than being recomputed.
        assert ic_after.value == ic_first.value

    def test_standalone_after_evaluate_reuses_cache(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        f.evaluate()
        ic_before_cache = f.artifacts.metric_outputs["ic"]
        # Call standalone — should return the cached object, not recompute.
        ic_call = f.ic()
        assert ic_call is ic_before_cache


# ---------------------------------------------------------------------------
# fl.evaluate ≡ fl.factor().evaluate() value equivalence (§3.5)
# ---------------------------------------------------------------------------

class TestEvaluateEquivalence:
    def test_profile_fields_bit_equal(self, noisy_panel):
        cfg = CrossSectionalConfig(forward_periods=5, n_groups=5)
        p_direct = fl.evaluate(noisy_panel, "Mom_20D", config=cfg)
        p_via_factor = fl.factor(noisy_panel, "Mom_20D", config=cfg).evaluate()
        # Same typed class
        assert type(p_direct) is type(p_via_factor)
        # Every field equal (we compare dataclass dict)
        d_direct = _dc.asdict(p_direct)
        d_via = _dc.asdict(p_via_factor)
        assert d_direct == d_via


# ---------------------------------------------------------------------------
# Derived-metric wiring (breakeven_cost / net_spread use other metrics)
# ---------------------------------------------------------------------------

class TestDerivedMetrics:
    def test_breakeven_reads_spread_and_turnover(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        be = f.breakeven_cost()
        assert isinstance(be, MetricOutput)
        # Both inputs should now be cached
        assert "quantile_spread" in f.artifacts.metric_outputs
        assert "turnover" in f.artifacts.metric_outputs
        assert "breakeven_cost" in f.artifacts.metric_outputs

    def test_net_spread_reads_spread_and_turnover(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        ns = f.net_spread()
        assert isinstance(ns, MetricOutput)
        assert "net_spread" in f.artifacts.metric_outputs

    def test_breakeven_override_threads_n_groups(self, noisy_panel):
        """``f.breakeven_cost(n_groups=3)`` reflects the tercile spread,
        not the config-bound decile spread. Override path must not
        pollute the config-bound cache."""
        f = fl.factor(noisy_panel, "Mom_20D", n_groups=10)
        be_default = f.breakeven_cost()
        be_override = f.breakeven_cost(n_groups=3)
        cached = f.artifacts.metric_outputs["breakeven_cost"]
        # Cache still reflects default (override bypassed)
        assert cached.value == be_default.value
        assert be_override is not cached

    def test_net_spread_override_threads_cost(self, noisy_panel):
        """Cheaper cost → smaller turnover drag → larger net_spread,
        holding bucketing fixed. Override does not pollute cache."""
        f = fl.factor(noisy_panel, "Mom_20D", estimated_cost_bps=30.0)
        ns_default = f.net_spread()
        ns_cheap = f.net_spread(estimated_cost_bps=5.0)
        assert ns_cheap.value >= ns_default.value
        cached = f.artifacts.metric_outputs["net_spread"]
        assert cached.value == ns_default.value


# ---------------------------------------------------------------------------
# L2 opt-in short-circuit (§3.3)
# ---------------------------------------------------------------------------

class TestLevel2ShortCircuit:
    def test_regime_ic_not_configured(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        result = f.regime_ic()
        assert result.name == "regime_ic"
        assert result.value == 0.0
        assert result.metadata["reason"] == "no_regime_labels_configured"
        assert result.metadata["p_value"] == 1.0

    def test_short_circuit_is_cached(self, noisy_panel):
        """Short-circuit output is stashed on first miss so repeat calls
        return the cached object (no rebuild per call)."""
        f = fl.factor(noisy_panel, "Mom_20D")
        first = f.regime_ic()
        second = f.regime_ic()
        assert first is second
        assert "regime_ic" in f.artifacts.metric_outputs

    def test_multi_horizon_ic_not_configured(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        result = f.multi_horizon_ic()
        assert result.metadata["reason"] == "no_multi_horizon_periods_configured"

    def test_spanning_alpha_not_configured(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        result = f.spanning_alpha()
        assert result.metadata["reason"] == "no_spanning_base_spreads_configured"

    def test_regime_ic_with_regime_labels_returns_real_output(self, noisy_panel):
        # Build a simple 2-regime split: first half "bull", second half "bear".
        dates = sorted(noisy_panel["date"].unique().to_list())
        half = len(dates) // 2
        regime_df = pl.DataFrame({
            "date": dates,
            "regime": ["bull"] * half + ["bear"] * (len(dates) - half),
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        cfg = CrossSectionalConfig(regime_labels=regime_df)
        f = fl.factor(noisy_panel, "Mom_20D", config=cfg)
        result = f.regime_ic()
        # Should NOT be the short-circuit — real regime_ic computed at
        # build_artifacts time and retrieved from cache.
        assert "per_regime" in result.metadata
        assert result.metadata.get("reason") != "no_regime_labels_configured"


# ---------------------------------------------------------------------------
# Factor subclass / AttributeError for cross-type calls (§3.10)
# ---------------------------------------------------------------------------

class TestCrossTypeAccess:
    def test_cs_factor_has_no_caar(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        # caar() is event-signal metric; not on CrossSectionalFactor.
        with pytest.raises(AttributeError):
            f.caar()


# ---------------------------------------------------------------------------
# Escape hatch: direct artifacts access for downstream tools
# ---------------------------------------------------------------------------

class TestEscapeHatch:
    def test_can_call_describe_profile_values(self, noisy_panel, capsys):
        f = fl.factor(noisy_panel, "Mom_20D")
        profile = f.evaluate()
        fl.describe_profile_values(profile, f.artifacts)
        # Doesn't raise; output goes to stdout
        captured = capsys.readouterr()
        assert captured.out  # non-empty

    def test_metric_outputs_accessible(self, noisy_panel):
        f = fl.factor(noisy_panel, "Mom_20D")
        f.ic()
        assert "ic" in f.artifacts.metric_outputs
        m = f.artifacts.metric_outputs["ic"]
        assert isinstance(m, MetricOutput)
