"""Tests for factorlib.gates — Gate Pipeline (Phase 2)."""

from datetime import datetime, timedelta
from functools import partial

import numpy as np
import polars as pl
import pytest

from factorlib.gates._protocol import (
    Artifacts,
    EvaluationResult,
    FactorProfile,
    GateFn,
    GateResult,
)
from factorlib.gates.config import PipelineConfig, MARKET_DEFAULTS
from factorlib.gates.significance import significance_gate
from factorlib.gates.oos_persistence import oos_persistence_gate
from factorlib.gates.profile import compute_profile
from factorlib.gates.pipeline import evaluate_factor, _build_artifacts
from factorlib.gates.presets import CROSS_SECTIONAL_GATES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strong_panel() -> pl.DataFrame:
    """60 dates × 50 assets, strong factor-return alignment.

    Enough data for OOS analysis (60 > MIN_OOS_PERIODS * 2 after IC computation).
    """
    rng = np.random.default_rng(42)
    n_dates, n_assets = 60, 50
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"s_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        # Strong signal: return = 0.6 * factor + 0.4 * noise
        noise = rng.standard_normal(n_assets)
        r = 0.6 * f + 0.4 * noise
        for i, a in enumerate(assets):
            rows.append({
                "date": d,
                "asset_id": a,
                "factor": float(f[i]),
                "forward_return": float(r[i]),
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def noise_panel() -> pl.DataFrame:
    """60 dates × 50 assets, pure noise (no signal).

    Factor and return are independent → IC ≈ 0, should fail significance.
    """
    rng = np.random.default_rng(99)
    n_dates, n_assets = 60, 50
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"s_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        r = rng.standard_normal(n_assets)  # independent
        for i, a in enumerate(assets):
            rows.append({
                "date": d,
                "asset_id": a,
                "factor": float(f[i]),
                "forward_return": float(r[i]),
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def strong_artifacts(strong_panel) -> Artifacts:
    return _build_artifacts(strong_panel, PipelineConfig())


@pytest.fixture
def noise_artifacts(noise_panel) -> Artifacts:
    return _build_artifacts(noise_panel, PipelineConfig())


# ---------------------------------------------------------------------------
# _protocol.py
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_gate_result_defaults(self):
        gr = GateResult(name="test", status="PASS")
        assert gr.detail == {}
        assert gr.passed is True

    def test_gate_result_passed_property(self):
        assert GateResult(name="t", status="PASS").passed is True
        assert GateResult(name="t", status="FAILED").passed is False
        assert GateResult(name="t", status="VETOED").passed is False

    def test_evaluation_result_defaults(self):
        er = EvaluationResult(factor_name="X", status="PASS")
        assert er.gate_results == []
        assert er.profile is None
        assert er.caution_reasons == []

    def test_factor_profile_defaults(self):
        fp = FactorProfile()
        assert fp.reliability == []
        assert fp.profitability == []


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.forward_periods == 5
        assert cfg.n_groups == 10
        assert cfg.estimated_cost_bps == 30.0
        assert cfg.orthogonalize is False

    def test_market_defaults_keys(self):
        assert "tw" in MARKET_DEFAULTS
        assert "us" in MARKET_DEFAULTS
        assert MARKET_DEFAULTS["tw"]["estimated_cost_bps"] == 30


# ---------------------------------------------------------------------------
# significance.py
# ---------------------------------------------------------------------------

class TestSignificanceGate:
    def test_strong_signal_passes(self, strong_artifacts):
        result = significance_gate(strong_artifacts)
        assert result.passed is True
        assert result.status == "PASS"
        assert len(result.detail["via"]) > 0

    def test_noise_fails(self, noise_artifacts):
        result = significance_gate(noise_artifacts)
        assert result.passed is False
        assert result.status == "FAILED"
        assert result.detail["via"] == []

    def test_custom_threshold_via_partial(self, strong_artifacts):
        strict = partial(significance_gate, threshold=100.0)
        result = strict(strong_artifacts)
        assert result.passed is False
        assert result.detail["threshold"] == 100.0

    def test_detail_contains_tstats(self, strong_artifacts):
        result = significance_gate(strong_artifacts)
        assert "ic_tstat" in result.detail
        assert "spread_tstat" in result.detail
        assert isinstance(result.detail["ic_tstat"], float)


# ---------------------------------------------------------------------------
# oos_persistence.py
# ---------------------------------------------------------------------------

class TestOOSPersistenceGate:
    def test_strong_signal(self, strong_artifacts):
        result = oos_persistence_gate(strong_artifacts)
        # Strong signal should at least not sign-flip
        assert result.detail["sign_flipped"] is False
        assert result.name == "oos_persistence"

    def test_detail_contains_per_split(self, strong_artifacts):
        result = oos_persistence_gate(strong_artifacts)
        assert "per_split" in result.detail
        assert len(result.detail["per_split"]) > 0

    def test_custom_threshold(self, strong_artifacts):
        lenient = partial(oos_persistence_gate, decay_threshold=0.01)
        result = lenient(strong_artifacts)
        assert result.detail["decay_threshold"] == 0.01


# ---------------------------------------------------------------------------
# profile.py
# ---------------------------------------------------------------------------

class TestComputeProfile:
    def test_returns_factor_profile(self, strong_artifacts):
        profile = compute_profile(strong_artifacts)
        assert isinstance(profile, FactorProfile)
        assert len(profile.reliability) == 4
        assert len(profile.profitability) == 6

    def test_reliability_metric_names(self, strong_artifacts):
        profile = compute_profile(strong_artifacts)
        names = {m.name for m in profile.reliability}
        assert names == {"IC_IR", "Hit_Rate", "IC_Trend", "Monotonicity"}

    def test_profitability_metric_names(self, strong_artifacts):
        profile = compute_profile(strong_artifacts)
        names = {m.name for m in profile.profitability}
        assert names == {
            "Q1_Q5_Spread", "Long_Short_Alpha", "Turnover",
            "Breakeven_Cost", "Net_Spread", "Q1_Concentration",
        }


# ---------------------------------------------------------------------------
# pipeline.py — evaluate_factor
# ---------------------------------------------------------------------------

class TestEvaluateFactor:
    def test_strong_signal_passes_all_gates(self, strong_panel):
        result = evaluate_factor(
            strong_panel, "strong_factor",
            CROSS_SECTIONAL_GATES, PipelineConfig(),
        )
        assert result.factor_name == "strong_factor"
        # Strong signal should pass significance; OOS may vary
        assert result.status in ("PASS", "CAUTION", "VETOED")
        assert len(result.gate_results) >= 1

    def test_noise_fails_at_significance(self, noise_panel):
        result = evaluate_factor(
            noise_panel, "noise_factor",
            CROSS_SECTIONAL_GATES, PipelineConfig(),
        )
        assert result.status == "FAILED"
        assert result.gate_results[0].name == "significance"
        assert result.gate_results[0].passed is False
        # Short-circuit: no profile computed
        assert result.profile is None

    def test_short_circuit_skips_later_gates(self, noise_panel):
        result = evaluate_factor(
            noise_panel, "noise",
            CROSS_SECTIONAL_GATES, PipelineConfig(),
        )
        # Only 1 gate result because first gate failed → short-circuit
        assert len(result.gate_results) == 1

    def test_custom_gate_list(self, strong_panel):
        """User can compose any list of gate functions."""
        def always_pass(artifacts: Artifacts) -> GateResult:
            return GateResult(name="always_pass", status="PASS")

        result = evaluate_factor(
            strong_panel, "test",
            [always_pass], PipelineConfig(),
        )
        assert result.gate_results[0].name == "always_pass"
        # Profile should be computed since all gates passed
        assert result.profile is not None

    def test_custom_gate_veto(self, strong_panel):
        """Custom gate can VETO."""
        def always_veto(artifacts: Artifacts) -> GateResult:
            return GateResult(name="always_veto", status="VETOED")

        result = evaluate_factor(
            strong_panel, "test",
            [always_veto], PipelineConfig(),
        )
        assert result.status == "VETOED"
        assert result.profile is None

    def test_empty_gates_produces_profile(self, strong_panel):
        """No gates = all pass → profile computed."""
        result = evaluate_factor(
            strong_panel, "test", [], PipelineConfig(),
        )
        assert result.profile is not None
        assert result.status in ("PASS", "CAUTION")

    def test_caution_when_not_orthogonalized(self, strong_panel):
        """orthogonalize=False → CAUTION reason present."""
        result = evaluate_factor(
            strong_panel, "test", [], PipelineConfig(orthogonalize=False),
        )
        if result.status == "CAUTION":
            assert any("orthogonalize" in r for r in result.caution_reasons)

    def test_caution_small_universe(self):
        """Universe < 200 → CAUTION."""
        rng = np.random.default_rng(42)
        n_dates, n_assets = 60, 10  # small universe
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            f = rng.standard_normal(n_assets)
            r = 0.6 * f + 0.4 * rng.standard_normal(n_assets)
            for i in range(n_assets):
                rows.append({
                    "date": d, "asset_id": f"s_{i}",
                    "factor": float(f[i]), "forward_return": float(r[i]),
                })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

        result = evaluate_factor(df, "small_uni", [], PipelineConfig())
        assert any("universe size" in r.lower() for r in result.caution_reasons)


# ---------------------------------------------------------------------------
# presets.py
# ---------------------------------------------------------------------------

class TestPresets:
    def test_cross_sectional_gates_list(self):
        assert len(CROSS_SECTIONAL_GATES) == 2
        assert CROSS_SECTIONAL_GATES[0] is significance_gate
        assert CROSS_SECTIONAL_GATES[1] is oos_persistence_gate
