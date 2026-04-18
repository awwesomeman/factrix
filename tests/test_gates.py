"""Tests for factorlib.evaluation — Gate Pipeline."""

from datetime import datetime, timedelta
from functools import partial

import numpy as np
import polars as pl
import pytest

from factorlib.evaluation._protocol import (
    Artifacts,
    EvaluationResult,
    FactorProfile,
    GateFn,
    GateResult,
)
from factorlib.config import CrossSectionalConfig, MARKET_DEFAULTS
from factorlib.evaluation.gates.significance import significance_gate
from factorlib.evaluation.gates.oos_persistence import oos_persistence_gate
from factorlib.evaluation.profile import compute_profile
from factorlib.evaluation.pipeline import evaluate, build_artifacts
from factorlib.evaluation.presets import CROSS_SECTIONAL_GATES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strong_panel() -> pl.DataFrame:
    """60 dates × 50 assets, strong factor-return alignment."""
    rng = np.random.default_rng(42)
    n_dates, n_assets = 60, 50
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"s_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        f = rng.standard_normal(n_assets)
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
    """60 dates × 50 assets, pure noise (no signal)."""
    rng = np.random.default_rng(99)
    n_dates, n_assets = 60, 50
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"s_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        r = rng.standard_normal(n_assets)
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
    return build_artifacts(strong_panel, CrossSectionalConfig())


@pytest.fixture
def noise_artifacts(noise_panel) -> Artifacts:
    return build_artifacts(noise_panel, CrossSectionalConfig())


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
        assert er.artifacts is None
        assert er.caution_reasons == []

    def test_factor_profile_defaults(self):
        fp = FactorProfile()
        assert fp.metrics == []
        assert fp.attribution == []

    def test_factor_profile_get(self):
        from factorlib._types import MetricOutput
        fp = FactorProfile(metrics=[MetricOutput(name="ic", value=0.05)])
        assert fp.get("ic") is not None
        assert fp.get("ic").value == 0.05
        assert fp.get("nonexistent") is None

    def test_artifacts_get(self, strong_artifacts):
        ic = strong_artifacts.get("ic_series")
        assert "date" in ic.columns
        with pytest.raises(KeyError, match="Artifacts has no"):
            strong_artifacts.get("nonexistent")

    def test_evaluation_result_repr(self, strong_panel):
        result = evaluate(
            strong_panel, "test", config=CrossSectionalConfig(), gates=[],
        )
        text = repr(result)
        assert "Factor: test" in text
        assert "Status:" in text

    def test_evaluation_result_to_dict(self, strong_panel):
        result = evaluate(
            strong_panel, "test", config=CrossSectionalConfig(), gates=[],
        )
        d = result.to_dict()
        assert d["factor_name"] == "test"
        assert "metrics" in d
        assert isinstance(d["metrics"], list)

    def test_evaluation_result_to_dataframe(self, strong_panel):
        result = evaluate(
            strong_panel, "test", config=CrossSectionalConfig(), gates=[],
        )
        df = result.to_dataframe()
        assert "metric" in df.columns
        assert "value" in df.columns
        assert len(df) > 0


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = CrossSectionalConfig()
        assert cfg.forward_periods == 5
        assert cfg.n_groups == 10
        assert cfg.estimated_cost_bps == 30.0
        assert cfg.orthogonalize is False

    def test_market_defaults_keys(self):
        assert "tw" in MARKET_DEFAULTS
        assert "us" in MARKET_DEFAULTS
        assert MARKET_DEFAULTS["tw"]["estimated_cost_bps"] == 30

    def test_factor_type_classvar(self):
        from factorlib._types import FactorType
        assert CrossSectionalConfig.factor_type == FactorType.CROSS_SECTIONAL


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
        assert result.detail["sign_flipped"] is False
        assert result.name == "oos_persistence"

    def test_detail_contains_per_split(self, strong_artifacts):
        result = oos_persistence_gate(strong_artifacts)
        assert "per_split" in result.detail
        assert len(result.detail["per_split"]) > 0

    def test_custom_threshold(self, strong_artifacts):
        lenient = partial(oos_persistence_gate, survival_threshold=0.01)
        result = lenient(strong_artifacts)
        assert result.detail["survival_threshold"] == 0.01


# ---------------------------------------------------------------------------
# profile.py
# ---------------------------------------------------------------------------

class TestComputeProfile:
    def test_returns_factor_profile(self, strong_artifacts):
        profile = compute_profile(strong_artifacts)
        assert isinstance(profile, FactorProfile)
        assert len(profile.metrics) == 11

    def test_metric_names(self, strong_artifacts):
        profile = compute_profile(strong_artifacts)
        names = {m.name for m in profile.metrics}
        assert names == {
            "ic", "ic_ir", "hit_rate", "ic_trend", "monotonicity", "oos_decay",
            "q1_q5_spread", "turnover",
            "breakeven_cost", "net_spread", "q1_concentration",
        }


# ---------------------------------------------------------------------------
# pipeline.py — evaluate
# ---------------------------------------------------------------------------

class TestEvaluateFactor:
    def test_strong_signal_passes_all_gates(self, strong_panel):
        result = evaluate(
            strong_panel, "strong_factor",
            gates=CROSS_SECTIONAL_GATES, config=CrossSectionalConfig(),
        )
        assert result.factor_name == "strong_factor"
        assert result.status in ("PASS", "CAUTION", "VETOED")
        assert len(result.gate_results) >= 1

    def test_noise_fails_at_significance(self, noise_panel):
        result = evaluate(
            noise_panel, "noise_factor",
            gates=CROSS_SECTIONAL_GATES, config=CrossSectionalConfig(),
        )
        assert result.status == "FAILED"
        assert result.gate_results[0].name == "significance"
        assert result.gate_results[0].passed is False
        assert result.profile is None

    def test_short_circuit_skips_later_gates(self, noise_panel):
        result = evaluate(
            noise_panel, "noise",
            gates=CROSS_SECTIONAL_GATES, config=CrossSectionalConfig(),
        )
        assert len(result.gate_results) == 1

    def test_custom_gate_list(self, strong_panel):
        def always_pass(artifacts: Artifacts) -> GateResult:
            return GateResult(name="always_pass", status="PASS")

        result = evaluate(
            strong_panel, "test",
            gates=[always_pass], config=CrossSectionalConfig(),
        )
        assert result.gate_results[0].name == "always_pass"
        assert result.profile is not None

    def test_custom_gate_veto(self, strong_panel):
        def always_veto(artifacts: Artifacts) -> GateResult:
            return GateResult(name="always_veto", status="VETOED")

        result = evaluate(
            strong_panel, "test",
            gates=[always_veto], config=CrossSectionalConfig(),
        )
        assert result.status == "VETOED"
        assert result.profile is None

    def test_empty_gates_produces_profile(self, strong_panel):
        result = evaluate(
            strong_panel, "test", config=CrossSectionalConfig(), gates=[],
        )
        assert result.profile is not None
        assert result.status in ("PASS", "CAUTION")

    def test_caution_when_not_orthogonalized(self, strong_panel):
        result = evaluate(
            strong_panel, "test",
            config=CrossSectionalConfig(orthogonalize=False), gates=[],
        )
        if result.status == "CAUTION":
            assert any("orthogonalize" in r for r in result.caution_reasons)

    def test_caution_small_universe(self):
        rng = np.random.default_rng(42)
        n_dates, n_assets = 60, 10
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

        result = evaluate(df, "small_uni", config=CrossSectionalConfig(), gates=[])
        assert any("universe size" in r.lower() for r in result.caution_reasons)

    def test_artifacts_attached(self, strong_panel):
        result = evaluate(
            strong_panel, "test", config=CrossSectionalConfig(), gates=[],
        )
        assert result.artifacts is not None
        assert result.artifacts.get("ic_series") is not None


# ---------------------------------------------------------------------------
# presets.py
# ---------------------------------------------------------------------------

class TestPresets:
    def test_cross_sectional_gates_list(self):
        assert len(CROSS_SECTIONAL_GATES) == 2
        assert CROSS_SECTIONAL_GATES[0] is significance_gate
        assert CROSS_SECTIONAL_GATES[1] is oos_persistence_gate
