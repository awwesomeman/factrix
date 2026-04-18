"""Parity snapshot: MacroCommonProfile (new) vs gate/profile pipeline (old).

Freezes numerical agreement for macro_common factors ahead of Phase B's
deletion of the gate/pipeline path.

Scope of parity:
- Raw TS β outputs (ts_beta_mean, ts_beta_tstat, ts_beta_p) must match.
- verdict(threshold=2.0) matches the legacy ``ts_significance`` gate —
  the legacy gate is already canonical-only (cross-sectional β t-stat)
  so there is no intermediate OR path; parity is tight.

Note: the macro_common legacy gate has no multi-path OR logic (unlike
event / fm), so there are no known divergences beyond the tri-state →
binary verdict collapse.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.config import MacroCommonConfig
from factorlib.evaluation.pipeline import build_artifacts, evaluate as legacy_evaluate
from factorlib.evaluation.profiles import MacroCommonProfile


def _macro_common(
    n_dates: int, n_assets: int, beta_true: float, seed: int,
) -> pl.DataFrame:
    """Common factor shared across assets; each asset has true β ± jitter."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    factor_series = rng.standard_normal(n_dates)
    asset_betas = beta_true + 0.1 * rng.standard_normal(n_assets)
    rows = []
    for t, d in enumerate(dates):
        for i in range(n_assets):
            r = asset_betas[i] * factor_series[t] + 0.2 * rng.standard_normal()
            rows.append({
                "date": d, "asset_id": f"a{i}",
                "factor": float(factor_series[t]),
                "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


FIXTURES = [
    ("strong_beta_+1.0", 1.0, 4001),
    ("moderate_+0.5", 0.5, 4002),
    ("weak_+0.1", 0.1, 4003),
    ("zero", 0.0, 4004),
    ("negative_-0.5", -0.5, 4005),
]


@pytest.fixture(scope="module", params=FIXTURES, ids=[f[0] for f in FIXTURES])
def parity_case(request):
    name, beta_true, seed = request.param
    df = _macro_common(n_dates=120, n_assets=15, beta_true=beta_true, seed=seed)
    config = MacroCommonConfig()
    art_new = build_artifacts(df, config)
    art_new.factor_name = name

    new_profile = MacroCommonProfile.from_artifacts(art_new)
    legacy = legacy_evaluate(df, name, config=config)
    return name, legacy, new_profile


class TestRawMetricParity:
    def test_ts_beta_tstat(self, parity_case):
        _, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "ts_significance")
        np.testing.assert_allclose(
            new.ts_beta_tstat, gate.detail["tstat"], rtol=1e-12,
        )

    def test_ts_beta_mean(self, parity_case):
        _, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "ts_significance")
        np.testing.assert_allclose(
            new.ts_beta_mean, gate.detail["mean_beta"], rtol=1e-12,
        )

    def test_ts_beta_p_matches_legacy_profile(self, parity_case):
        name, legacy, new = parity_case
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited on gate failure")
        ts_m = legacy.profile.get("ts_beta")
        assert ts_m is not None
        legacy_p = float(ts_m.metadata.get("p_value", 1.0))
        np.testing.assert_allclose(new.ts_beta_p, legacy_p, rtol=1e-12)


class TestVerdictParity:
    """Legacy ts_significance gate is canonical-only, so parity is tight."""

    def test_verdict_matches_gate(self, parity_case):
        name, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "ts_significance")
        if gate.status == "PASS":
            assert new.verdict(threshold=2.0) == "PASS", (
                f"{name}: legacy PASSED but new verdict != PASS"
            )
        else:
            assert new.verdict(threshold=2.0) == "FAILED", (
                f"{name}: legacy FAILED but new verdict != FAILED"
            )


class TestKnownDivergences:
    def test_new_verdict_is_binary(self, parity_case):
        _, _, new = parity_case
        assert new.verdict(threshold=2.0) in ("PASS", "FAILED")
