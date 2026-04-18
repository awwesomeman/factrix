"""Parity snapshot: MacroPanelProfile (new) vs gate/profile pipeline (old).

Freezes numerical agreement for macro_panel factors ahead of Phase B's
deletion of the gate/pipeline path.

Scope of parity:
- Raw FM/pooled outputs (fm_beta_tstat, fm_beta_p, pooled_beta_tstat,
  pooled_beta_p) must match.
- verdict(threshold=2.0) matches the legacy ``fm_significance`` gate
  on the FM-canonical-agreeing subset; pooled-only passes are the
  known divergence (new canonical is FM alone).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.config import MacroPanelConfig
from factorlib.evaluation.pipeline import build_artifacts, evaluate as legacy_evaluate
from factorlib.evaluation.profiles import MacroPanelProfile


def _macro_panel(
    n_dates: int, n_countries: int, signal: float, seed: int,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        fvals = rng.standard_normal(n_countries)
        for i in range(n_countries):
            r = signal * fvals[i] + (1 - abs(signal)) * rng.standard_normal()
            rows.append({
                "date": d, "asset_id": f"c{i}",
                "factor": float(fvals[i]), "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


FIXTURES = [
    ("strong_+0.5", 0.5, 3001),
    ("good_+0.3", 0.3, 3002),
    ("marginal_+0.1", 0.1, 3003),
    ("noise_+0.0", 0.0, 3004),
    ("reverse_-0.3", -0.3, 3005),
]


@pytest.fixture(scope="module", params=FIXTURES, ids=[f[0] for f in FIXTURES])
def parity_case(request):
    name, coef, seed = request.param
    df = _macro_panel(n_dates=80, n_countries=12, signal=coef, seed=seed)
    config = MacroPanelConfig()
    art_new = build_artifacts(df, config)
    art_new.factor_name = name

    new_profile = MacroPanelProfile.from_artifacts(art_new)
    legacy = legacy_evaluate(df, name, config=config)
    return name, legacy, new_profile


class TestRawMetricParity:
    def test_fm_tstat(self, parity_case):
        _, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "fm_significance")
        np.testing.assert_allclose(
            new.fm_beta_tstat, gate.detail["fm_tstat"], rtol=1e-12,
        )

    def test_pooled_tstat(self, parity_case):
        _, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "fm_significance")
        np.testing.assert_allclose(
            new.pooled_beta_tstat, gate.detail["pooled_tstat"], rtol=1e-12,
        )

    def test_fm_p_matches_legacy_profile(self, parity_case):
        name, legacy, new = parity_case
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited")
        fm_m = legacy.profile.get("fm_beta")
        assert fm_m is not None
        np.testing.assert_allclose(
            new.fm_beta_p, float(fm_m.metadata["p_value"]), rtol=1e-12,
        )

    def test_pooled_p_matches_legacy_profile(self, parity_case):
        name, legacy, new = parity_case
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited")
        pooled_m = legacy.profile.get("pooled_beta")
        assert pooled_m is not None
        np.testing.assert_allclose(
            new.pooled_beta_p, float(pooled_m.metadata["p_value"]), rtol=1e-12,
        )


class TestVerdictParity:
    def test_canonical_verdict_matches_when_fm_drove_gate(self, parity_case):
        name, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "fm_significance")
        via = gate.detail.get("via", [])

        if gate.status == "PASS" and via == ["FM_beta"]:
            assert new.verdict(threshold=2.0) == "PASS", (
                f"{name}: legacy PASS via FM_beta-only but new verdict != PASS"
            )
        elif gate.status == "FAILED":
            assert new.verdict(threshold=2.0) == "FAILED", (
                f"{name}: legacy FAILED but new verdict != FAILED"
            )
        else:
            pytest.skip(
                f"{name}: legacy passed via {via}; new verdict is "
                f"canonical-FM-only by design"
            )


class TestKnownDivergences:
    def test_new_verdict_is_binary(self, parity_case):
        _, _, new = parity_case
        assert new.verdict(threshold=2.0) in ("PASS", "FAILED")

    def test_fm_pooled_sign_mismatch_surfaces_in_diagnose(self, parity_case):
        """Old CAUTION fired on (FM β · pooled β) < 0; the new rule
        keeps the same behaviour via macro_panel.fm_pooled_sign_mismatch."""
        name, _, new = parity_case
        codes = {d.code for d in new.diagnose()}
        if (new.fm_beta_mean * new.pooled_beta) < 0 and abs(new.pooled_beta) > 1e-9:
            assert "macro_panel.fm_pooled_sign_mismatch" in codes, (
                f"{name}: FM/pooled sign mismatch must surface as a diagnose"
            )
