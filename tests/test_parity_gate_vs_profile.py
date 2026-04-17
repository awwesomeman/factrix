"""Parity snapshot: new profile-era path vs old gate-era path.

Confirms the refactor has not silently drifted the underlying
numerical metric outputs or the PASS/FAILED verdict for typical
cross-sectional factors.

Scope:
- Raw metric numbers (ic_tstat, ic_p, spread_tstat, spread_p) must
  match because both paths call the same metric functions.
- The profile's verdict(threshold=2.0) must match the old gate's
  "significance" gate PASS when the canonical test (IC) agreed.

Known divergences (documented, not bugs):
- Old verdict was tri-state {PASS, CAUTION, VETOED}; new verdict is
  binary {PASS, FAILED}. Old CAUTION cases -- factors that passed
  significance via Q1-Q5 spread only -- now show up as FAILED verdict
  plus a warn-severity diagnose() hit. The parity test covers the
  canonical-agreeing subset.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.config import CrossSectionalConfig
from factorlib.evaluation.pipeline import build_artifacts, evaluate as legacy_evaluate
from factorlib.evaluation.profiles import CrossSectionalProfile


def _panel(n_dates: int, n_assets: int, signal_coef: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        r = signal_coef * f + (1 - abs(signal_coef)) * noise
        for i in range(n_assets):
            rows.append({
                "date": d, "asset_id": f"a{i}",
                "factor": float(f[i]), "forward_return": float(r[i]),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


# Parametrize over a handful of strength levels so the snapshot spans
# both "definitely significant" and "marginal" regimes.
FIXTURES = [
    ("strong_+0.5", 0.5, 1001),
    ("good_+0.3", 0.3, 1002),
    ("marginal_+0.1", 0.1, 1003),
    ("noise_+0.0", 0.0, 1004),
    ("reverse_-0.3", -0.3, 1005),
    ("strong_-0.5", -0.5, 1006),
    ("moderate_+0.2", 0.2, 1007),
]


@pytest.fixture(params=FIXTURES, ids=[f[0] for f in FIXTURES])
def parity_case(request):
    name, coef, seed = request.param
    df = _panel(n_dates=60, n_assets=30, signal_coef=coef, seed=seed)
    config = CrossSectionalConfig()
    # Build artifacts once; both paths reuse them.
    art_old = build_artifacts(df, config)
    art_new = build_artifacts(df, config)
    art_new.factor_name = name

    new_profile = CrossSectionalProfile.from_artifacts(art_new)
    legacy = legacy_evaluate(df, name, config=config)
    return name, legacy, new_profile


class TestRawMetricParity:
    """Both paths call the same metric functions so the numbers must match."""

    def test_ic_tstat(self, parity_case):
        name, legacy, new = parity_case
        # legacy.profile may be None on FAILED; when present, pull IC.
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited on gate failure")
        legacy_ic = legacy.profile.get("ic")
        assert legacy_ic is not None
        np.testing.assert_allclose(new.ic_tstat, legacy_ic.stat, rtol=1e-12)

    def test_ic_p(self, parity_case):
        name, legacy, new = parity_case
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited on gate failure")
        legacy_ic = legacy.profile.get("ic")
        assert legacy_ic is not None
        legacy_p = float(legacy_ic.metadata["p_value"])
        np.testing.assert_allclose(new.ic_p, legacy_p, rtol=1e-12)

    def test_spread_tstat(self, parity_case):
        name, legacy, new = parity_case
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited")
        legacy_spread = legacy.profile.get("q1_q5_spread")
        assert legacy_spread is not None
        np.testing.assert_allclose(
            new.spread_tstat, legacy_spread.stat, rtol=1e-12,
        )

    def test_spread_p(self, parity_case):
        name, legacy, new = parity_case
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited")
        legacy_spread = legacy.profile.get("q1_q5_spread")
        assert legacy_spread is not None
        np.testing.assert_allclose(
            new.spread_p,
            float(legacy_spread.metadata["p_value"]),
            rtol=1e-12,
        )


class TestVerdictParity:
    """For the canonical-agreeing subset, new verdict should match old gate."""

    def test_canonical_verdict_matches_significance_gate(self, parity_case):
        name, legacy, new = parity_case
        # Old path: gate "significance" exists and reports PASS iff either
        # IC or spread crossed threshold. We only parity-check when IC
        # alone drove the pass/fail decision.
        sig_gate = next(
            (g for g in legacy.gate_results if g.name == "significance"),
            None,
        )
        if sig_gate is None:
            pytest.skip(f"{name}: no significance gate in legacy result")

        via = sig_gate.detail.get("via", [])
        # Case 1: old PASS via IC-only → new must PASS at threshold 2.0
        if sig_gate.status == "PASS" and via == ["IC"]:
            assert new.verdict(threshold=2.0) == "PASS", (
                f"{name}: legacy PASS via IC but new verdict != PASS"
            )
        # Case 2: old FAILED (no IC and no spread pass) → new must FAIL
        elif sig_gate.status == "FAILED":
            assert new.verdict(threshold=2.0) == "FAILED", (
                f"{name}: legacy FAILED but new verdict != FAILED"
            )
        # Case 3: PASS via spread-only or via both: not a canonical-agree
        # case. Document the expected behaviour: new verdict looks only
        # at IC, so spread-only cases become FAILED (not parity-checked).
        else:
            pytest.skip(
                f"{name}: legacy passed via {via}; "
                f"new verdict is canonical-IC-only by design"
            )


class TestKnownDivergences:
    """Document the expected (not a bug) behavioural differences."""

    def test_new_verdict_is_binary(self, parity_case):
        _, _, new = parity_case
        assert new.verdict(threshold=2.0) in ("PASS", "FAILED")
        # Specifically: "CAUTION" is NOT a legal verdict output.
        assert new.verdict(threshold=2.0) != "CAUTION"

    def test_spread_only_case_warns_in_diagnose(self, parity_case):
        """Cases that would have been CAUTION under gates (IC fails,
        spread succeeds) should now surface as a diagnose() warning
        even if verdict is FAILED."""
        name, _, new = parity_case
        if new.ic_p > 0.05 and new.spread_p <= 0.05:
            codes = {d.code for d in new.diagnose()}
            assert "cs.ic_weak_spread_strong" in codes, (
                f"{name}: expected cs.ic_weak_spread_strong diagnose "
                f"when IC weak and spread strong."
            )
