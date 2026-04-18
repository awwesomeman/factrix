"""Parity snapshot: EventProfile (new) vs gate/profile pipeline (old).

Parallel to ``tests/test_parity_gate_vs_profile.py`` but covers the
event_signal factor type. Phase B will delete the gate/pipeline code;
this test is a deletion blocker — it freezes the numerical agreement
between both paths so the refactor cannot silently drift event-study
statistics.

Scope of parity:
- Raw event-study metric outputs (caar_tstat, caar_p, bmp_zstat,
  bmp_p, event_hit_rate, event_hit_rate_p) must match — both paths
  call the same metric functions over the same artifacts.
- verdict(threshold=2.0) PASS/FAILED must match the legacy
  ``event_significance`` gate output on the canonical-agreeing subset
  (CAAR drove the decision; BMP / hit_rate -only passes are a
  documented known divergence).

Known divergences (documented, not bugs):
- Legacy gate fired PASS on CAAR OR BMP OR hit_rate >= threshold; new
  verdict only reads CAAR (``caar_p``). When legacy passed via BMP or
  hit_rate alone, new verdict is FAILED by design — mirroring the CS
  "spread-only pass" divergence retired with gates.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.config import EventConfig
from factorlib.evaluation.pipeline import build_artifacts, evaluate as legacy_evaluate
from factorlib.evaluation.profiles import EventProfile


def _event_panel(
    n_dates: int, n_assets: int, event_rate: float, signal_coef: float, seed: int,
) -> pl.DataFrame:
    """Build an event panel with tunable event rate and signal-to-noise mix."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for i in range(n_assets):
            u = rng.random()
            if u < event_rate / 2:
                f = 1
            elif u < event_rate:
                f = -1
            else:
                f = 0
            noise = float(rng.standard_normal())
            r = signal_coef * f + (1 - abs(signal_coef)) * noise
            rows.append({
                "date": d, "asset_id": f"ev{i}",
                "factor": float(f), "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


FIXTURES = [
    ("strong_pos", 0.25, 0.5, 2001),
    ("moderate_pos", 0.25, 0.25, 2002),
    ("weak_pos", 0.25, 0.05, 2003),
    ("reverse", 0.25, -0.3, 2004),
    ("noise", 0.25, 0.0, 2005),
    ("dense_events", 0.5, 0.2, 2006),
]


@pytest.fixture(scope="module", params=FIXTURES, ids=[f[0] for f in FIXTURES])
def parity_case(request):
    name, event_rate, coef, seed = request.param
    df = _event_panel(
        n_dates=120, n_assets=15,
        event_rate=event_rate, signal_coef=coef, seed=seed,
    )
    config = EventConfig()
    art_new = build_artifacts(df, config)
    art_new.factor_name = name

    new_profile = EventProfile.from_artifacts(art_new)
    legacy = legacy_evaluate(df, name, config=config)
    return name, legacy, new_profile


class TestRawMetricParity:
    def test_caar_tstat(self, parity_case):
        _, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "event_significance")
        np.testing.assert_allclose(
            new.caar_tstat, gate.detail["caar_tstat"], rtol=1e-12,
        )

    def test_bmp_zstat(self, parity_case):
        _, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "event_significance")
        np.testing.assert_allclose(
            new.bmp_zstat, gate.detail["bmp_zstat"], rtol=1e-12,
        )

    def test_event_hit_rate(self, parity_case):
        _, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "event_significance")
        np.testing.assert_allclose(
            new.event_hit_rate, gate.detail["hit_rate"], rtol=1e-12,
        )

    def test_caar_p_matches_legacy_profile(self, parity_case):
        name, legacy, new = parity_case
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited on gate failure")
        caar_m = legacy.profile.get("caar")
        assert caar_m is not None
        np.testing.assert_allclose(
            new.caar_p, float(caar_m.metadata["p_value"]), rtol=1e-12,
        )

    def test_bmp_p_matches_legacy_profile(self, parity_case):
        name, legacy, new = parity_case
        if legacy.profile is None:
            pytest.skip(f"{name}: legacy short-circuited on gate failure")
        bmp_m = legacy.profile.get("bmp_sar")
        assert bmp_m is not None
        np.testing.assert_allclose(
            new.bmp_p, float(bmp_m.metadata["p_value"]), rtol=1e-12,
        )


class TestVerdictParity:
    """New verdict matches legacy gate on CAAR-canonical-agreeing cases."""

    def test_canonical_verdict_matches_when_caar_drove_gate(self, parity_case):
        name, legacy, new = parity_case
        gate = next(g for g in legacy.gate_results if g.name == "event_significance")
        via = gate.detail.get("via", [])

        # Case 1: legacy PASSED via CAAR alone → new verdict must PASS.
        if gate.status == "PASS" and via == ["CAAR"]:
            assert new.verdict(threshold=2.0) == "PASS", (
                f"{name}: legacy PASS via CAAR-only but new verdict != PASS"
            )
        # Case 2: legacy FAILED (no path crossed) → new verdict must FAIL.
        elif gate.status == "FAILED":
            assert new.verdict(threshold=2.0) == "FAILED", (
                f"{name}: legacy FAILED but new verdict != FAILED"
            )
        # Case 3: legacy passed via BMP / hit_rate (or combination). The
        # new canonical is CAAR alone; BMP-only and hit_rate-only passes
        # intentionally become FAILED under the canonical-binary policy.
        else:
            pytest.skip(
                f"{name}: legacy passed via {via}; new verdict is "
                f"canonical-CAAR-only by design"
            )


class TestKnownDivergences:
    def test_new_verdict_is_binary(self, parity_case):
        _, _, new = parity_case
        assert new.verdict(threshold=2.0) in ("PASS", "FAILED")

    def test_bmp_only_pass_surfaces_in_diagnose(self, parity_case):
        """When CAAR disagrees with BMP (opposite signs / inflation),
        diagnose must still fire the event.* rule so the old CAUTION
        information is not silently lost."""
        name, _, new = parity_case
        codes = {d.code for d in new.diagnose()}
        if (new.caar_mean * new.bmp_sar_mean) < 0 and abs(new.bmp_sar_mean) > 1e-9:
            assert "event.caar_bmp_sign_mismatch" in codes, (
                f"{name}: CAAR/BMP sign mismatch must surface as a diagnose"
            )
