"""Smoke test mirroring the README ``Typical usage`` quickstart.

Guards against silent drift between the README's import path and the
public surface — if ``compute_forward_return`` is moved or the
re-export is removed, this test fails before the README does.
"""

from __future__ import annotations

import factrix as fx
from factrix.preprocess import compute_forward_return


def test_readme_quickstart_runs_end_to_end() -> None:
    raw = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=500, ic_target=0.08, seed=2024
    )
    panel = compute_forward_return(raw, forward_periods=5)

    cfg = fx.AnalysisConfig.individual_continuous(
        metric=fx.Metric.IC, forward_periods=5
    )
    profile = fx.evaluate(panel, cfg)

    assert profile.verdict() in (fx.Verdict.PASS, fx.Verdict.FAIL)
    assert isinstance(profile.primary_p, float)
    assert 0.0 <= profile.primary_p <= 1.0


def test_compute_forward_return_is_reachable_from_namespace() -> None:
    assert fx.preprocess.compute_forward_return is compute_forward_return
