"""v0.5 ``_evaluate`` — mode derivation + registry dispatch + collapse."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import InfoCode, StatCode, Verdict
from factrix._errors import ModeAxisError
from factrix._evaluate import _derive_mode, _evaluate
from factrix._profile import FactorProfile


def _build_panel(
    *, n_dates: int, n_assets: int, seed: int, factor_strength: float = 0.0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    rows: list[dict[str, object]] = []
    for d in dates:
        fwd = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        factor = factor_strength * fwd + (1.0 - factor_strength) * noise
        for j in range(n_assets):
            rows.append({
                "date": d, "asset_id": f"A{j:03d}",
                "factor": float(factor[j]),
                "forward_return": float(fwd[j]),
            })
    return pl.DataFrame(rows)


def _build_timeseries(*, n_dates: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    rows = [
        {
            "date": d,
            "asset_id": "SPY",
            "factor": float(rng.standard_normal()),
            "forward_return": float(rng.standard_normal()),
        }
        for d in dates
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mode derivation
# ---------------------------------------------------------------------------


class TestDeriveMode:
    def test_panel_when_multi_asset(self) -> None:
        panel = _build_panel(n_dates=10, n_assets=15, seed=1)
        assert _derive_mode(panel) is Mode.PANEL

    def test_timeseries_when_single_asset(self) -> None:
        ts = _build_timeseries(n_dates=30, seed=1)
        assert _derive_mode(ts) is Mode.TIMESERIES

    def test_timeseries_when_zero_rows(self) -> None:
        empty = pl.DataFrame(schema={
            "date": pl.Date, "asset_id": pl.Utf8,
            "factor": pl.Float64, "forward_return": pl.Float64,
        })
        assert _derive_mode(empty) is Mode.TIMESERIES


# ---------------------------------------------------------------------------
# Happy path — IC PANEL end-to-end
# ---------------------------------------------------------------------------


class TestIcPanelEndToEnd:
    @pytest.fixture(scope="class")
    def profile(self) -> FactorProfile:
        panel = _build_panel(
            n_dates=60, n_assets=30, seed=42, factor_strength=0.95,
        )
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        return _evaluate(panel, cfg)

    def test_returns_factor_profile(self, profile: FactorProfile) -> None:
        assert isinstance(profile, FactorProfile)

    def test_mode_is_panel(self, profile: FactorProfile) -> None:
        assert profile.mode is Mode.PANEL

    def test_pass_verdict_on_strong_factor(self, profile: FactorProfile) -> None:
        assert profile.verdict() is Verdict.PASS

    def test_no_collapse_info_note(self, profile: FactorProfile) -> None:
        assert InfoCode.SCOPE_AXIS_COLLAPSED not in profile.info_notes

    def test_ic_p_populated(self, profile: FactorProfile) -> None:
        assert StatCode.IC_P in profile.stats


# ---------------------------------------------------------------------------
# ModeAxisError on undefined cell (§5.5)
# ---------------------------------------------------------------------------


class TestModeAxisError:
    def test_individual_continuous_n1_raises(self) -> None:
        ts = _build_timeseries(n_dates=40, seed=2)
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        with pytest.raises(ModeAxisError) as exc_info:
            _evaluate(ts, cfg)
        # Suggested fix points at the only legal CONTINUOUS path at N=1.
        assert exc_info.value.suggested_fix == AnalysisConfig.common_continuous()

    def test_individual_continuous_fm_n1_raises(self) -> None:
        ts = _build_timeseries(n_dates=40, seed=3)
        cfg = AnalysisConfig.individual_continuous(metric=Metric.FM)
        with pytest.raises(ModeAxisError):
            _evaluate(ts, cfg)


# ---------------------------------------------------------------------------
# Sparse N=1 collapse routing (§5.4.1)
# ---------------------------------------------------------------------------


class TestSparseCollapse:
    def test_individual_sparse_n1_routes_to_sentinel_entry(self) -> None:
        # Sentinel cell is now wired — verify the routing reaches a real
        # FactorProfile and the collapse InfoCode is attached.
        ts = _build_timeseries(n_dates=40, seed=4)
        profile = _evaluate(ts, AnalysisConfig.individual_sparse())
        assert profile.mode is Mode.TIMESERIES
        assert InfoCode.SCOPE_AXIS_COLLAPSED in profile.info_notes

    def test_common_sparse_n1_routes_to_same_sentinel_entry(self) -> None:
        ts = _build_timeseries(n_dates=40, seed=5)
        profile = _evaluate(ts, AnalysisConfig.common_sparse())
        assert profile.mode is Mode.TIMESERIES
        assert InfoCode.SCOPE_AXIS_COLLAPSED in profile.info_notes

    def test_panel_sparse_does_not_collapse(self) -> None:
        # PANEL routing keeps the user-facing scope intact; the CAAR
        # cell handles individual_sparse without going through the
        # _SCOPE_COLLAPSED sentinel (collapse only fires at N=1).
        panel = _build_panel(n_dates=30, n_assets=20, seed=6)
        # Inject a sparse trigger column so compute_caar has events.
        sparse_factor = (np.arange(len(panel)) % 11 == 0).astype(np.float64)
        panel = panel.with_columns(pl.Series("factor", sparse_factor))
        profile = _evaluate(panel, AnalysisConfig.individual_sparse())
        assert profile.mode is Mode.PANEL
        assert InfoCode.SCOPE_AXIS_COLLAPSED not in profile.info_notes


# ---------------------------------------------------------------------------
# Review fix TC-FALLBACK-None: monkey-patch the registry so a legal
# AnalysisConfig produces a registry miss with no _FALLBACK_MAP entry,
# verifying _evaluate's `suggested_fix=None` branch (the suffix="" path
# in the ModeAxisError construction) is exercised.
# ---------------------------------------------------------------------------


class TestFallbackNoneBranch:
    def test_unregistered_cell_with_no_fallback_raises_clean_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from factrix import _evaluate as evaluate_mod
        from factrix._registry import _DispatchKey, _DISPATCH_REGISTRY

        # Pop the live (COMMON, CONTINUOUS, PANEL) entry so a real
        # call with that config produces a registry miss. _FALLBACK_MAP
        # has no entry for (COMMON, CONTINUOUS, PANEL) — N≥2 paths are
        # always supposed to be wired — so the lookup yields None and
        # ModeAxisError is raised with suggested_fix=None.
        key = _DispatchKey(
            FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.PANEL,
        )
        original = _DISPATCH_REGISTRY.pop(key)
        try:
            panel = _build_panel(n_dates=30, n_assets=20, seed=11)
            with pytest.raises(ModeAxisError) as exc:
                evaluate_mod._evaluate(
                    panel, AnalysisConfig.common_continuous(),
                )
            assert exc.value.suggested_fix is None
            # Suffix should be empty — no "Suggested fix:" tail.
            assert "Suggested fix:" not in str(exc.value)
        finally:
            _DISPATCH_REGISTRY[key] = original
