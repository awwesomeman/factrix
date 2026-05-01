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
        ts = _build_timeseries(n_dates=20, seed=4)
        with pytest.raises(NotImplementedError, match="TSDummy"):
            _evaluate(ts, AnalysisConfig.individual_sparse())

    def test_common_sparse_n1_routes_to_same_sentinel_entry(self) -> None:
        ts = _build_timeseries(n_dates=20, seed=5)
        with pytest.raises(NotImplementedError, match="TSDummy"):
            _evaluate(ts, AnalysisConfig.common_sparse())

    def test_panel_sparse_routes_to_panel_entry(self) -> None:
        panel = _build_panel(n_dates=30, n_assets=20, seed=6)
        with pytest.raises(NotImplementedError, match="CAARSparse"):
            _evaluate(panel, AnalysisConfig.individual_sparse())


# ---------------------------------------------------------------------------
# Stub procedure NotImplementedError still surfaces
# ---------------------------------------------------------------------------


class TestStubsSurfaceThroughEvaluate:
    @pytest.mark.parametrize(
        "build,cfg_factory,match",
        [
            pytest.param(
                lambda: _build_panel(n_dates=30, n_assets=15, seed=8),
                lambda: AnalysisConfig.common_continuous(),
                "CommonContPanel",
                id="common_continuous_panel",
            ),
            pytest.param(
                lambda: _build_timeseries(n_dates=40, seed=9),
                lambda: AnalysisConfig.common_continuous(),
                "TSBetaCont",
                id="common_continuous_timeseries",
            ),
        ],
    )
    def test_stub_raises_with_procedure_name(
        self,
        build,
        cfg_factory,
        match: str,
    ) -> None:
        with pytest.raises(NotImplementedError, match=match):
            _evaluate(build(), cfg_factory())
