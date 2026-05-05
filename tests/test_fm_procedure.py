"""v0.5 ``_FMContPanelProcedure.compute`` end-to-end tests.

Mirrors the IC PANEL test setup: synthetic ``T × N`` panel with a
``factor_strength``-mixed signal. FM-λ is the per-date OLS slope of
``forward_return`` on ``factor``; under a strong-mix scenario λ → 1.0.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import StatCode, Verdict, WarningCode
from factrix._evaluate import _evaluate
from factrix._procedures import InputSchema, _FMContPanelProcedure
from factrix._profile import FactorProfile
from factrix._registry import _DISPATCH_REGISTRY, _DispatchKey
from factrix._stats.constants import auto_bartlett


def _make_panel(
    *,
    n_dates: int,
    n_assets: int,
    seed: int,
    factor_strength: float,
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
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(factor[j]),
                    "forward_return": float(fwd[j]),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def fm_config() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.FM)


class TestRegistryWiring:
    def test_registered_procedure_is_real_implementation(self) -> None:
        key = _DispatchKey(
            FactorScope.INDIVIDUAL,
            Signal.CONTINUOUS,
            Metric.FM,
            Mode.PANEL,
        )
        proc = _DISPATCH_REGISTRY[key].procedure
        assert isinstance(proc, _FMContPanelProcedure)

    def test_input_schema_columns(self) -> None:
        schema = _FMContPanelProcedure.INPUT_SCHEMA
        assert isinstance(schema, InputSchema)
        assert set(schema.required_columns) == {
            "date",
            "asset_id",
            "factor",
            "forward_return",
        }


class TestStrongFactor:
    @pytest.fixture(scope="class")
    def profile(self, fm_config: AnalysisConfig) -> FactorProfile:
        panel = _make_panel(
            n_dates=60,
            n_assets=30,
            seed=42,
            factor_strength=0.95,
        )
        return _FMContPanelProcedure().compute(panel, fm_config)

    def test_returns_factor_profile(self, profile: FactorProfile) -> None:
        assert isinstance(profile, FactorProfile)

    def test_mode_panel(self, profile: FactorProfile) -> None:
        assert profile.mode is Mode.PANEL

    def test_n_obs_equals_date_count(self, profile: FactorProfile) -> None:
        assert profile.n_obs == 60

    def test_passes_verdict(self, profile: FactorProfile) -> None:
        assert profile.verdict() is Verdict.PASS

    def test_low_primary_p(self, profile: FactorProfile) -> None:
        assert profile.primary_p < 0.01

    def test_lambda_mean_near_factor_strength(
        self,
        profile: FactorProfile,
    ) -> None:
        # factor = 0.95 * fwd + 0.05 * noise → OLS slope of fwd on factor
        # is dominated by the 0.95 component; λ should land in (0.5, 1.5).
        assert 0.5 < profile.stats[StatCode.FM_LAMBDA_MEAN] < 1.5

    def test_required_stats_keys_present(self, profile: FactorProfile) -> None:
        for key in (
            StatCode.FM_LAMBDA_MEAN,
            StatCode.FM_LAMBDA_T_NW,
            StatCode.FM_LAMBDA_P,
            StatCode.NW_LAGS_USED,
        ):
            assert key in profile.stats

    def test_primary_p_matches_fm_p_stat(self, profile: FactorProfile) -> None:
        assert profile.primary_p == profile.stats[StatCode.FM_LAMBDA_P]

    def test_nw_lags_floor_at_forward_periods_minus_one(
        self,
        profile: FactorProfile,
        fm_config: AnalysisConfig,
    ) -> None:
        expected = float(max(auto_bartlett(60), fm_config.forward_periods - 1))
        assert profile.stats[StatCode.NW_LAGS_USED] == expected


class TestRandomFactor:
    @pytest.fixture(scope="class")
    def profile(self, fm_config: AnalysisConfig) -> FactorProfile:
        # T=120, seed=10 — same clean-null seed used by the IC random
        # test (factor independent of returns → λ-mean ≈ 0, p well > 0.10).
        panel = _make_panel(
            n_dates=120,
            n_assets=30,
            seed=10,
            factor_strength=0.0,
        )
        return _FMContPanelProcedure().compute(panel, fm_config)

    def test_random_factor_fails_at_default_threshold(
        self,
        profile: FactorProfile,
    ) -> None:
        assert profile.verdict() is Verdict.FAIL
        assert profile.primary_p > 0.10

    def test_lambda_mean_near_zero(self, profile: FactorProfile) -> None:
        assert abs(profile.stats[StatCode.FM_LAMBDA_MEAN]) < 0.10


class TestEndToEndViaEvaluate:
    def test_evaluate_dispatches_to_fm_procedure(
        self,
        fm_config: AnalysisConfig,
    ) -> None:
        panel = _make_panel(
            n_dates=40,
            n_assets=20,
            seed=123,
            factor_strength=0.9,
        )
        profile = _evaluate(panel, fm_config)
        assert isinstance(profile, FactorProfile)
        assert StatCode.FM_LAMBDA_P in profile.stats
        assert profile.verdict() is Verdict.PASS


class TestFMShortPeriodsWarning:
    """`(INDIVIDUAL, CONTINUOUS, FM, PANEL)` propagates short-periods warning."""

    def test_borderline_periods_emits_warning(
        self,
        fm_config: AnalysisConfig,
    ) -> None:
        # MIN_FM_PERIODS_HARD=4 ≤ n_periods=10 < MIN_FM_PERIODS_WARN=30
        panel = _make_panel(
            n_dates=10,
            n_assets=15,
            seed=99,
            factor_strength=0.5,
        )
        profile = _FMContPanelProcedure().compute(panel, fm_config)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in profile.warnings

    def test_long_periods_silent(self, fm_config: AnalysisConfig) -> None:
        panel = _make_panel(
            n_dates=60,
            n_assets=15,
            seed=100,
            factor_strength=0.5,
        )
        profile = _FMContPanelProcedure().compute(panel, fm_config)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS not in profile.warnings
