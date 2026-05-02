"""v0.5 ``_ICContPanelProcedure.compute`` end-to-end tests.

Synthetic panel: ``T`` dates × ``N`` assets. ``forward_return`` is
i.i.d. standard normal; the strong-factor scenario builds the factor
as ``forward_return + small_noise`` so per-date Spearman IC is
near-perfect, the random-factor scenario builds it as independent
noise. Verdict / primary_p / stats keys are the contract.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import StatCode, Verdict
from factrix._procedures import InputSchema, _ICContPanelProcedure
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
    """Build a (date, asset_id, factor, forward_return) panel.

    ``factor_strength`` ∈ [0, 1] mixes the future return into the
    factor: ``factor = strength * fwd_return + (1 - strength) * noise``.
    Strength=1.0 → near-perfect IC; strength=0.0 → pure noise.
    """
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    assets = [f"A{i:03d}" for i in range(n_assets)]

    rows: list[dict[str, object]] = []
    for d in dates:
        fwd = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        factor = factor_strength * fwd + (1.0 - factor_strength) * noise
        for a, f, r in zip(assets, factor, fwd):
            rows.append({
                "date": d, "asset_id": a,
                "factor": float(f), "forward_return": float(r),
            })
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def ic_config() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.IC)


class TestRegistryWiring:
    def test_registered_procedure_is_real_implementation(self) -> None:
        key = _DispatchKey(
            FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC, Mode.PANEL,
        )
        proc = _DISPATCH_REGISTRY[key].procedure
        assert isinstance(proc, _ICContPanelProcedure)

    def test_input_schema_columns(self) -> None:
        schema = _ICContPanelProcedure.INPUT_SCHEMA
        assert isinstance(schema, InputSchema)
        assert set(schema.required_columns) == {
            "date", "asset_id", "factor", "forward_return",
        }


class TestStrongFactor:
    @pytest.fixture(scope="class")
    def profile(self, ic_config: AnalysisConfig) -> FactorProfile:
        panel = _make_panel(
            n_dates=60, n_assets=30, seed=42, factor_strength=0.95,
        )
        return _ICContPanelProcedure().compute(panel, ic_config)

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

    def test_positive_ic_mean(self, profile: FactorProfile) -> None:
        assert profile.stats[StatCode.IC_MEAN] > 0.5

    def test_required_stats_keys_present(self, profile: FactorProfile) -> None:
        for key in (
            StatCode.IC_MEAN,
            StatCode.IC_T_NW,
            StatCode.IC_P,
            StatCode.NW_LAGS_USED,
        ):
            assert key in profile.stats

    def test_primary_p_matches_ic_p_stat(self, profile: FactorProfile) -> None:
        assert profile.primary_p == profile.stats[StatCode.IC_P]

    def test_nw_lags_floor_at_forward_periods_minus_one(
        self, profile: FactorProfile, ic_config: AnalysisConfig,
    ) -> None:
        # auto_bartlett(60)=3, forward_periods-1=4 → HH floor binds.
        expected = float(max(auto_bartlett(60), ic_config.forward_periods - 1))
        assert profile.stats[StatCode.NW_LAGS_USED] == expected


class TestRandomFactor:
    @pytest.fixture(scope="class")
    def profile(self, ic_config: AnalysisConfig) -> FactorProfile:
        # Smoke-test on a clean-null sample. Under H0 ~5% of random
        # factors PASS at α=0.05 by definition, so a single-seed
        # assertion is only meaningful with a seed empirically far
        # from the rejection band. seed=10 lands at primary_p ≈ 0.92,
        # ic_mean ≈ -0.002 — uncontroversially null.
        panel = _make_panel(
            n_dates=120, n_assets=30, seed=10, factor_strength=0.0,
        )
        return _ICContPanelProcedure().compute(panel, ic_config)

    def test_random_factor_fails_at_default_threshold(
        self, profile: FactorProfile,
    ) -> None:
        assert profile.verdict() is Verdict.FAIL
        assert profile.primary_p > 0.10

    def test_ic_mean_near_zero(self, profile: FactorProfile) -> None:
        assert abs(profile.stats[StatCode.IC_MEAN]) < 0.05


class TestProfileConfigPassthrough:
    def test_config_attached_to_profile(
        self, ic_config: AnalysisConfig,
    ) -> None:
        panel = _make_panel(
            n_dates=20, n_assets=15, seed=7, factor_strength=0.5,
        )
        profile = _ICContPanelProcedure().compute(panel, ic_config)
        assert profile.config is ic_config


class TestSparsePanelDropsThinDates:
    def test_thin_dates_dropped_via_min_ic_periods(
        self, ic_config: AnalysisConfig,
    ) -> None:
        # Build 30 dates × 30 assets, then drop assets on the first 10
        # dates so they fall below MIN_IC_PERIODS=10.
        panel = _make_panel(
            n_dates=30, n_assets=30, seed=11, factor_strength=0.7,
        )
        first_10 = panel["date"].unique().sort()[:10].to_list()
        thinned = pl.concat([
            panel.filter(~pl.col("date").is_in(first_10)),
            panel.filter(pl.col("date").is_in(first_10)).group_by(
                "date", maintain_order=True,
            ).head(5),
        ]).sort("date", "asset_id")

        profile = _ICContPanelProcedure().compute(thinned, ic_config)
        assert profile.n_obs == 20
