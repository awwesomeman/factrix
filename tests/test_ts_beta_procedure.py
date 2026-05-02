"""v0.5 ``_TSBetaContTimeseriesProcedure.compute`` end-to-end tests.

Single-asset (N=1) panels: T daily observations, ``factor`` as a
broadcast continuous series, ``forward_return`` either correlated
with the factor (strong-β scenario) or independent (random scenario).
Covers plan §5.2 TIMESERIES continuous + I5 T-stratification + I6 ADF
persistence diagnose.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Mode, Signal
from factrix._codes import StatCode, Verdict, WarningCode
from factrix._errors import InsufficientSampleError
from factrix._evaluate import _evaluate
from factrix._procedures import InputSchema, _TSBetaContTimeseriesProcedure
from factrix._profile import FactorProfile
from factrix._registry import _DISPATCH_REGISTRY, _DispatchKey
from factrix._stats.constants import MIN_T_HARD, MIN_T_RELIABLE, auto_bartlett


def _make_ts(
    *,
    n_dates: int,
    seed: int,
    beta: float = 0.0,
    factor_kind: str = "iid",
) -> pl.DataFrame:
    """Build a single-asset panel with controllable β and factor process.

    ``factor_kind="iid"`` → factor ~ N(0,1) (stationary, ADF rejects).
    ``factor_kind="rw"`` → factor is a random walk (non-stationary,
    ADF should fail to reject → PERSISTENT_REGRESSOR warning).
    """
    rng = np.random.default_rng(seed)
    if factor_kind == "iid":
        factor = rng.standard_normal(n_dates)
    elif factor_kind == "rw":
        factor = np.cumsum(rng.standard_normal(n_dates))
    else:  # pragma: no cover
        raise ValueError(f"unknown factor_kind: {factor_kind}")
    eps = rng.standard_normal(n_dates)
    fwd = beta * factor + eps

    start = dt.date(2024, 1, 1)
    rows = [
        {
            "date": start + dt.timedelta(days=i),
            "asset_id": "SPY",
            "factor": float(factor[i]),
            "forward_return": float(fwd[i]),
        }
        for i in range(n_dates)
    ]
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg() -> AnalysisConfig:
    return AnalysisConfig.common_continuous()


class TestRegistryWiring:
    def test_registered_procedure_is_real(self) -> None:
        key = _DispatchKey(
            FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.TIMESERIES,
        )
        assert isinstance(
            _DISPATCH_REGISTRY[key].procedure,
            _TSBetaContTimeseriesProcedure,
        )

    def test_input_schema_columns(self) -> None:
        schema = _TSBetaContTimeseriesProcedure.INPUT_SCHEMA
        assert isinstance(schema, InputSchema)
        assert set(schema.required_columns) == {
            "date", "asset_id", "factor", "forward_return",
        }


class TestStrongBeta:
    @pytest.fixture(scope="class")
    def profile(self, cfg: AnalysisConfig) -> FactorProfile:
        ts = _make_ts(n_dates=120, seed=42, beta=0.8, factor_kind="iid")
        return _TSBetaContTimeseriesProcedure().compute(ts, cfg)

    def test_mode_timeseries(self, profile: FactorProfile) -> None:
        assert profile.mode is Mode.TIMESERIES

    def test_n_obs_equals_T(self, profile: FactorProfile) -> None:
        assert profile.n_obs == 120

    def test_passes_verdict(self, profile: FactorProfile) -> None:
        assert profile.verdict() is Verdict.PASS

    def test_low_primary_p(self, profile: FactorProfile) -> None:
        assert profile.primary_p < 0.001

    def test_beta_close_to_truth(self, profile: FactorProfile) -> None:
        # True β=0.8; finite-sample noise allows a 30% band.
        assert 0.55 < profile.stats[StatCode.TS_BETA] < 1.05

    def test_required_stats_keys_present(self, profile: FactorProfile) -> None:
        for key in (
            StatCode.TS_BETA,
            StatCode.TS_BETA_T_NW,
            StatCode.TS_BETA_P,
            StatCode.FACTOR_ADF_P,
            StatCode.NW_LAGS_USED,
        ):
            assert key in profile.stats

    def test_no_persistent_regressor_warning_on_iid_factor(
        self, profile: FactorProfile,
    ) -> None:
        assert WarningCode.PERSISTENT_REGRESSOR not in profile.warnings

    def test_no_unreliable_se_warning_at_T_120(
        self, profile: FactorProfile,
    ) -> None:
        assert WarningCode.UNRELIABLE_SE_SHORT_SERIES not in profile.warnings


class TestRandomFactor:
    @pytest.fixture(scope="class")
    def profile(self, cfg: AnalysisConfig) -> FactorProfile:
        ts = _make_ts(n_dates=120, seed=10, beta=0.0, factor_kind="iid")
        return _TSBetaContTimeseriesProcedure().compute(ts, cfg)

    def test_random_factor_fails(self, profile: FactorProfile) -> None:
        assert profile.verdict() is Verdict.FAIL
        assert profile.primary_p > 0.10

    def test_beta_near_zero(self, profile: FactorProfile) -> None:
        assert abs(profile.stats[StatCode.TS_BETA]) < 0.20


class TestPersistentRegressor:
    def test_random_walk_factor_triggers_warning(
        self, cfg: AnalysisConfig,
    ) -> None:
        ts = _make_ts(n_dates=120, seed=7, beta=0.0, factor_kind="rw")
        profile = _TSBetaContTimeseriesProcedure().compute(ts, cfg)
        assert WarningCode.PERSISTENT_REGRESSOR in profile.warnings
        assert profile.stats[StatCode.FACTOR_ADF_P] > 0.10


class TestSampleSizeStratification:
    def test_T_below_hard_floor_raises(self, cfg: AnalysisConfig) -> None:
        ts = _make_ts(n_dates=MIN_T_HARD - 1, seed=1, beta=0.5)
        with pytest.raises(InsufficientSampleError, match="MIN_T_HARD"):
            _TSBetaContTimeseriesProcedure().compute(ts, cfg)

    def test_T_in_warning_band_emits_warning(self, cfg: AnalysisConfig) -> None:
        # MIN_T_HARD <= T < MIN_T_RELIABLE → verdict + UNRELIABLE_SE warn.
        ts = _make_ts(n_dates=MIN_T_HARD, seed=2, beta=0.5)
        profile = _TSBetaContTimeseriesProcedure().compute(ts, cfg)
        assert WarningCode.UNRELIABLE_SE_SHORT_SERIES in profile.warnings
        assert profile.n_obs == MIN_T_HARD

    def test_T_at_reliable_floor_no_se_warning(
        self, cfg: AnalysisConfig,
    ) -> None:
        ts = _make_ts(n_dates=MIN_T_RELIABLE, seed=3, beta=0.5)
        profile = _TSBetaContTimeseriesProcedure().compute(ts, cfg)
        assert WarningCode.UNRELIABLE_SE_SHORT_SERIES not in profile.warnings


class TestNwLagFloor:
    def test_nw_lags_floor_at_forward_periods_minus_one(
        self, cfg: AnalysisConfig,
    ) -> None:
        ts = _make_ts(n_dates=60, seed=4, beta=0.5)
        profile = _TSBetaContTimeseriesProcedure().compute(ts, cfg)
        expected = float(max(auto_bartlett(60), cfg.forward_periods - 1))
        assert profile.stats[StatCode.NW_LAGS_USED] == expected


class TestEndToEndViaEvaluate:
    def test_evaluate_dispatches_to_ts_beta(
        self, cfg: AnalysisConfig,
    ) -> None:
        ts = _make_ts(n_dates=80, seed=99, beta=0.7)
        profile = _evaluate(ts, cfg)
        assert profile.mode is Mode.TIMESERIES
        assert StatCode.TS_BETA_P in profile.stats
        assert profile.verdict() is Verdict.PASS
