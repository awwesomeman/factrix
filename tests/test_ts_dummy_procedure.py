"""v0.5 ``_TSDummySparseTimeseriesProcedure.compute`` end-to-end tests.

Plan §5.2 TIMESERIES sparse + §5.4.1 sentinel collapse. Single-asset
panels with sparse ``{-1, 0, +1}`` triggers in the ``factor`` column
and ``forward_return`` either driven by the trigger (strong-β) or
independent (random).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Mode, Signal
from factrix._codes import InfoCode, StatCode, Verdict, WarningCode
from factrix._errors import InsufficientSampleError
from factrix._evaluate import _evaluate
from factrix._procedures import (
    InputSchema,
    _event_temporal_hhi,
    _has_event_window_overlap,
    _TSDummySparseTimeseriesProcedure,
)
from factrix._profile import FactorProfile
from factrix._registry import _DISPATCH_REGISTRY, _SCOPE_COLLAPSED, _DispatchKey
from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_WARN


def _make_sparse_ts(
    *,
    n_dates: int,
    seed: int,
    event_positions: list[int] | None = None,
    event_density: float = 0.05,
    beta: float = 0.0,
    noise_kind: str = "iid",
) -> pl.DataFrame:
    """Build a single-asset sparse panel.

    ``event_positions`` overrides random placement when provided.
    ``noise_kind="ar1"`` generates AR(1)(0.6) residuals to trigger
    Ljung-Box autocorrelation detection.
    """
    rng = np.random.default_rng(seed)
    d = np.zeros(n_dates)
    if event_positions is None:
        n_events = max(1, int(n_dates * event_density))
        positions = rng.choice(n_dates, size=n_events, replace=False)
    else:
        positions = np.array(event_positions)
    signs = rng.choice([-1.0, 1.0], size=len(positions))
    d[positions] = signs

    if noise_kind == "iid":
        eps = rng.standard_normal(n_dates)
    elif noise_kind == "ar1":
        eps = np.zeros(n_dates)
        innov = rng.standard_normal(n_dates)
        phi = 0.6
        eps[0] = innov[0]
        for t in range(1, n_dates):
            eps[t] = phi * eps[t - 1] + innov[t]
    else:  # pragma: no cover
        raise ValueError(f"unknown noise_kind: {noise_kind}")
    fwd = beta * d + eps

    start = dt.date(2024, 1, 1)
    rows = [
        {
            "date": start + dt.timedelta(days=i),
            "asset_id": "SPY",
            "factor": float(d[i]),
            "forward_return": float(fwd[i]),
        }
        for i in range(n_dates)
    ]
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg_individual() -> AnalysisConfig:
    return AnalysisConfig.individual_sparse()


@pytest.fixture(scope="module")
def cfg_common() -> AnalysisConfig:
    return AnalysisConfig.common_sparse()


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistryWiring:
    def test_sentinel_keyed_entry_is_real(self) -> None:
        key = _DispatchKey(
            _SCOPE_COLLAPSED,
            Signal.SPARSE,
            None,
            Mode.TIMESERIES,
        )
        assert isinstance(
            _DISPATCH_REGISTRY[key].procedure,
            _TSDummySparseTimeseriesProcedure,
        )

    def test_input_schema(self) -> None:
        schema = _TSDummySparseTimeseriesProcedure.INPUT_SCHEMA
        assert isinstance(schema, InputSchema)
        assert set(schema.required_columns) == {
            "date",
            "asset_id",
            "factor",
            "forward_return",
        }


# ---------------------------------------------------------------------------
# Strong / random / serial-correlation
# ---------------------------------------------------------------------------


class TestStrongTrigger:
    @pytest.fixture(scope="class")
    def profile(self, cfg_individual: AnalysisConfig) -> FactorProfile:
        # 12 well-spaced events on a 120-day series, β=2.0 → t very large.
        ts = _make_sparse_ts(
            n_dates=120,
            seed=42,
            event_positions=list(range(5, 120, 10)),
            beta=2.0,
        )
        return _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )

    def test_passes_verdict(self, profile: FactorProfile) -> None:
        assert profile.verdict() is Verdict.PASS

    def test_low_primary_p(self, profile: FactorProfile) -> None:
        assert profile.primary_p < 0.001

    def test_beta_close_to_truth(self, profile: FactorProfile) -> None:
        assert 1.5 < profile.stats[StatCode.TS_BETA] < 2.5

    def test_required_stats_present(self, profile: FactorProfile) -> None:
        for key in (
            StatCode.TS_BETA,
            StatCode.TS_BETA_T_NW,
            StatCode.TS_BETA_P,
            StatCode.LJUNG_BOX_P,
            StatCode.EVENT_TEMPORAL_HHI,
            StatCode.NW_LAGS_USED,
        ):
            assert key in profile.stats


class TestRandomSparse:
    def test_random_dummy_fails_at_default_threshold(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        ts = _make_sparse_ts(
            n_dates=120,
            seed=10,
            event_density=0.08,
            beta=0.0,
        )
        profile = _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )
        assert profile.verdict() is Verdict.FAIL


class TestSerialCorrelationWarning:
    def test_ar1_residuals_trigger_serial_correlation(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        ts = _make_sparse_ts(
            n_dates=200,
            seed=99,
            event_density=0.05,
            beta=0.0,
            noise_kind="ar1",
        )
        profile = _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )
        assert WarningCode.SERIAL_CORRELATION_DETECTED in profile.warnings
        assert profile.stats[StatCode.LJUNG_BOX_P] < 0.05


# ---------------------------------------------------------------------------
# Event-window overlap warning
# ---------------------------------------------------------------------------


class TestEventOverlapWarning:
    def test_clustered_events_trigger_overlap(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        # Three events three days apart — well below 2*forward_periods=10.
        ts = _make_sparse_ts(
            n_dates=120,
            seed=3,
            event_positions=[20, 23, 26],
            beta=0.0,
        )
        profile = _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )
        assert WarningCode.EVENT_WINDOW_OVERLAP in profile.warnings

    def test_well_spaced_events_do_not_trigger(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        ts = _make_sparse_ts(
            n_dates=120,
            seed=4,
            event_positions=list(range(5, 120, 20)),
            beta=0.0,
        )
        profile = _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )
        assert WarningCode.EVENT_WINDOW_OVERLAP not in profile.warnings


# ---------------------------------------------------------------------------
# T-stratification
# ---------------------------------------------------------------------------


class TestSampleSizeStratification:
    def test_T_below_hard_floor_raises(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        ts = _make_sparse_ts(
            n_dates=MIN_PERIODS_HARD - 1,
            seed=1,
            event_positions=[2, 6, 10],
            beta=1.0,
        )
        with pytest.raises(InsufficientSampleError):
            _TSDummySparseTimeseriesProcedure().compute(ts, cfg_individual)

    def test_T_in_warning_band_emits_warning(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        ts = _make_sparse_ts(
            n_dates=MIN_PERIODS_HARD,
            seed=2,
            event_positions=[3, 8, 13, 18],
            beta=1.0,
        )
        profile = _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in profile.warnings

    def test_T_at_reliable_floor_no_se_warning(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        ts = _make_sparse_ts(
            n_dates=MIN_PERIODS_WARN,
            seed=3,
            event_positions=list(range(4, 30, 5)),
            beta=1.0,
        )
        profile = _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS not in profile.warnings


# ---------------------------------------------------------------------------
# Sentinel routing — both individual_sparse and common_sparse converge
# ---------------------------------------------------------------------------


class TestSentinelCollapse:
    def test_individual_and_common_produce_identical_stats(
        self,
        cfg_individual: AnalysisConfig,
        cfg_common: AnalysisConfig,
    ) -> None:
        ts = _make_sparse_ts(
            n_dates=120,
            seed=55,
            event_positions=list(range(5, 120, 10)),
            beta=1.5,
        )
        prof_i = _evaluate(ts, cfg_individual)
        prof_c = _evaluate(ts, cfg_common)
        assert prof_i.stats == prof_c.stats
        assert prof_i.primary_p == prof_c.primary_p
        # Both paths attach the collapse info note.
        assert InfoCode.SCOPE_AXIS_COLLAPSED in prof_i.info_notes
        assert InfoCode.SCOPE_AXIS_COLLAPSED in prof_c.info_notes


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestEventTemporalHHI:
    def test_empty_signal_returns_zero(self) -> None:
        d = np.zeros(50)
        assert _event_temporal_hhi(d) == 0.0

    def test_uniform_events_near_inverse_n_bins(self) -> None:
        # 10 events spaced evenly across 100 positions, default n_bins=10.
        d = np.zeros(100)
        d[np.arange(5, 100, 10)] = 1.0
        assert _event_temporal_hhi(d) == pytest.approx(1 / 10)

    def test_clustered_events_near_one(self) -> None:
        # All events in the first bin → HHI = 1.0.
        d = np.zeros(100)
        d[:5] = 1.0
        assert _event_temporal_hhi(d) == 1.0


class TestEventWindowOverlap:
    def test_single_event_no_overlap(self) -> None:
        d = np.zeros(50)
        d[10] = 1.0
        assert _has_event_window_overlap(d, forward_periods=5) is False

    def test_close_events_flag_overlap(self) -> None:
        d = np.zeros(50)
        d[10] = 1.0
        d[14] = 1.0  # gap of 4 < 2*5
        assert _has_event_window_overlap(d, forward_periods=5) is True

    def test_far_events_no_overlap(self) -> None:
        d = np.zeros(50)
        d[10] = 1.0
        d[30] = 1.0  # gap of 20 > 2*5
        assert _has_event_window_overlap(d, forward_periods=5) is False


# ---------------------------------------------------------------------------
# Review fix TC-Hansen-Hodrick: NW lag floor pin for TS-dummy
# ---------------------------------------------------------------------------


class TestNWLagsFloorTSDummy:
    """Pins ``max(auto_bartlett(T), forward_periods - 1)`` lag selection
    on the TS-dummy procedure (mirrors the IC / FM / CAAR / TS-β tests
    that already pin the same Hansen-Hodrick floor on their cells)."""

    def test_short_series_uses_hh_floor(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        # T=24 → auto_bartlett(24)=int(4*0.24**(2/9))=int(2.93)=2; the
        # config's forward_periods=5 → HH floor = 5-1 = 4. Floor wins.
        ts = _make_sparse_ts(
            n_dates=24,
            seed=99,
            event_density=0.20,
            beta=2.0,
        )
        profile = _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )
        assert profile.stats[StatCode.NW_LAGS_USED] == 4.0

    def test_long_series_uses_auto_bartlett(
        self,
        cfg_individual: AnalysisConfig,
    ) -> None:
        # T=400 → auto_bartlett(400)=int(4*4**(2/9))=int(5.34)=5; HH
        # floor = 5-1 = 4. auto_bartlett wins.
        ts = _make_sparse_ts(
            n_dates=400,
            seed=7,
            event_density=0.05,
            beta=1.5,
        )
        profile = _TSDummySparseTimeseriesProcedure().compute(
            ts,
            cfg_individual,
        )
        assert profile.stats[StatCode.NW_LAGS_USED] == 5.0
