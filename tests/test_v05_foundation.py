"""v0.5 foundation tests — enums, constants, exceptions, AnalysisConfig."""

from __future__ import annotations

import dataclasses

import pytest

from factrix._analysis_config import _FALLBACK_MAP, AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import InfoCode, StatCode, Verdict, WarningCode
from factrix._errors import (
    ConfigError,
    FactrixError,
    IncompatibleAxisError,
    InsufficientSampleError,
    ModeAxisError,
)
from factrix._stats.constants import MIN_T_HARD, MIN_T_RELIABLE, auto_bartlett


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestAxisEnums:
    def test_factor_scope_values(self) -> None:
        assert FactorScope.INDIVIDUAL.value == "individual"
        assert FactorScope.COMMON.value == "common"

    def test_signal_values(self) -> None:
        assert Signal.CONTINUOUS.value == "continuous"
        assert Signal.SPARSE.value == "sparse"

    def test_metric_values(self) -> None:
        assert Metric.IC.value == "ic"
        assert Metric.FM.value == "fm"

    def test_mode_values(self) -> None:
        assert Mode.PANEL.value == "panel"
        assert Mode.TIMESERIES.value == "timeseries"


class TestCodeEnums:
    def test_verdict_values(self) -> None:
        assert Verdict.PASS.value == "pass"
        assert Verdict.FAIL.value == "fail"

    def test_warning_codes_present(self) -> None:
        # Spot-check the four live codes (§7.3) — each is raised by at
        # least one procedure in factrix/_procedures.py.
        for code in (
            WarningCode.UNRELIABLE_SE_SHORT_SERIES,
            WarningCode.EVENT_WINDOW_OVERLAP,
            WarningCode.PERSISTENT_REGRESSOR,
            WarningCode.SERIAL_CORRELATION_DETECTED,
        ):
            assert isinstance(code.value, str)
            assert isinstance(code.description, str) and code.description

    def test_info_code_scope_collapsed(self) -> None:
        assert InfoCode.SCOPE_AXIS_COLLAPSED.value == "scope_axis_collapsed"
        assert InfoCode.SCOPE_AXIS_COLLAPSED.description

    def test_warning_descriptions_cover_every_member(self) -> None:
        # Review fix UX-4: every WarningCode has a human-readable gloss.
        for code in WarningCode:
            assert code.description, f"{code} missing description"

    def test_stat_code_population(self) -> None:
        # Representative stats for each metric family + diagnostics.
        assert StatCode.IC_MEAN.value == "ic_mean"
        assert StatCode.FM_LAMBDA_T_NW.value == "fm_lambda_t_nw"
        assert StatCode.TS_BETA_P.value == "ts_beta_p"
        assert StatCode.CAAR_MEAN.value == "caar_mean"
        assert StatCode.NW_LAGS_USED.value == "nw_lags_used"


# ---------------------------------------------------------------------------
# Stats constants
# ---------------------------------------------------------------------------


class TestStatsConstants:
    def test_thresholds_ordered(self) -> None:
        assert MIN_T_HARD < MIN_T_RELIABLE

    def test_threshold_values(self) -> None:
        # Pinned per §5.2 — these are statistical contract, not a default.
        assert MIN_T_HARD == 20
        assert MIN_T_RELIABLE == 30

    def test_auto_bartlett_floor(self) -> None:
        assert auto_bartlett(1) == 1

    def test_auto_bartlett_typical(self) -> None:
        # Newey & West (1994): floor(4 * (T/100) ** (2/9)).
        # T=10  → 4 * 0.1**(2/9)  ≈ 2.398 → 2
        # T=100 → 4 * 1.0**(2/9)  = 4
        # T=1000 → 4 * 10**(2/9)  ≈ 6.669 → 6
        assert auto_bartlett(10) == 2
        assert auto_bartlett(100) == 4
        assert auto_bartlett(1000) == 6


# ---------------------------------------------------------------------------
# Exception hierarchy (§4.5)
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_config_error_is_factrix_error(self) -> None:
        assert issubclass(ConfigError, FactrixError)

    @pytest.mark.parametrize(
        "cls",
        [IncompatibleAxisError, ModeAxisError, InsufficientSampleError],
    )
    def test_three_concrete_subclasses(self, cls: type[ConfigError]) -> None:
        assert issubclass(cls, ConfigError)

    def test_suggested_fix_carries_through(self) -> None:
        cfg = AnalysisConfig.common_continuous()
        err = ModeAxisError("dummy", suggested_fix=cfg)
        assert err.suggested_fix is cfg

    def test_suggested_fix_defaults_to_none(self) -> None:
        err = InsufficientSampleError(
            "T below floor", actual_T=10, required_T=20,
        )
        assert err.suggested_fix is None
        assert err.actual_T == 10 and err.required_T == 20

    def test_message_passthrough(self) -> None:
        err = IncompatibleAxisError("explanation here")
        assert "explanation here" in str(err)


# ---------------------------------------------------------------------------
# AnalysisConfig factories (§4.2)
# ---------------------------------------------------------------------------


class TestFactories:
    def test_individual_continuous_default_metric_is_ic(self) -> None:
        cfg = AnalysisConfig.individual_continuous()
        assert cfg.scope is FactorScope.INDIVIDUAL
        assert cfg.signal is Signal.CONTINUOUS
        assert cfg.metric is Metric.IC

    def test_individual_continuous_metric_fm(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.FM)
        assert cfg.metric is Metric.FM

    def test_individual_sparse_metric_none(self) -> None:
        cfg = AnalysisConfig.individual_sparse()
        assert cfg.metric is None

    def test_common_continuous_metric_none(self) -> None:
        cfg = AnalysisConfig.common_continuous()
        assert cfg.scope is FactorScope.COMMON
        assert cfg.signal is Signal.CONTINUOUS
        assert cfg.metric is None

    def test_common_sparse_metric_none(self) -> None:
        cfg = AnalysisConfig.common_sparse()
        assert cfg.scope is FactorScope.COMMON
        assert cfg.signal is Signal.SPARSE
        assert cfg.metric is None

    def test_forward_periods_pass_through(self) -> None:
        cfg = AnalysisConfig.individual_continuous(forward_periods=20)
        assert cfg.forward_periods == 20

    def test_factories_emit_five_distinct_legal_tuples(self) -> None:
        # Invariant: factories produce exactly the five legal cells.
        configs = [
            AnalysisConfig.individual_continuous(),
            AnalysisConfig.individual_continuous(metric=Metric.FM),
            AnalysisConfig.individual_sparse(),
            AnalysisConfig.common_continuous(),
            AnalysisConfig.common_sparse(),
        ]
        triples = {(c.scope, c.signal, c.metric) for c in configs}
        assert len(triples) == 5

    def test_frozen_dataclass(self) -> None:
        cfg = AnalysisConfig.individual_continuous()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.forward_periods = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Axis validation (§4.5 single-entry)
# ---------------------------------------------------------------------------


class TestAxisValidation:
    def test_legal_direct_construction(self) -> None:
        cfg = AnalysisConfig(
            scope=FactorScope.INDIVIDUAL,
            signal=Signal.CONTINUOUS,
            metric=Metric.IC,
        )
        assert cfg.metric is Metric.IC

    @pytest.mark.parametrize(
        "scope,signal,metric",
        [
            # SPARSE with metric set — covered by IncompatibleAxis.
            (FactorScope.INDIVIDUAL, Signal.SPARSE, Metric.IC),
            (FactorScope.INDIVIDUAL, Signal.SPARSE, Metric.FM),
            (FactorScope.COMMON, Signal.SPARSE, Metric.IC),
            # (COMMON, CONTINUOUS) with metric set — deferred per v0.5.
            (FactorScope.COMMON, Signal.CONTINUOUS, Metric.IC),
            (FactorScope.COMMON, Signal.CONTINUOUS, Metric.FM),
            # (INDIVIDUAL, CONTINUOUS) without metric — illegal.
            (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, None),
        ],
    )
    def test_illegal_triples_raise(
        self,
        scope: FactorScope,
        signal: Signal,
        metric: Metric | None,
    ) -> None:
        with pytest.raises(IncompatibleAxisError):
            AnalysisConfig(scope=scope, signal=signal, metric=metric)


# ---------------------------------------------------------------------------
# Round-trip + invalid from_dict (§4.6)
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param(
            lambda: AnalysisConfig.individual_continuous(),
            id="individual_continuous_ic",
        ),
        pytest.param(
            lambda: AnalysisConfig.individual_continuous(metric=Metric.FM),
            id="individual_continuous_fm",
        ),
        pytest.param(
            lambda: AnalysisConfig.individual_sparse(),
            id="individual_sparse",
        ),
        pytest.param(
            lambda: AnalysisConfig.common_continuous(),
            id="common_continuous",
        ),
        pytest.param(
            lambda: AnalysisConfig.common_sparse(),
            id="common_sparse",
        ),
    ],
)
def legal_config(request: pytest.FixtureRequest) -> AnalysisConfig:
    return request.param()


class TestRoundTrip:
    def test_to_dict_then_from_dict(
        self, legal_config: AnalysisConfig,
    ) -> None:
        assert AnalysisConfig.from_dict(legal_config.to_dict()) == legal_config

    def test_to_dict_serialises_metric_none(self) -> None:
        d = AnalysisConfig.individual_sparse().to_dict()
        assert d["metric"] is None

    def test_to_dict_uses_string_values(self) -> None:
        d = AnalysisConfig.individual_continuous().to_dict()
        assert d["scope"] == "individual"
        assert d["signal"] == "continuous"
        assert d["metric"] == "ic"

    def test_invalid_tuple_via_from_dict_raises(self) -> None:
        # Validates that __post_init__ is the single source of truth —
        # from_dict cannot bypass axis validation.
        with pytest.raises(IncompatibleAxisError):
            AnalysisConfig.from_dict({
                "scope": "individual",
                "signal": "sparse",
                "metric": "ic",
                "forward_periods": 5,
            })

    def test_from_dict_unknown_enum_value_raises(self) -> None:
        # StrEnum constructor raises ValueError on unknown member.
        with pytest.raises(ValueError):
            AnalysisConfig.from_dict({
                "scope": "not_a_scope",
                "signal": "continuous",
                "metric": "ic",
            })


# ---------------------------------------------------------------------------
# Fallback map (§4.5 A4)
# ---------------------------------------------------------------------------


class TestFallbackMap:
    def test_individual_continuous_n1_fallback(self) -> None:
        suggested = _FALLBACK_MAP[
            (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Mode.TIMESERIES)
        ]()
        assert suggested == AnalysisConfig.common_continuous()

    def test_fallback_values_are_legal_configs(self) -> None:
        # Each lazy factory must yield a valid AnalysisConfig (i.e. its
        # own __post_init__ accepts it).
        for factory in _FALLBACK_MAP.values():
            cfg = factory()
            assert isinstance(cfg, AnalysisConfig)
