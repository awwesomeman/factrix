"""v0.5 registry SSOT + procedure stubs + FactorProfile semantics.

Covers refactor_api.md §4.4 (registry as SSOT, A1), §4.4.2 (Profile
shape + verdict policy), §5.4.1 (Mode B sparse collapse), §7.5
(naming-consistency invariants).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import InfoCode, StatCode, Verdict, WarningCode
from factrix._errors import IncompatibleAxisError
from factrix._procedures import FactorProcedure, InputSchema
from factrix._profile import FactorProfile
from factrix._registry import (
    _DISPATCH_REGISTRY,
    _SCOPE_COLLAPSED,
    _DispatchKey,
    _RegistryEntry,
    _ScopeCollapsedSentinel,
    matches_user_axis,
    register,
)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


class TestRegistryShape:
    def test_seven_entries_registered(self) -> None:
        assert len(_DISPATCH_REGISTRY) == 7

    def test_panel_entries_present(self) -> None:
        for key in (
            _DispatchKey(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC, Mode.PANEL),
            _DispatchKey(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.FM, Mode.PANEL),
            _DispatchKey(FactorScope.INDIVIDUAL, Signal.SPARSE, None, Mode.PANEL),
            _DispatchKey(FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.PANEL),
            _DispatchKey(FactorScope.COMMON, Signal.SPARSE, None, Mode.PANEL),
        ):
            assert key in _DISPATCH_REGISTRY

    def test_timeseries_entries_present(self) -> None:
        # COMMON × CONTINUOUS legal at N=1; sparse cell uses sentinel.
        assert _DispatchKey(
            FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.TIMESERIES,
        ) in _DISPATCH_REGISTRY
        assert _DispatchKey(
            _SCOPE_COLLAPSED, Signal.SPARSE, None, Mode.TIMESERIES,
        ) in _DISPATCH_REGISTRY

    def test_individual_continuous_timeseries_absent(self) -> None:
        # §5.5 — undefined at N=1; raises ModeAxisError at evaluate time.
        for metric in (Metric.IC, Metric.FM):
            assert _DispatchKey(
                FactorScope.INDIVIDUAL, Signal.CONTINUOUS, metric, Mode.TIMESERIES,
            ) not in _DISPATCH_REGISTRY

    def test_every_entry_has_use_case_and_refs(self) -> None:
        for entry in _DISPATCH_REGISTRY.values():
            assert entry.canonical_use_case
            assert isinstance(entry.references, tuple)
            assert all(isinstance(r, str) for r in entry.references)


# ---------------------------------------------------------------------------
# Sentinel + register()
# ---------------------------------------------------------------------------


class TestSentinel:
    def test_sentinel_repr(self) -> None:
        assert repr(_SCOPE_COLLAPSED) == "_SCOPE_COLLAPSED"

    def test_sentinel_singleton_class(self) -> None:
        assert isinstance(_SCOPE_COLLAPSED, _ScopeCollapsedSentinel)

    def test_sentinel_distinct_from_factor_scope(self) -> None:
        assert _SCOPE_COLLAPSED is not FactorScope.INDIVIDUAL
        assert _SCOPE_COLLAPSED is not FactorScope.COMMON


class _NoopProcedure:
    """Throwaway procedure used by the duplicate-key test."""

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema()

    def compute(self, raw, config):  # pragma: no cover - never called
        raise RuntimeError("noop")


class TestRegisterIdempotency:
    def test_duplicate_key_rejected(self) -> None:
        existing_key = next(iter(_DISPATCH_REGISTRY))
        with pytest.raises(ValueError, match="already registered"):
            register(
                existing_key,
                _NoopProcedure(),
                use_case="dup",
            )


# ---------------------------------------------------------------------------
# Factory ↔ registry invariant
# ---------------------------------------------------------------------------


class TestFactoryRegistryInvariant:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: AnalysisConfig.individual_continuous(),
            lambda: AnalysisConfig.individual_continuous(metric=Metric.FM),
            lambda: AnalysisConfig.individual_sparse(),
            lambda: AnalysisConfig.common_continuous(),
            lambda: AnalysisConfig.common_sparse(),
        ],
    )
    def test_factory_output_matches_registry(self, factory) -> None:
        cfg = factory()
        assert matches_user_axis(cfg.scope, cfg.signal, cfg.metric)

    def test_validator_rejects_outside_registry(self) -> None:
        with pytest.raises(IncompatibleAxisError):
            AnalysisConfig(
                scope=FactorScope.INDIVIDUAL,
                signal=Signal.SPARSE,
                metric=Metric.IC,
            )


# ---------------------------------------------------------------------------
# FactorProcedure Protocol conformance
# ---------------------------------------------------------------------------


class TestProcedureProtocol:
    def test_every_registered_procedure_satisfies_protocol(self) -> None:
        for entry in _DISPATCH_REGISTRY.values():
            assert isinstance(entry.procedure, FactorProcedure)

    def test_input_schema_present(self) -> None:
        for entry in _DISPATCH_REGISTRY.values():
            assert isinstance(entry.procedure.INPUT_SCHEMA, InputSchema)


# ---------------------------------------------------------------------------
# FactorProfile.verdict (§4.4.2 / §7.5 naming invariants)
# ---------------------------------------------------------------------------


def _make_profile(
    *,
    primary_p: float,
    stats: dict[StatCode, float] | None = None,
    warnings: frozenset[WarningCode] = frozenset(),
    info_notes: frozenset[InfoCode] = frozenset(),
) -> FactorProfile:
    return FactorProfile(
        config=AnalysisConfig.individual_continuous(),
        mode=Mode.PANEL,
        primary_p=primary_p,
        n_obs=100,
        warnings=warnings,
        info_notes=info_notes,
        stats=stats or {},
    )


class TestVerdict:
    def test_pass_below_default_threshold(self) -> None:
        assert _make_profile(primary_p=0.01).verdict() is Verdict.PASS

    def test_fail_at_or_above_default_threshold(self) -> None:
        # Strict `<` per §4.4.2; equality fails.
        assert _make_profile(primary_p=0.05).verdict() is Verdict.FAIL
        assert _make_profile(primary_p=0.10).verdict() is Verdict.FAIL

    def test_threshold_override(self) -> None:
        prof = _make_profile(primary_p=0.07)
        assert prof.verdict(threshold=0.10) is Verdict.PASS
        assert prof.verdict(threshold=0.05) is Verdict.FAIL

    def test_gate_override_uses_stats_value(self) -> None:
        prof = _make_profile(
            primary_p=0.50,  # would FAIL on primary
            stats={StatCode.IC_P: 0.001},
        )
        assert prof.verdict(gate=StatCode.IC_P) is Verdict.PASS

    def test_gate_keyerror_when_stat_missing(self) -> None:
        prof = _make_profile(primary_p=0.01, stats={})
        with pytest.raises(KeyError):
            prof.verdict(gate=StatCode.FM_LAMBDA_P)


class TestProfileImmutability:
    def test_frozen(self) -> None:
        import dataclasses

        prof = _make_profile(primary_p=0.01)
        with pytest.raises(dataclasses.FrozenInstanceError):
            prof.primary_p = 0.99  # type: ignore[misc]

    def test_default_collections_empty(self) -> None:
        prof = FactorProfile(
            config=AnalysisConfig.common_continuous(),
            mode=Mode.TIMESERIES,
            primary_p=0.5,
            n_obs=50,
        )
        assert prof.warnings == frozenset()
        assert prof.info_notes == frozenset()
        assert dict(prof.stats) == {}


class TestDiagnose:
    def test_diagnose_serialises_enums_to_strings(self) -> None:
        prof = _make_profile(
            primary_p=0.02,
            stats={StatCode.IC_MEAN: 0.05, StatCode.NW_LAGS_USED: 4.0},
            warnings=frozenset({WarningCode.UNRELIABLE_SE_SHORT_SERIES}),
            info_notes=frozenset({InfoCode.SCOPE_AXIS_COLLAPSED}),
        )
        d = prof.diagnose()
        assert d["mode"] == "panel"
        assert d["n_obs"] == 100
        assert d["primary_p"] == 0.02
        assert "unreliable_se_short_series" in d["warnings"]
        assert "scope_axis_collapsed" in d["info_notes"]
        assert d["stats"]["ic_mean"] == 0.05


# ---------------------------------------------------------------------------
# matches_user_axis edge cases
# ---------------------------------------------------------------------------


class TestMatchesUserAxis:
    @pytest.mark.parametrize(
        "scope,signal,metric",
        [
            (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC),
            (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.FM),
            (FactorScope.INDIVIDUAL, Signal.SPARSE, None),
            (FactorScope.COMMON, Signal.CONTINUOUS, None),
            (FactorScope.COMMON, Signal.SPARSE, None),
        ],
    )
    def test_legal_triples_match(
        self,
        scope: FactorScope,
        signal: Signal,
        metric: Metric | None,
    ) -> None:
        assert matches_user_axis(scope, signal, metric)

    @pytest.mark.parametrize(
        "scope,signal,metric",
        [
            (FactorScope.INDIVIDUAL, Signal.SPARSE, Metric.IC),
            (FactorScope.COMMON, Signal.CONTINUOUS, Metric.FM),
            (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, None),
        ],
    )
    def test_illegal_triples_do_not_match(
        self,
        scope: FactorScope,
        signal: Signal,
        metric: Metric | None,
    ) -> None:
        assert not matches_user_axis(scope, signal, metric)

    def test_individual_sparse_matches_via_panel_entry(self) -> None:
        # Direct PANEL entry exists; sentinel TIMESERIES entry would
        # also accept INDIVIDUAL via the collapse rule.
        assert matches_user_axis(
            FactorScope.INDIVIDUAL, Signal.SPARSE, None,
        )
