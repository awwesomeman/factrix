"""v0.5 ``multi_factor.bhy`` — family partitioning + step-up FDR."""

from __future__ import annotations

import pytest

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import StatCode
from factrix._multi_factor import _family_key, bhy
from factrix._profile import FactorProfile
from factrix._registry import _SCOPE_COLLAPSED, _DispatchKey


def _profile(
    *,
    config: AnalysisConfig,
    mode: Mode,
    primary_p: float,
    stats: dict[StatCode, float] | None = None,
) -> FactorProfile:
    """Build a ``FactorProfile`` for BHY testing without running compute."""
    base_stats: dict[StatCode, float] = {StatCode.IC_P: primary_p}
    if stats:
        base_stats.update(stats)
    return FactorProfile(
        config=config,
        mode=mode,
        primary_p=primary_p,
        n_obs=100,
        stats=base_stats,
    )


# ---------------------------------------------------------------------------
# Family-key derivation (§5.6)
# ---------------------------------------------------------------------------


class TestFamilyKey:
    def test_panel_ic_family_key(self) -> None:
        prof = _profile(
            config=AnalysisConfig.individual_continuous(metric=Metric.IC),
            mode=Mode.PANEL,
            primary_p=0.01,
        )
        assert _family_key(prof) == _DispatchKey(
            FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC, Mode.PANEL,
        )

    def test_panel_fm_distinct_from_panel_ic(self) -> None:
        prof_ic = _profile(
            config=AnalysisConfig.individual_continuous(metric=Metric.IC),
            mode=Mode.PANEL,
            primary_p=0.01,
        )
        prof_fm = _profile(
            config=AnalysisConfig.individual_continuous(metric=Metric.FM),
            mode=Mode.PANEL,
            primary_p=0.02,
        )
        assert _family_key(prof_ic) != _family_key(prof_fm)

    def test_sparse_n1_individual_and_common_share_family(self) -> None:
        prof_i = _profile(
            config=AnalysisConfig.individual_sparse(),
            mode=Mode.TIMESERIES,
            primary_p=0.03,
        )
        prof_c = _profile(
            config=AnalysisConfig.common_sparse(),
            mode=Mode.TIMESERIES,
            primary_p=0.04,
        )
        assert _family_key(prof_i) == _family_key(prof_c)
        assert _family_key(prof_i).scope is _SCOPE_COLLAPSED

    def test_panel_sparse_does_not_collapse(self) -> None:
        prof = _profile(
            config=AnalysisConfig.individual_sparse(),
            mode=Mode.PANEL,
            primary_p=0.05,
        )
        assert _family_key(prof).scope is FactorScope.INDIVIDUAL

    def test_mode_a_and_mode_b_not_same_family(self) -> None:
        prof_a = _profile(
            config=AnalysisConfig.common_continuous(),
            mode=Mode.PANEL,
            primary_p=0.01,
        )
        prof_b = _profile(
            config=AnalysisConfig.common_continuous(),
            mode=Mode.TIMESERIES,
            primary_p=0.01,
        )
        assert _family_key(prof_a) != _family_key(prof_b)


# ---------------------------------------------------------------------------
# bhy partitioning + step-up correctness
# ---------------------------------------------------------------------------


class TestBhyEmpty:
    def test_empty_input_returns_empty(self) -> None:
        assert bhy([]) == []


class TestBhyStepUp:
    def test_low_p_values_pass_at_default_threshold(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        profiles = [
            _profile(config=cfg, mode=Mode.PANEL, primary_p=p)
            for p in [0.001, 0.002, 0.003]
        ]
        survivors = bhy(profiles)
        assert len(survivors) == 3

    def test_high_p_values_fail(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        profiles = [
            _profile(config=cfg, mode=Mode.PANEL, primary_p=p)
            for p in [0.5, 0.7, 0.9]
        ]
        survivors = bhy(profiles)
        assert survivors == []


class TestBhyFamilyIsolation:
    def test_families_evaluated_independently(self) -> None:
        # Tiny p in family A; only large p in family B. A passes, B fails.
        cfg_a = AnalysisConfig.individual_continuous(metric=Metric.IC)
        cfg_b = AnalysisConfig.individual_continuous(metric=Metric.FM)
        profiles = [
            _profile(config=cfg_a, mode=Mode.PANEL, primary_p=0.001),
            _profile(config=cfg_a, mode=Mode.PANEL, primary_p=0.002),
            _profile(config=cfg_b, mode=Mode.PANEL, primary_p=0.50),
            _profile(config=cfg_b, mode=Mode.PANEL, primary_p=0.80),
        ]
        survivors = bhy(profiles)
        # Only family A's two profiles survive.
        assert len(survivors) == 2
        for s in survivors:
            assert s.config.metric is Metric.IC

    def test_sparse_n1_individual_and_common_pool_into_one_family(self) -> None:
        # Three profiles sharing the sentinel family; their p-values
        # are jointly evaluated even though two come from
        # individual_sparse() and one from common_sparse().
        profiles = [
            _profile(
                config=AnalysisConfig.individual_sparse(),
                mode=Mode.TIMESERIES,
                primary_p=0.001,
            ),
            _profile(
                config=AnalysisConfig.individual_sparse(),
                mode=Mode.TIMESERIES,
                primary_p=0.002,
            ),
            _profile(
                config=AnalysisConfig.common_sparse(),
                mode=Mode.TIMESERIES,
                primary_p=0.003,
            ),
        ]
        keys = {_family_key(p) for p in profiles}
        assert len(keys) == 1
        survivors = bhy(profiles)
        assert len(survivors) == 3


# ---------------------------------------------------------------------------
# threshold + gate
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_threshold_loosens_decision(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        # Single test, p=0.04: passes at threshold=0.05, fails at 0.01.
        prof = _profile(config=cfg, mode=Mode.PANEL, primary_p=0.04)
        assert bhy([prof], threshold=0.05) == [prof]
        assert bhy([prof], threshold=0.01) == []


class TestGateOverride:
    def test_gate_reads_from_stats_not_primary_p(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        # primary_p=0.99 (would FAIL); stats[FM_LAMBDA_P]=0.001 (would PASS)
        prof = _profile(
            config=cfg,
            mode=Mode.PANEL,
            primary_p=0.99,
            stats={StatCode.FM_LAMBDA_P: 0.001},
        )
        assert bhy([prof], gate=StatCode.FM_LAMBDA_P) == [prof]

    def test_missing_gate_raises_keyerror(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        prof = _profile(config=cfg, mode=Mode.PANEL, primary_p=0.001)
        # CAAR_P never populated for an IC profile.
        with pytest.raises(KeyError):
            bhy([prof], gate=StatCode.CAAR_P)
