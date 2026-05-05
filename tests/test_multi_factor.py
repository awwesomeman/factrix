"""v0.5 ``multi_factor.bhy`` — family partitioning + step-up FDR."""

from __future__ import annotations

import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import StatCode
from factrix._multi_factor import _family_key, _FamilyKey, bhy
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
        n_assets=1 if mode is Mode.TIMESERIES else 30,
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
        assert _family_key(prof) == _FamilyKey(
            dispatch=_DispatchKey(
                FactorScope.INDIVIDUAL,
                Signal.CONTINUOUS,
                Metric.IC,
                Mode.PANEL,
            ),
            forward_periods=5,
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
        assert _family_key(prof_i).dispatch.scope is _SCOPE_COLLAPSED

    def test_panel_sparse_does_not_collapse(self) -> None:
        prof = _profile(
            config=AnalysisConfig.individual_sparse(),
            mode=Mode.PANEL,
            primary_p=0.05,
        )
        assert _family_key(prof).dispatch.scope is FactorScope.INDIVIDUAL

    def test_distinct_horizons_get_distinct_families(self) -> None:
        # Same dispatch cell, different forward_periods → different
        # families. Pooling horizons would dilute the BHY step-up
        # threshold and silently inflate FDR.
        prof_h5 = _profile(
            config=AnalysisConfig.individual_continuous(
                metric=Metric.IC,
                forward_periods=5,
            ),
            mode=Mode.PANEL,
            primary_p=0.01,
        )
        prof_h20 = _profile(
            config=AnalysisConfig.individual_continuous(
                metric=Metric.IC,
                forward_periods=20,
            ),
            mode=Mode.PANEL,
            primary_p=0.01,
        )
        assert _family_key(prof_h5) != _family_key(prof_h20)
        assert _family_key(prof_h5).dispatch == _family_key(prof_h20).dispatch
        assert _family_key(prof_h5).forward_periods == 5
        assert _family_key(prof_h20).forward_periods == 20

    def test_panel_and_timeseries_not_same_family(self) -> None:
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
            _profile(config=cfg, mode=Mode.PANEL, primary_p=p) for p in [0.5, 0.7, 0.9]
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

    def test_horizons_isolated_within_same_cell(self) -> None:
        # Two horizons of the same cell. If pooled, h=20's large p's
        # would borrow h=5's strong evidence and inflate the survivor
        # count beyond the per-horizon truth.
        cfg_h5 = AnalysisConfig.individual_continuous(
            metric=Metric.IC,
            forward_periods=5,
        )
        cfg_h20 = AnalysisConfig.individual_continuous(
            metric=Metric.IC,
            forward_periods=20,
        )
        profiles = [
            _profile(config=cfg_h5, mode=Mode.PANEL, primary_p=0.001),
            _profile(config=cfg_h5, mode=Mode.PANEL, primary_p=0.002),
            _profile(config=cfg_h20, mode=Mode.PANEL, primary_p=0.60),
            _profile(config=cfg_h20, mode=Mode.PANEL, primary_p=0.80),
        ]
        survivors = bhy(profiles)
        assert len(survivors) == 2
        for s in survivors:
            assert s.config.forward_periods == 5

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


# ---------------------------------------------------------------------------
# UX-2 review fix: cross-family no-op warning
# ---------------------------------------------------------------------------


class TestSingletonFamilyWarning:
    def test_warns_on_multiple_singleton_families(self) -> None:
        """README's BHY-of-the-day bug: 3 distinct cells × 1 profile each
        produces 3 size-1 families — BHY ≡ raw threshold, no FDR control."""
        ic = _profile(
            config=AnalysisConfig.individual_continuous(metric=Metric.IC),
            mode=Mode.PANEL,
            primary_p=0.04,
        )
        fm = _profile(
            config=AnalysisConfig.individual_continuous(metric=Metric.FM),
            mode=Mode.PANEL,
            primary_p=0.04,
        )
        common = _profile(
            config=AnalysisConfig.common_continuous(),
            mode=Mode.PANEL,
            primary_p=0.04,
        )
        with pytest.warns(RuntimeWarning, match="single profile"):
            bhy([ic, fm, common], threshold=0.05)

    def test_silent_on_single_family(self) -> None:
        """One family with one profile is the legitimate single-candidate
        case — the cross-family no-op heuristic does not fire."""
        prof = _profile(
            config=AnalysisConfig.individual_continuous(metric=Metric.IC),
            mode=Mode.PANEL,
            primary_p=0.04,
        )
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            bhy([prof], threshold=0.05)

    def test_silent_when_all_families_have_two_plus(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        a = _profile(config=cfg, mode=Mode.PANEL, primary_p=0.01)
        b = _profile(config=cfg, mode=Mode.PANEL, primary_p=0.02)
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            bhy([a, b], threshold=0.05)


# ---------------------------------------------------------------------------
# bhy gate validation: must be a p-value StatCode
# ---------------------------------------------------------------------------


class TestBhyGateValidation:
    def test_non_p_gate_rejected(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        prof = _profile(
            config=cfg,
            mode=Mode.PANEL,
            primary_p=0.04,
            stats={StatCode.IC_T_NW: 2.5},
        )
        # IC_T_NW is a t-stat, not a probability — BHY step-up math
        # would silently corrupt if fed t-stats.
        with pytest.raises(ValueError, match="requires p-value input"):
            bhy([prof], threshold=0.05, gate=StatCode.IC_T_NW)

    def test_p_gate_accepted(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        prof = _profile(
            config=cfg,
            mode=Mode.PANEL,
            primary_p=0.99,
            stats={StatCode.FM_LAMBDA_P: 0.001},
        )
        survivors = bhy(
            [prof],
            threshold=0.05,
            gate=StatCode.FM_LAMBDA_P,
        )
        assert survivors == [prof]

    def test_default_gate_uses_primary_p(self) -> None:
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        prof = _profile(config=cfg, mode=Mode.PANEL, primary_p=0.01)
        # No exception even though stats has no *_P; primary_p is used.
        assert bhy([prof], threshold=0.05) == [prof]


class TestStatCodeIsPValue:
    def test_p_codes_marked(self) -> None:
        for code in (
            StatCode.IC_P,
            StatCode.FM_LAMBDA_P,
            StatCode.TS_BETA_P,
            StatCode.CAAR_P,
            StatCode.FACTOR_ADF_P,
            StatCode.LJUNG_BOX_P,
        ):
            assert code.is_p_value, f"{code.name} should be flagged as p-value"

    def test_non_p_codes_unmarked(self) -> None:
        for code in (
            StatCode.IC_MEAN,
            StatCode.IC_T_NW,
            StatCode.FM_LAMBDA_MEAN,
            StatCode.FM_LAMBDA_T_NW,
            StatCode.TS_BETA,
            StatCode.TS_BETA_T_NW,
            StatCode.CAAR_MEAN,
            StatCode.CAAR_T_NW,
            StatCode.EVENT_TEMPORAL_HHI,
            StatCode.NW_LAGS_USED,
        ):
            assert not code.is_p_value, f"{code.name} should NOT be flagged as p-value"
