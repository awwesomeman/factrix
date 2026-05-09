"""v0.5 ``multi_factor.bhy`` — _resolve_family-backed step-up FDR (#161)."""

from __future__ import annotations

import warnings

import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric, Mode
from factrix._codes import StatCode
from factrix._errors import UserInputError
from factrix._multi_factor import bhy
from factrix._profile import FactorProfile


def _profile(
    *,
    factor_id: str = "factor",
    forward_periods: int = 5,
    primary_p: float,
    context: dict[str, object] | None = None,
    stats: dict[StatCode, float] | None = None,
    metric: Metric = Metric.IC,
    mode: Mode = Mode.PANEL,
) -> FactorProfile:
    cfg = AnalysisConfig.individual_continuous(
        metric=metric, forward_periods=forward_periods
    )
    base_stats: dict[StatCode, float] = {StatCode.IC_P: primary_p}
    if stats:
        base_stats.update(stats)
    return FactorProfile(
        config=cfg,
        mode=mode,
        primary_p=primary_p,
        n_obs=100,
        n_assets=30,
        factor_id=factor_id,
        context=context or {},
        stats=base_stats,
    )


# ---------------------------------------------------------------------------
# Step-up basics
# ---------------------------------------------------------------------------


class TestBhyEmpty:
    def test_empty_input_returns_empty(self) -> None:
        assert bhy([]) == []


class TestBhyStepUp:
    def test_low_p_values_pass_at_default_q(self) -> None:
        profiles = [
            _profile(factor_id=f"f{i}", primary_p=p)
            for i, p in enumerate([0.001, 0.002, 0.003])
        ]
        assert len(bhy(profiles)) == 3

    def test_high_p_values_fail(self) -> None:
        profiles = [
            _profile(factor_id=f"f{i}", primary_p=p)
            for i, p in enumerate([0.5, 0.7, 0.9])
        ]
        assert bhy(profiles) == []

    def test_q_loosens_decision(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        assert bhy([prof], q=0.05) == [prof]
        assert bhy([prof], q=0.01) == []


# ---------------------------------------------------------------------------
# expand_over: per-bucket independent step-up (BB2014)
# ---------------------------------------------------------------------------


class TestExpandOver:
    def test_buckets_evaluated_independently(self) -> None:
        # bucket "bull": all tiny p → both pass
        # bucket "bear": all large p → both fail
        profiles = [
            _profile(factor_id="f1", primary_p=0.001, context={"regime": "bull"}),
            _profile(factor_id="f2", primary_p=0.002, context={"regime": "bull"}),
            _profile(factor_id="f1", primary_p=0.5, context={"regime": "bear"}),
            _profile(factor_id="f2", primary_p=0.6, context={"regime": "bear"}),
        ]
        survivors = bhy(profiles, expand_over=["regime"])
        assert len(survivors) == 2
        assert {s.context["regime"] for s in survivors} == {"bull"}

    def test_no_expand_over_pools_all_into_one_family(self) -> None:
        # Distinct identities → all pass dedup; one step-up over 4 entries.
        profiles = [
            _profile(factor_id=f"f{i}", primary_p=p)
            for i, p in enumerate([0.001, 0.002, 0.5, 0.6])
        ]
        survivors = bhy(profiles)
        # All 4 in one family; only the small p's survive step-up.
        assert {s.factor_id for s in survivors} == {"f0", "f1"}

    def test_unknown_expand_over_raises(self) -> None:
        profiles = [_profile(factor_id="f1", primary_p=0.01, context={"regime": "x"})]
        with pytest.raises(UserInputError):
            bhy(profiles, expand_over=["regiem"])

    def test_singleton_buckets_warn(self) -> None:
        profiles = [
            _profile(factor_id="f1", primary_p=0.04, context={"regime": "a"}),
            _profile(factor_id="f1", primary_p=0.04, context={"regime": "b"}),
            _profile(factor_id="f1", primary_p=0.04, context={"regime": "c"}),
        ]
        with pytest.warns(RuntimeWarning, match="single profile"):
            bhy(profiles, expand_over=["regime"])


# ---------------------------------------------------------------------------
# p_stat alternate p-value
# ---------------------------------------------------------------------------


class TestPStat:
    def test_p_stat_reads_from_stats_not_primary_p(self) -> None:
        prof = _profile(
            factor_id="f1",
            primary_p=0.99,
            stats={StatCode.FM_LAMBDA_P: 0.001},
        )
        assert bhy([prof], p_stat=StatCode.FM_LAMBDA_P) == [prof]

    def test_non_p_stat_rejected(self) -> None:
        prof = _profile(
            factor_id="f1",
            primary_p=0.04,
            stats={StatCode.IC_T_NW: 2.5},
        )
        with pytest.raises(UserInputError, match="not a probability"):
            bhy([prof], p_stat=StatCode.IC_T_NW)

    def test_missing_p_stat_raises_user_error(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.001)
        with pytest.raises(UserInputError):
            bhy([prof], p_stat=StatCode.CAAR_P)


# ---------------------------------------------------------------------------
# Identity uniqueness (#160 anti-shopping defense, family layer)
# ---------------------------------------------------------------------------


class TestIdentityUniqueness:
    def test_duplicate_identity_raises(self) -> None:
        # Default factor_id="factor" on both → same identity → duplicate.
        # v0.4 auto-cell-split would have hidden this; v0.5 surfaces it.
        profiles = [
            _profile(primary_p=0.01, metric=Metric.IC),
            _profile(primary_p=0.02, metric=Metric.FM),
        ]
        with pytest.raises(UserInputError, match="duplicate"):
            bhy(profiles)


# ---------------------------------------------------------------------------
# Deprecated v0.4 kwargs (threshold= → q=, gate= → p_stat=)
# ---------------------------------------------------------------------------


class TestDeprecatedKwargs:
    def test_threshold_alias_warns(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        with pytest.warns(DeprecationWarning, match="threshold"):
            survivors = bhy([prof], threshold=0.05)
        assert survivors == [prof]

    def test_gate_alias_warns(self) -> None:
        prof = _profile(
            factor_id="f1",
            primary_p=0.99,
            stats={StatCode.FM_LAMBDA_P: 0.001},
        )
        with pytest.warns(DeprecationWarning, match="gate"):
            survivors = bhy([prof], gate=StatCode.FM_LAMBDA_P)
        assert survivors == [prof]

    def test_threshold_and_q_collide(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        with (
            pytest.raises(TypeError, match="not both"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            bhy([prof], threshold=0.05, q=0.01)

    def test_gate_and_p_stat_collide(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        with (
            pytest.raises(TypeError, match="not both"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            bhy([prof], gate=StatCode.IC_P, p_stat=StatCode.IC_P)

    def test_unknown_kwarg_raises(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        with pytest.raises(TypeError, match="unexpected keyword"):
            bhy([prof], wibble=1)


# ---------------------------------------------------------------------------
# Mixed-horizon migration foot-gun (v0.4 → v0.5)
# ---------------------------------------------------------------------------


class TestMixedHorizonWarning:
    def test_mixed_forward_periods_without_expand_over_warns(self) -> None:
        # v0.4 auto-isolated horizons; v0.5 caller must split. Warn loud
        # so a sweep that previously enjoyed implicit isolation does not
        # silently inflate FDR after upgrade.
        profiles = [
            _profile(factor_id="f1", forward_periods=5, primary_p=0.001),
            _profile(factor_id="f2", forward_periods=20, primary_p=0.001),
        ]
        with pytest.warns(RuntimeWarning, match="forward_periods"):
            bhy(profiles)

    def test_mixed_horizons_silent_when_expand_over_set(self) -> None:
        # Caller has explicitly declared per-bucket families; mixed-
        # horizon warning is suppressed even though horizons differ.
        profiles = [
            _profile(
                factor_id="f1",
                forward_periods=5,
                primary_p=0.001,
                context={"regime": "a"},
            ),
            _profile(
                factor_id="f2",
                forward_periods=20,
                primary_p=0.001,
                context={"regime": "a"},
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            bhy(profiles, expand_over=["regime"])

    def test_uniform_horizon_silent(self) -> None:
        profiles = [
            _profile(factor_id="f1", forward_periods=5, primary_p=0.001),
            _profile(factor_id="f2", forward_periods=5, primary_p=0.001),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            bhy(profiles)


# ---------------------------------------------------------------------------
# Duplicate-identity error message includes actionable hint
# ---------------------------------------------------------------------------


class TestDuplicateIdentityHint:
    def test_error_message_suggests_factor_id_or_expand_over(self) -> None:
        profiles = [
            _profile(factor_id="factor", primary_p=0.01),
            _profile(factor_id="factor", primary_p=0.02),
        ]
        with pytest.raises(UserInputError) as exc:
            bhy(profiles)
        msg = str(exc.value)
        assert "factor_id" in msg
        assert "expand_over" in msg
