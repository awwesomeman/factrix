"""v0.13 ``multi_factor.partial_conjunction`` — contract-bearing path
for "factor X significant in k of m conditions" (#162)."""

from __future__ import annotations

import warnings

import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric, Mode
from factrix._codes import StatCode
from factrix._errors import UserInputError
from factrix._multi_factor import Survivors, partial_conjunction
from factrix._profile import FactorProfile


def _profile(
    *,
    factor_id: str,
    forward_periods: int = 5,
    primary_p: float,
    context: dict[str, object] | None = None,
) -> FactorProfile:
    cfg = AnalysisConfig.individual_continuous(
        metric=Metric.IC, forward_periods=forward_periods
    )
    return FactorProfile(
        config=cfg,
        mode=Mode.PANEL,
        primary_p=primary_p,
        primary_stat=2.0,
        primary_stat_name=StatCode.T_NW,
        n_obs=100,
        n_pairs=3000,
        n_periods=100,
        n_assets=30,
        factor_id=factor_id,
        context=context or {},
        stats={StatCode.P_NW: primary_p},
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestMinPassValidation:
    def test_min_pass_one_raises_with_pointer_to_bhy(self) -> None:
        prof = [
            _profile(factor_id="f", primary_p=0.01, context={"u": "a"}),
            _profile(factor_id="f", primary_p=0.02, context={"u": "b"}),
        ]
        with pytest.raises(UserInputError) as ex:
            partial_conjunction(prof, min_pass=1, expand_over=["u"])
        assert ex.value.field == "min_pass"
        message = str(ex.value)
        assert "bhy" in message and "expand_over" in message
        assert "union" in message

    def test_min_pass_zero_raises(self) -> None:
        prof = [_profile(factor_id="f", primary_p=0.01, context={"u": "a"})]
        with pytest.raises(UserInputError):
            partial_conjunction(prof, min_pass=0, expand_over=["u"])

    def test_min_pass_negative_raises(self) -> None:
        prof = [_profile(factor_id="f", primary_p=0.01, context={"u": "a"})]
        with pytest.raises(UserInputError):
            partial_conjunction(prof, min_pass=-1, expand_over=["u"])


class TestExpandOverValidation:
    def test_empty_expand_over_raises(self) -> None:
        prof = [_profile(factor_id="f", primary_p=0.01, context={"u": "a"})]
        with pytest.raises(UserInputError) as ex:
            partial_conjunction(prof, min_pass=2, expand_over=[])
        assert "expand_over" in str(ex.value)

    def test_unknown_expand_over_key_raises(self) -> None:
        prof = [_profile(factor_id="f", primary_p=0.01, context={"u": "a"})]
        with pytest.raises(UserInputError):
            partial_conjunction(prof, min_pass=2, expand_over=["regime_id"])

    def test_identity_field_in_expand_over_raises(self) -> None:
        prof = [_profile(factor_id="f", primary_p=0.01, context={"u": "a"})]
        with pytest.raises(UserInputError):
            partial_conjunction(prof, min_pass=2, expand_over=["factor_id"])


class TestNConditionsValidation:
    def test_n_conditions_less_than_min_pass_raises(self) -> None:
        prof = [_profile(factor_id="f", primary_p=0.01, context={"u": "a"})]
        with pytest.raises(UserInputError) as ex:
            partial_conjunction(prof, min_pass=3, n_conditions=2, expand_over=["u"])
        assert ex.value.field == "n_conditions"

    def test_strict_mismatch_raises(self) -> None:
        prof = [
            _profile(factor_id="f", primary_p=0.01, context={"u": "a"}),
            _profile(factor_id="f", primary_p=0.02, context={"u": "b"}),
            _profile(factor_id="f", primary_p=0.03, context={"u": "c"}),
        ]
        with pytest.raises(UserInputError) as ex:
            partial_conjunction(prof, min_pass=2, n_conditions=2, expand_over=["u"])
        assert ex.value.field == "n_conditions"

    def test_insufficient_conditions_raises(self) -> None:
        prof = [
            _profile(factor_id="f", primary_p=0.01, context={"u": "a"}),
            _profile(factor_id="f", primary_p=0.02, context={"u": "b"}),
        ]
        with pytest.raises(UserInputError):
            partial_conjunction(prof, min_pass=3, expand_over=["u"])


# ---------------------------------------------------------------------------
# PC p-value math (BH2008 Bonferroni-style: (m - k + 1) * p_((k)))
# ---------------------------------------------------------------------------


class TestPCPValueMath:
    def test_full_conjunction_equals_max(self) -> None:
        prof = [
            _profile(factor_id="f", primary_p=0.01, context={"u": "a"}),
            _profile(factor_id="f", primary_p=0.03, context={"u": "b"}),
        ]
        sv = partial_conjunction(prof, min_pass=2, n_conditions=2, expand_over=["u"])
        assert sv.pc_p is not None
        assert sv.pc_p[0] == pytest.approx(0.03)

    def test_partial_conjunction_bonferroni_scaled(self) -> None:
        ps = [0.001, 0.005, 0.04, 0.5]
        prof = [
            _profile(factor_id="f", primary_p=p, context={"u": f"u{i}"})
            for i, p in enumerate(ps)
        ]
        sv = partial_conjunction(
            prof, min_pass=2, n_conditions=4, expand_over=["u"], q=0.5
        )
        assert sv.pc_p is not None
        assert sv.pc_p[0] == pytest.approx(3 * 0.005)

    def test_pc_p_capped_at_one(self) -> None:
        prof = [
            _profile(factor_id="f", primary_p=0.4, context={"u": "a"}),
            _profile(factor_id="f", primary_p=0.5, context={"u": "b"}),
            _profile(factor_id="f", primary_p=0.6, context={"u": "c"}),
        ]
        sv = partial_conjunction(
            prof, min_pass=2, n_conditions=3, expand_over=["u"], q=1.0
        )
        assert sv.pc_p is not None
        assert sv.pc_p[0] <= 1.0


# ---------------------------------------------------------------------------
# Survivor selection
# ---------------------------------------------------------------------------


class TestSurvivorSelection:
    def test_only_robust_factor_survives(self) -> None:
        # mom passes both universes; val only one
        prof = [
            _profile(factor_id="mom", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="mom", primary_p=0.002, context={"u": "b"}),
            _profile(factor_id="val", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="val", primary_p=0.30, context={"u": "b"}),
        ]
        sv = partial_conjunction(
            prof, min_pass=2, n_conditions=2, expand_over=["u"], q=0.05
        )
        assert [p.factor_id for p in sv.profiles] == ["mom"]

    def test_returns_survivors_with_pc_metadata(self) -> None:
        prof = [
            _profile(factor_id="f", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="f", primary_p=0.002, context={"u": "b"}),
        ]
        sv = partial_conjunction(
            prof, min_pass=2, n_conditions=2, expand_over=["u"], q=0.05
        )
        assert isinstance(sv, Survivors)
        assert sv.min_pass == 2
        assert sv.pc_p is not None
        assert sv.n_passed_uncorr is not None
        assert sv.expand_over == ("u",)
        assert sv.n_total[("f", 5)] == 2

    def test_n_passed_uncorr_counts_raw_below_q(self) -> None:
        prof = [
            _profile(factor_id="f", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="f", primary_p=0.04, context={"u": "b"}),
            _profile(factor_id="f", primary_p=0.20, context={"u": "c"}),
        ]
        sv = partial_conjunction(
            prof, min_pass=2, n_conditions=3, expand_over=["u"], q=0.5
        )
        assert sv.n_passed_uncorr is not None
        assert int(sv.n_passed_uncorr[0]) == 3


# ---------------------------------------------------------------------------
# Lenient (n_conditions=None) mode
# ---------------------------------------------------------------------------


class TestLenientMode:
    def test_lenient_infers_m_per_identity(self) -> None:
        prof = [
            _profile(factor_id="A", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="A", primary_p=0.002, context={"u": "b"}),
            _profile(factor_id="B", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="B", primary_p=0.002, context={"u": "b"}),
            _profile(factor_id="B", primary_p=0.003, context={"u": "c"}),
        ]
        with pytest.warns(RuntimeWarning, match="heterogeneous"):
            sv = partial_conjunction(prof, min_pass=2, expand_over=["u"], q=0.5)
        assert sv.n_total[("A", 5)] == 2
        assert sv.n_total[("B", 5)] == 3

    def test_lenient_homogeneous_m_no_warning(self) -> None:
        prof = [
            _profile(factor_id="A", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="A", primary_p=0.002, context={"u": "b"}),
            _profile(factor_id="B", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="B", primary_p=0.002, context={"u": "b"}),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            partial_conjunction(prof, min_pass=2, expand_over=["u"], q=0.5)


# ---------------------------------------------------------------------------
# Empty input edge case
# ---------------------------------------------------------------------------


class TestEmpty:
    def test_empty_profiles_returns_empty_survivors(self) -> None:
        sv = partial_conjunction([], min_pass=2, expand_over=["u"])
        assert sv.profiles == []
        assert len(sv.adj_p) == 0
        assert sv.min_pass == 2
        assert sv.pc_p is not None and len(sv.pc_p) == 0
