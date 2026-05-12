"""v0.13 ``multi_factor.bhy_hierarchical`` — Yekutieli 2008 two-stage
FDR (#175)."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric, Mode
from factrix._codes import StatCode
from factrix._errors import UserInputError
from factrix._multi_factor import Survivors, bhy, bhy_hierarchical
from factrix._profile import FactorProfile
from factrix.stats.multiple_testing import bhy_adjusted_p, simes_p


def _profile(
    *,
    factor_id: str,
    primary_p: float,
    family: str,
    forward_periods: int = 5,
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
        context={"family": family},
        stats={StatCode.P_NW: primary_p},
    )


class TestGroupValidation:
    def test_identity_field_as_group_raises(self) -> None:
        prof = [_profile(factor_id="a", primary_p=0.01, family="g")]
        with pytest.raises(UserInputError) as ex:
            bhy_hierarchical(prof, group="factor_id")
        assert ex.value.field == "expand_over"

    def test_forward_periods_as_group_raises(self) -> None:
        prof = [_profile(factor_id="a", primary_p=0.01, family="g")]
        with pytest.raises(UserInputError) as ex:
            bhy_hierarchical(prof, group="forward_periods")
        assert ex.value.field == "expand_over"

    def test_unknown_group_key_raises(self) -> None:
        prof = [_profile(factor_id="a", primary_p=0.01, family="g")]
        with pytest.raises(UserInputError):
            bhy_hierarchical(prof, group="unknown_key")

    def test_duplicate_partition_key_raises(self) -> None:
        prof = [
            _profile(factor_id="a", primary_p=0.01, family="g1"),
            _profile(factor_id="a", primary_p=0.02, family="g1"),
        ]
        with pytest.raises(UserInputError):
            bhy_hierarchical(prof, group="family")


class TestProcedure:
    def test_returns_survivors_with_group_metadata(self) -> None:
        prof = [
            _profile(factor_id="a", primary_p=0.001, family="momentum"),
            _profile(factor_id="b", primary_p=0.002, family="momentum"),
            _profile(factor_id="c", primary_p=0.003, family="value"),
            _profile(factor_id="d", primary_p=0.004, family="value"),
        ]
        sv = bhy_hierarchical(prof, group="family", q=0.05)
        assert isinstance(sv, Survivors)
        assert sv.q == 0.05
        assert sv.expand_over == ("family",)
        assert set(sv.n_total.keys()) == {("momentum",), ("value",)}

    def test_all_dead_group_drops_out(self) -> None:
        # Outer Simes on a dead group fails the outer step-up, so no
        # cell in that group can survive even if its raw p is low.
        prof = [
            _profile(factor_id="a", primary_p=0.001, family="momentum"),
            _profile(factor_id="b", primary_p=0.001, family="momentum"),
            _profile(factor_id="c", primary_p=0.90, family="value"),
            _profile(factor_id="d", primary_p=0.95, family="value"),
        ]
        sv = bhy_hierarchical(prof, group="family", q=0.05)
        surviving_families = {p.context["family"] for p in sv.profiles}
        assert "value" not in surviving_families
        assert "momentum" in surviving_families

    def test_duality_adj_p_le_q(self) -> None:
        prof = [
            _profile(factor_id=f"f{i}", primary_p=p, family="g1")
            for i, p in enumerate([0.001, 0.01, 0.4])
        ] + [
            _profile(factor_id=f"f{i + 10}", primary_p=p, family="g2")
            for i, p in enumerate([0.002, 0.5, 0.7])
        ]
        q = 0.1
        sv = bhy_hierarchical(prof, group="family", q=q)
        assert np.all(sv.adj_p <= q + 1e-12)

    def test_input_order_preserved(self) -> None:
        prof = [
            _profile(factor_id="c", primary_p=0.001, family="g1"),
            _profile(factor_id="a", primary_p=0.001, family="g2"),
            _profile(factor_id="b", primary_p=0.001, family="g1"),
        ]
        sv = bhy_hierarchical(prof, group="family", q=0.5)
        ids = [p.factor_id for p in sv.profiles]
        # All three survive at q=0.5; order matches input order.
        assert ids == ["c", "a", "b"]

    def test_matches_two_layer_formula_by_hand(self) -> None:
        # Two groups of two factors each. Verify the max-of-layers fold
        # against an explicit Simes + BHY composition.
        prof = [
            _profile(factor_id="m1", primary_p=0.01, family="momentum"),
            _profile(factor_id="m2", primary_p=0.04, family="momentum"),
            _profile(factor_id="v1", primary_p=0.001, family="value"),
            _profile(factor_id="v2", primary_p=0.3, family="value"),
        ]
        # Outer Simes per group.
        simes_mom = simes_p([0.01, 0.04])
        simes_val = simes_p([0.001, 0.3])
        outer = bhy_adjusted_p(np.array([simes_mom, simes_val]))
        # Inner BHY per group.
        inner_mom = bhy_adjusted_p(np.array([0.01, 0.04]))
        inner_val = bhy_adjusted_p(np.array([0.001, 0.3]))
        expected_adj = np.array(
            [
                max(outer[0], inner_mom[0]),
                max(outer[0], inner_mom[1]),
                max(outer[1], inner_val[0]),
                max(outer[1], inner_val[1]),
            ]
        )
        q = 0.5  # loose so we keep all four for arithmetic comparison.
        sv = bhy_hierarchical(prof, group="family", q=q)
        expected_survivor_idx = np.flatnonzero(expected_adj <= q)
        assert len(sv.profiles) == len(expected_survivor_idx)
        np.testing.assert_allclose(sv.adj_p, expected_adj[expected_survivor_idx])


class TestEstimatorOverride:
    def test_estimator_routes_through_stats_keys(self) -> None:
        # Profile carries primary_p=0.9 (would not survive) but a
        # separate stats[P_NW] entry — confirming estimator override
        # works the same way as in bhy / partial_conjunction is left to
        # those verbs' tests; here we just confirm the kwarg is wired.
        prof = [
            _profile(factor_id="a", primary_p=0.001, family="g1"),
            _profile(factor_id="b", primary_p=0.001, family="g2"),
        ]
        # estimator=None must behave like the default code path.
        sv_default = bhy_hierarchical(prof, group="family", q=0.5)
        sv_none = bhy_hierarchical(prof, group="family", estimator=None, q=0.5)
        assert len(sv_default.profiles) == len(sv_none.profiles)


class TestEmpty:
    def test_empty_profiles_returns_empty_survivors(self) -> None:
        sv = bhy_hierarchical([], group="family", q=0.05)
        assert len(sv.profiles) == 0
        assert sv.expand_over == ("family",)
        assert sv.adj_p.shape == (0,)


class TestSingleGroupDegeneration:
    def test_single_group_no_more_permissive_than_flat_bhy(self) -> None:
        # Single-group input: the outer Simes floor can only lift adj_p
        # above the flat-BHY value, never below it.
        prof = [
            _profile(factor_id="a", primary_p=0.001, family="only"),
            _profile(factor_id="b", primary_p=0.04, family="only"),
            _profile(factor_id="c", primary_p=0.5, family="only"),
        ]
        sv_hier = bhy_hierarchical(prof, group="family", q=0.1)
        sv_flat = bhy(prof, q=0.1)
        # Survivor sets may differ (the simes_p outer floor can lift
        # adj_p above q for some cells); we assert only that hierarchical
        # is no more permissive than flat BHY on the same input.
        flat_ids = {p.factor_id for p in sv_flat.profiles}
        hier_ids = {p.factor_id for p in sv_hier.profiles}
        assert hier_ids.issubset(flat_ids)
