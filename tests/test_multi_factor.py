"""v0.5 ``multi_factor.bhy`` — _resolve_family-backed step-up FDR (#161)."""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric, Mode
from factrix._codes import StatCode
from factrix._errors import UserInputError
from factrix._multi_factor import Survivors, bhy
from factrix._profile import FactorProfile
from factrix.stats import NeweyWest


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
    base_stats: dict[StatCode, float] = {StatCode.P: primary_p}
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
        result = bhy([])
        assert result.profiles == []
        assert len(result.adj_q) == 0


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
        assert bhy(profiles).profiles == []

    def test_q_loosens_decision(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        assert bhy([prof], q=0.05).profiles == [prof]
        assert bhy([prof], q=0.01).profiles == []


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
        result = bhy(profiles, expand_over=["regime"])
        assert len(result) == 2
        assert {p.context["regime"] for p in result.profiles} == {"bull"}

    def test_no_expand_over_pools_all_into_one_family(self) -> None:
        # Distinct identities → all pass dedup; one step-up over 4 entries.
        profiles = [
            _profile(factor_id=f"f{i}", primary_p=p)
            for i, p in enumerate([0.001, 0.002, 0.5, 0.6])
        ]
        result = bhy(profiles)
        # All 4 in one family; only the small p's survive step-up.
        assert {p.factor_id for p in result.profiles} == {"f0", "f1"}

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
# estimator override (#170)
# ---------------------------------------------------------------------------


class TestEstimatorOverride:
    def test_estimator_reads_dispatched_stat_not_primary_p(self) -> None:
        prof = _profile(
            factor_id="f1",
            metric=Metric.FM,
            primary_p=0.99,
            stats={StatCode.P: 0.001},
        )
        assert bhy([prof], estimator=NeweyWest()).profiles == [prof]

    def test_missing_dispatched_stat_raises_user_error(self) -> None:
        # NeweyWest dispatches to StatCode.P; a profile without P in
        # stats must surface the missing-key error.
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC, forward_periods=5)
        prof = FactorProfile(
            config=cfg,
            mode=Mode.PANEL,
            primary_p=0.001,
            n_obs=100,
            n_assets=30,
            factor_id="f1",
            stats={StatCode.MEAN: 0.05},
        )
        with pytest.raises(UserInputError) as exc:
            bhy([prof], estimator=NeweyWest())
        err = exc.value
        assert err.field == "estimator"
        assert err.value == "NeweyWest"
        assert "P" in (err.expected or "")

    def test_legacy_p_stat_kwarg_is_unrecognised(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        with pytest.raises(TypeError, match="p_stat"):
            bhy([prof], p_stat=StatCode.P)


# ---------------------------------------------------------------------------
# Identity uniqueness (#160 anti-shopping defense, family layer)
# ---------------------------------------------------------------------------


class TestIdentityUniqueness:
    def test_duplicate_identity_raises(self) -> None:
        # Default factor_id="factor" on both → same identity → duplicate.
        # The previous auto-cell-split would have hidden this; #161
        # surfaces it.
        profiles = [
            _profile(primary_p=0.01, metric=Metric.IC),
            _profile(primary_p=0.02, metric=Metric.FM),
        ]
        with pytest.raises(UserInputError, match="duplicate"):
            bhy(profiles)


# ---------------------------------------------------------------------------
# Deprecated v0.4 kwargs (threshold= → q=)
# ---------------------------------------------------------------------------


class TestDeprecatedKwargs:
    def test_threshold_alias_warns(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        with pytest.warns(DeprecationWarning, match="threshold"):
            result = bhy([prof], threshold=0.05)
        assert result.profiles == [prof]

    def test_threshold_and_q_collide(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        with (
            pytest.raises(TypeError, match="not both"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            bhy([prof], threshold=0.05, q=0.01)

    def test_unknown_kwarg_raises(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.04)
        with pytest.raises(TypeError, match="unexpected keyword"):
            bhy([prof], wibble=1)


# ---------------------------------------------------------------------------
# Mixed-horizon migration foot-gun (#161 contract change)
# ---------------------------------------------------------------------------


class TestMixedHorizonWarning:
    def test_mixed_forward_periods_without_expand_over_warns(self) -> None:
        # bhy used to auto-isolate horizons; caller now must split.
        # Warn loud so a sweep that previously enjoyed implicit
        # isolation does not silently inflate FDR after upgrade.
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


# ---------------------------------------------------------------------------
# Survivors container (#171, batch 1: dataclass + repr/HTML, no bhy plumb yet)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Survivors contract: bhy producer-side (#171, batch 2)
# ---------------------------------------------------------------------------


class TestBhyReturnsSurvivors:
    def test_return_type_is_survivors(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.001)
        assert isinstance(bhy([prof]), Survivors)

    def test_adj_q_matches_bhy_adjusted_p_and_mask_definitional(self) -> None:
        # Single contract: survivor set == {i: adj_q[i] <= q}, and
        # adj_q values come from bhy_adjusted_p of the bucket-local
        # p_array. Both halves of the duality in one assertion block.
        from factrix.stats.multiple_testing import bhy_adjusted_p

        ps = [0.001, 0.02, 0.04, 0.5]
        profiles = [_profile(factor_id=f"f{i}", primary_p=p) for i, p in enumerate(ps)]
        q = 0.05
        result = bhy(profiles, q=q)
        full_adj = bhy_adjusted_p(np.array(ps))
        expected_ids = [profiles[i].factor_id for i, a in enumerate(full_adj) if a <= q]
        assert [p.factor_id for p in result.profiles] == expected_ids
        np.testing.assert_allclose(
            result.adj_q,
            [full_adj[i] for i in range(len(ps)) if full_adj[i] <= q],
            rtol=1e-12,
        )
        assert (result.adj_q <= q).all()

    def test_n_total_and_adj_q_per_bucket(self) -> None:
        # Multi-bucket: n_total per bucket AND adj_q per surviving
        # profile equals bhy_adjusted_p of that profile's own bucket
        # slice (not pooled across buckets).
        from factrix.stats.multiple_testing import bhy_adjusted_p

        bull_ps = [0.001, 0.002, 0.003]
        profiles = [
            _profile(factor_id="f1", primary_p=bull_ps[0], context={"regime": "bull"}),
            _profile(factor_id="f2", primary_p=bull_ps[1], context={"regime": "bull"}),
            _profile(factor_id="f3", primary_p=bull_ps[2], context={"regime": "bull"}),
            _profile(factor_id="f1", primary_p=0.5, context={"regime": "bear"}),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = bhy(profiles, expand_over=["regime"])

        assert result.n_total == {("bull",): 3, ("bear",): 1}

        bull_adj = bhy_adjusted_p(np.array(bull_ps))
        for prof, adj in zip(result.profiles, result.adj_q, strict=True):
            assert prof.context["regime"] == "bull"  # bear bucket fails
            bull_idx = ["f1", "f2", "f3"].index(prof.factor_id)
            assert adj == pytest.approx(bull_adj[bull_idx], rel=1e-12)

    def test_ties_in_p_handled(self) -> None:
        # Tied p-values are a known step-up edge: adj_q stays a
        # function of bhy_adjusted_p only — no separate tie-handling
        # path drift.
        from factrix.stats.multiple_testing import bhy_adjusted_p

        ps = [0.01, 0.01, 0.5]
        profiles = [_profile(factor_id=f"f{i}", primary_p=p) for i, p in enumerate(ps)]
        result = bhy(profiles, q=0.05)
        full_adj = bhy_adjusted_p(np.array(ps))
        np.testing.assert_allclose(
            result.adj_q,
            [full_adj[i] for i in range(len(ps)) if full_adj[i] <= 0.05],
            rtol=1e-12,
        )

    def test_expand_over_tuple_reflects_input(self) -> None:
        profiles = [_profile(factor_id="f1", primary_p=0.001)]
        assert bhy(profiles).expand_over == ()
        profiles_x = [
            _profile(factor_id="f1", primary_p=0.001, context={"u": "a"}),
            _profile(factor_id="f2", primary_p=0.001, context={"u": "a"}),
        ]
        assert bhy(profiles_x, expand_over=["u"]).expand_over == ("u",)

    def test_q_field_records_nominal_target(self) -> None:
        prof = _profile(factor_id="f1", primary_p=0.001)
        assert bhy([prof], q=0.1).q == 0.1
        assert bhy([prof]).q == 0.05  # default


@pytest.fixture
def surv_single_bucket() -> Survivors:
    profiles = [
        _profile(factor_id="f1", primary_p=0.001),
        _profile(factor_id="f2", primary_p=0.012),
    ]
    return Survivors(
        profiles=profiles,
        adj_q=np.array([0.002, 0.024]),
        q=0.05,
        expand_over=(),
        n_total={(): 2},
    )


@pytest.fixture
def surv_multi_bucket() -> Survivors:
    profiles = [
        _profile(factor_id="f1", primary_p=0.001, context={"universe_id": "tw50"}),
        _profile(factor_id="f2", primary_p=0.020, context={"universe_id": "tw100"}),
    ]
    return Survivors(
        profiles=profiles,
        adj_q=np.array([0.002, 0.040]),
        q=0.05,
        expand_over=("universe_id",),
        n_total={("tw50",): 1, ("tw100",): 1},
    )


class TestSurvivorsBasics:
    def test_len_matches_profiles(self, surv_single_bucket: Survivors) -> None:
        assert len(surv_single_bucket) == 2

    def test_frozen_dataclass(self, surv_single_bucket: Survivors) -> None:
        with pytest.raises(FrozenInstanceError):
            surv_single_bucket.q = 0.1  # type: ignore[misc]


class TestSurvivorsReprText:
    def test_single_bucket_three_columns(self, surv_single_bucket: Survivors) -> None:
        text = repr(surv_single_bucket)
        assert "Survivors(" in text
        assert "n=2" in text and "q=0.05" in text
        assert "identity" in text and "primary_p" in text and "adj_q" in text
        assert "expand_over_values" not in text
        assert "'f1'" in text and "'f2'" in text

    def test_multi_bucket_adds_expand_over_column(
        self, surv_multi_bucket: Survivors
    ) -> None:
        text = repr(surv_multi_bucket)
        assert "expand_over_values" in text
        assert "expand_over=['universe_id']" in text
        assert "tw50" in text and "tw100" in text

    def test_empty_survivors_renders_header_only(self) -> None:
        empty = Survivors(
            profiles=[],
            adj_q=np.array([]),
            q=0.05,
            expand_over=(),
            n_total={(): 0},
        )
        text = repr(empty)
        assert text.startswith("Survivors(")
        assert "\n" not in text


class TestSurvivorsReprHtml:
    def test_html_has_table_and_caption(self, surv_single_bucket: Survivors) -> None:
        markup = surv_single_bucket._repr_html_()
        assert markup.startswith("<table")
        assert "factrix-survivors" in markup
        assert "<caption>" in markup and "Survivors" in markup
        assert "<thead>" in markup and "<tbody>" in markup

    def test_html_three_columns_single_bucket(
        self, surv_single_bucket: Survivors
    ) -> None:
        markup = surv_single_bucket._repr_html_()
        assert markup.count("<th ") == 3
        assert "expand_over_values" not in markup

    def test_html_four_columns_multi_bucket(self, surv_multi_bucket: Survivors) -> None:
        markup = surv_multi_bucket._repr_html_()
        assert markup.count("<th ") == 4
        assert "expand_over_values" in markup

    def test_html_escapes_special_chars(self) -> None:
        profile = _profile(
            factor_id="<bad>",
            primary_p=0.01,
            context={"universe_id": "<x>"},
        )
        surv = Survivors(
            profiles=[profile],
            adj_q=np.array([0.02]),
            q=0.05,
            expand_over=("universe_id",),
            n_total={("<x>",): 1},
        )
        markup = surv._repr_html_()
        assert "<bad>" not in markup
        assert "&lt;bad&gt;" in markup


class TestSurvivorsBackRefIdentity:
    def test_profile_back_ref_is_input_identity(self) -> None:
        # Contract: Survivors holds the same FactorProfile objects fed
        # in (is-identity, not just ==). Lets users walk back to
        # context / stats without re-resolution.
        original = [
            _profile(factor_id="f1", primary_p=0.001),
            _profile(factor_id="f2", primary_p=0.012),
        ]
        surv = Survivors(
            profiles=original,
            adj_q=np.array([0.002, 0.024]),
            q=0.05,
            expand_over=(),
            n_total={(): 2},
        )
        for inp, out in zip(original, surv.profiles, strict=True):
            assert inp is out
