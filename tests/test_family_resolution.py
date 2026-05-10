"""Coverage for the shared family resolution layer (#161, #170)."""

from __future__ import annotations

import pytest
from factrix import AnalysisConfig, Metric
from factrix._axis import FactorScope, Mode, Signal
from factrix._codes import StatCode
from factrix._errors import UserInputError
from factrix._family import _resolve_family
from factrix._profile import FactorProfile
from factrix.stats import NeweyWest


def _cfg(forward_periods: int = 5) -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(
        metric=Metric.IC, forward_periods=forward_periods
    )


def _profile(
    *,
    factor_id: str = "f1",
    forward_periods: int = 5,
    primary_p: float = 0.04,
    context: dict[str, object] | None = None,
    stats: dict[StatCode, float] | None = None,
) -> FactorProfile:
    return FactorProfile(
        config=_cfg(forward_periods),
        mode=Mode.PANEL,
        primary_p=primary_p,
        n_obs=60,
        n_assets=20,
        factor_id=factor_id,
        context=context or {},
        stats=stats or {},
    )


def test_default_path_no_expand_over_returns_one_entry_per_profile() -> None:
    profiles = [_profile(factor_id="f1"), _profile(factor_id="f2")]
    entries = _resolve_family(profiles, verb="bhy")
    assert [e.identity for e in entries] == [("f1", 5), ("f2", 5)]
    assert all(e.expand_over_values == () for e in entries)
    assert [e.p_value for e in entries] == [0.04, 0.04]


def test_estimator_none_falls_back_to_primary_p() -> None:
    p = _profile(primary_p=0.012, stats={StatCode.IC_P: 0.5})
    [entry] = _resolve_family([p], verb="bhy")
    assert entry.p_value == 0.012


def test_estimator_supplied_reads_dispatched_stat() -> None:
    p = _profile(primary_p=0.5, stats={StatCode.IC_P: 0.012})
    [entry] = _resolve_family([p], verb="bhy", estimator=NeweyWest())
    assert entry.p_value == 0.012


def test_estimator_not_applicable_to_cell_raises() -> None:
    class _Picky:
        @property
        def name(self) -> str:
            return "Picky"

        @property
        def description(self) -> str:
            return "Rejects every cell."

        def applicable_to(self, scope: FactorScope, signal: Signal) -> bool:
            return False

        def emits_for(
            self, scope: FactorScope, signal: Signal, metric: object
        ) -> StatCode:
            raise AssertionError("emits_for must not be called when not applicable")

    p = _profile(stats={StatCode.IC_P: 0.04})
    with pytest.raises(UserInputError) as exc:
        _resolve_family([p], verb="bhy", estimator=_Picky())
    err = exc.value
    assert err.field == "estimator"
    assert err.value == "Picky"
    assert "applicable" in (err.expected or "")
    assert "api/bhy#estimator" in err.docs_url


def test_expand_over_single_dim_partitions_per_context_value() -> None:
    profiles = [
        _profile(factor_id="f1", context={"universe_id": "tw50"}),
        _profile(factor_id="f1", context={"universe_id": "tw100"}),
    ]
    entries = _resolve_family(profiles, verb="bhy", expand_over=["universe_id"])
    assert [e.expand_over_values for e in entries] == [("tw50",), ("tw100",)]


def test_expand_over_multi_dim_keeps_caller_key_order() -> None:
    profiles = [
        _profile(factor_id="f1", context={"universe_id": "tw50", "regime_id": "bull"}),
        _profile(factor_id="f1", context={"universe_id": "tw50", "regime_id": "bear"}),
    ]
    entries = _resolve_family(
        profiles, verb="bhy", expand_over=["universe_id", "regime_id"]
    )
    assert [e.expand_over_values for e in entries] == [
        ("tw50", "bull"),
        ("tw50", "bear"),
    ]


def test_unknown_expand_over_raises_with_fuzzy_suggestion() -> None:
    profiles = [_profile(context={"universe_id": "tw50", "regime_id": "bull"})]
    with pytest.raises(UserInputError) as exc:
        _resolve_family(profiles, verb="bhy", expand_over=["univere_id"])
    err = exc.value
    assert err.field == "expand_over"
    assert err.value == "univere_id"
    assert "universe_id" in err.suggestions
    assert "api/bhy#expand_over" in err.docs_url


def test_expand_over_missing_on_some_profiles_raises() -> None:
    profiles = [
        _profile(context={"universe_id": "tw50", "regime_id": "bull"}),
        _profile(context={"universe_id": "tw50"}),  # no regime_id
    ]
    with pytest.raises(UserInputError) as exc:
        _resolve_family(profiles, verb="bhy", expand_over=["regime_id"])
    assert exc.value.field == "expand_over"
    assert exc.value.value == "regime_id"


@pytest.mark.parametrize("identity_field", ["factor_id", "forward_periods"])
def test_expand_over_hitting_identity_raises(identity_field: str) -> None:
    profiles = [_profile(context={"universe_id": "tw50"})]
    with pytest.raises(UserInputError) as exc:
        _resolve_family(profiles, verb="bhy", expand_over=[identity_field])
    err = exc.value
    assert err.field == "expand_over"
    assert err.value == identity_field
    assert "identity" in (err.expected or "")


def test_duplicate_partition_key_raises() -> None:
    profiles = [
        _profile(factor_id="f1", forward_periods=5),
        _profile(factor_id="f1", forward_periods=5),
    ]
    with pytest.raises(UserInputError) as exc:
        _resolve_family(profiles, verb="bhy")
    assert exc.value.field == "profiles"
    assert "duplicate" in str(exc.value)


def test_duplicate_partition_key_with_expand_over_compares_full_key() -> None:
    # Same identity but different expand_over values → no duplicate.
    profiles = [
        _profile(factor_id="f1", context={"universe_id": "tw50"}),
        _profile(factor_id="f1", context={"universe_id": "tw100"}),
    ]
    entries = _resolve_family(profiles, verb="bhy", expand_over=["universe_id"])
    assert len(entries) == 2

    # Same identity AND same expand_over value → duplicate.
    dup = [
        _profile(factor_id="f1", context={"universe_id": "tw50"}),
        _profile(factor_id="f1", context={"universe_id": "tw50"}),
    ]
    with pytest.raises(UserInputError):
        _resolve_family(dup, verb="bhy", expand_over=["universe_id"])


def test_missing_dispatched_stat_raises_with_available_keys() -> None:
    # FM cell profile, but stats only has IC_P populated → NeweyWest
    # dispatches to FM_LAMBDA_P which is missing.
    cfg = AnalysisConfig.individual_continuous(metric=Metric.FM, forward_periods=5)
    p = FactorProfile(
        config=cfg,
        mode=Mode.PANEL,
        primary_p=0.04,
        n_obs=60,
        n_assets=20,
        factor_id="f1",
        stats={StatCode.IC_P: 0.04},
    )
    with pytest.raises(UserInputError) as exc:
        _resolve_family([p], verb="bhy", estimator=NeweyWest())
    err = exc.value
    assert err.field == "estimator"
    assert err.value == "NeweyWest"
    assert "FM_LAMBDA_P" in (err.expected or "")
    assert "IC_P" in err.candidates
    assert "api/bhy#estimator" in err.docs_url
