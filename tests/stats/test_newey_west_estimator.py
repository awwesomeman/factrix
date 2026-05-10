"""``NeweyWest`` reference Estimator tests (#170).

Validates the dispatch table that maps each user-facing cell to its
procedure-canonical NW HAC p-value StatCode. Numerical correctness of
the underlying NW HAC math is owned by ``tests/stats/test_newey_west.py``;
this file only exercises the metadata + dispatch surface that
``_resolve_family`` will rely on.
"""

from __future__ import annotations

import pytest
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import StatCode
from factrix._registry import _DISPATCH_REGISTRY, _DispatchKey
from factrix.stats import Estimator, NeweyWest
from factrix.stats.newey_west import _EMITS


def test_satisfies_estimator_protocol() -> None:
    assert isinstance(NeweyWest(), Estimator)


def test_name_uses_class_identifier() -> None:
    assert NeweyWest().name == "NeweyWest"


def test_description_is_cell_agnostic() -> None:
    desc = NeweyWest().description
    assert "Bartlett" in desc
    assert "auto-bandwidth" in desc
    # Cell semantics ("IC mean", "CAAR series", etc.) belong in
    # _STAT_DESCRIPTIONS, not on the estimator itself.
    assert "IC" not in desc and "CAAR" not in desc


@pytest.mark.parametrize(
    ("scope", "signal", "metric", "expected"),
    [
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC, StatCode.IC_P),
        (
            FactorScope.INDIVIDUAL,
            Signal.CONTINUOUS,
            Metric.FM,
            StatCode.FM_LAMBDA_P,
        ),
        (FactorScope.INDIVIDUAL, Signal.SPARSE, None, StatCode.CAAR_P),
        (FactorScope.COMMON, Signal.CONTINUOUS, None, StatCode.TS_BETA_P),
        (FactorScope.COMMON, Signal.SPARSE, None, StatCode.TS_BETA_P),
    ],
)
def test_emits_for_known_cells(
    scope: FactorScope,
    signal: Signal,
    metric: Metric | None,
    expected: StatCode,
) -> None:
    assert NeweyWest().emits_for(scope, signal, metric) is expected


@pytest.mark.parametrize(
    ("scope", "signal"),
    [
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS),
        (FactorScope.INDIVIDUAL, Signal.SPARSE),
        (FactorScope.COMMON, Signal.CONTINUOUS),
        (FactorScope.COMMON, Signal.SPARSE),
    ],
)
def test_applicable_to_all_user_facing_cells(
    scope: FactorScope, signal: Signal
) -> None:
    assert NeweyWest().applicable_to(scope, signal)


def test_emits_table_stays_in_sync_with_dispatch_registry() -> None:
    # The _EMITS dispatch table is a hardcoded reverse index from cell
    # to procedure-emitted p-value StatCode. If a procedure's
    # EMITS_STATS gains or drops a *_P entry, this test fails so the
    # estimator side is updated in the same PR (drift would otherwise
    # surface as a runtime KeyError on bhy(estimator=NeweyWest())).
    for (scope, signal, metric), code in _EMITS.items():
        entry = _DISPATCH_REGISTRY.get(_DispatchKey(scope, signal, metric, Mode.PANEL))
        assert entry is not None, (
            f"_EMITS lists ({scope.value}, {signal.value}, {metric}) "
            "but no PANEL procedure is registered for it"
        )
        assert code in entry.procedure.EMITS_STATS, (
            f"NeweyWest dispatches ({scope.value}, {signal.value}, "
            f"{metric}) → {code.name}, but the procedure's EMITS_STATS "
            f"does not include it: {sorted(s.name for s in entry.procedure.EMITS_STATS)}"
        )
