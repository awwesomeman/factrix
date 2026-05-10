"""``NeweyWest`` reference Estimator tests (#170, #187).

Validates the cell-agnostic dispatch surface (`emits_for` always returns
`StatCode.P`) and applicability across user-facing cells. Numerical
correctness of the underlying NW HAC math is owned by
``tests/stats/test_newey_west.py``.
"""

from __future__ import annotations

import pytest
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import StatCode
from factrix._registry import _DISPATCH_REGISTRY
from factrix.stats import Estimator, NeweyWest


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
    ("scope", "signal", "metric"),
    [
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC),
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.FM),
        (FactorScope.INDIVIDUAL, Signal.SPARSE, None),
        (FactorScope.COMMON, Signal.CONTINUOUS, None),
        (FactorScope.COMMON, Signal.SPARSE, None),
    ],
)
def test_emits_for_returns_flat_p(
    scope: FactorScope,
    signal: Signal,
    metric: Metric | None,
) -> None:
    assert NeweyWest().emits_for(scope, signal, metric) is StatCode.P


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


def test_panel_procedures_emit_the_dispatched_code() -> None:
    # Every PANEL procedure must populate StatCode.P, since NeweyWest
    # dispatches there cell-agnostically. Drift would surface as a
    # runtime KeyError on bhy(estimator=NeweyWest()).
    for key, entry in _DISPATCH_REGISTRY.items():
        if key.mode is not Mode.PANEL:
            continue
        assert StatCode.P in entry.procedure.EMITS_STATS, (
            f"PANEL procedure for ({key.scope}, {key.signal}, {key.metric}) "
            f"does not emit StatCode.P: "
            f"{sorted(s.name for s in entry.procedure.EMITS_STATS)}"
        )
