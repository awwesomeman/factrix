"""``HansenHodrick`` Estimator instance tests (#184).

Cell-agnostic dispatch (`emits_for` → `StatCode.P_HH`) plus restricted
applicability ((INDIVIDUAL, CONTINUOUS) only). Math correctness is owned
by ``tests/stats/test_hansen_hodrick.py``.
"""

from __future__ import annotations

import pytest
from factrix import list_estimators
from factrix._axis import FactorScope, Metric, Signal
from factrix._codes import StatCode
from factrix.stats import Estimator, HansenHodrick


def test_satisfies_estimator_protocol() -> None:
    assert isinstance(HansenHodrick(), Estimator)


def test_name_uses_class_identifier() -> None:
    assert HansenHodrick().name == "HansenHodrick"


def test_description_is_cell_agnostic() -> None:
    desc = HansenHodrick().description
    assert "Hansen-Hodrick" in desc
    assert "rectangular" in desc
    assert "IC" not in desc and "FM" not in desc


@pytest.mark.parametrize(
    ("scope", "signal", "expected"),
    [
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, True),
        (FactorScope.INDIVIDUAL, Signal.SPARSE, False),
        (FactorScope.COMMON, Signal.CONTINUOUS, False),
        (FactorScope.COMMON, Signal.SPARSE, False),
    ],
)
def test_applicable_to(scope: FactorScope, signal: Signal, expected: bool) -> None:
    assert HansenHodrick().applicable_to(scope, signal) is expected


@pytest.mark.parametrize("metric", [Metric.IC, Metric.FM])
def test_emits_for_returns_p_hh(metric: Metric) -> None:
    code = HansenHodrick().emits_for(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, metric)
    assert code is StatCode.P_HH


def test_listed_for_individual_continuous() -> None:
    names = list_estimators(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)
    assert "HansenHodrick" in names
    assert "NeweyWest" in names


def test_not_listed_for_sparse_or_common() -> None:
    for scope, signal in [
        (FactorScope.INDIVIDUAL, Signal.SPARSE),
        (FactorScope.COMMON, Signal.CONTINUOUS),
        (FactorScope.COMMON, Signal.SPARSE),
    ]:
        assert "HansenHodrick" not in list_estimators(scope, signal)
