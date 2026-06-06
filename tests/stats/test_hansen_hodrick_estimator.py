"""``HansenHodrick`` Estimator instance tests.

Cell-agnostic dispatch (`emits_for` → `StatCode.P_HH`) plus restricted
applicability ((INDIVIDUAL, DENSE) only). Math correctness is owned
by ``tests/stats/test_hansen_hodrick.py``.
"""

from __future__ import annotations

import pytest
from factrix import list_estimators
from factrix._axis import FactorDensity, FactorScope
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
    ("scope", "density", "expected"),
    [
        (FactorScope.INDIVIDUAL, FactorDensity.DENSE, True),
        (FactorScope.INDIVIDUAL, FactorDensity.SPARSE, False),
        (FactorScope.COMMON, FactorDensity.DENSE, False),
        (FactorScope.COMMON, FactorDensity.SPARSE, False),
    ],
)
def test_applicable_to(
    scope: FactorScope, density: FactorDensity, expected: bool
) -> None:
    assert HansenHodrick().applicable_to(scope, density) is expected


def test_emits_for_returns_p_hh() -> None:
    code = HansenHodrick().emits_for(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
    assert code is StatCode.P_HH


def test_listed_for_individual_continuous() -> None:
    names = list_estimators(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
    assert "HansenHodrick" in names
    assert "NeweyWest" in names


def test_not_listed_for_sparse_or_common() -> None:
    for scope, density in [
        (FactorScope.INDIVIDUAL, FactorDensity.SPARSE),
        (FactorScope.COMMON, FactorDensity.DENSE),
        (FactorScope.COMMON, FactorDensity.SPARSE),
    ]:
        assert "HansenHodrick" not in list_estimators(scope, density)
