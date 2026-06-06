"""Estimator protocol surface tests.

Verify the runtime-checkable protocol shape so that downstream `_resolve_family`
dispatch can rely on `isinstance(obj, Estimator)` and the four required members
(`name` / `description` / `applicable_to` / `emits_for`).
"""

from __future__ import annotations

from factrix._axis import FactorDensity, FactorScope
from factrix._codes import StatCode
from factrix.stats import Estimator


class _Stub:
    """Minimal Estimator-conforming class for protocol surface checks."""

    @property
    def name(self) -> str:
        return "Stub"

    @property
    def description(self) -> str:
        return "Test stub."

    def applicable_to(self, scope: FactorScope, density: FactorDensity) -> bool:
        return scope is FactorScope.INDIVIDUAL and density is FactorDensity.DENSE

    def emits_for(self, scope: FactorScope, density: FactorDensity) -> StatCode:
        return StatCode.P_NW


class _MissingEmitsFor:
    @property
    def name(self) -> str:
        return "x"

    @property
    def description(self) -> str:
        return "x"

    def applicable_to(self, scope: FactorScope, density: FactorDensity) -> bool:
        return True


def test_runtime_checkable_accepts_conforming_stub() -> None:
    assert isinstance(_Stub(), Estimator)


def test_runtime_checkable_rejects_missing_member() -> None:
    assert not isinstance(_MissingEmitsFor(), Estimator)


def test_member_calls_route_through_protocol() -> None:
    e: Estimator = _Stub()
    assert e.name == "Stub"
    assert e.description == "Test stub."
    assert e.applicable_to(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
    assert not e.applicable_to(FactorScope.COMMON, FactorDensity.SPARSE)
    assert e.emits_for(FactorScope.INDIVIDUAL, FactorDensity.DENSE) is StatCode.P_NW
