"""Validation paths for the @register_profile decorator.

The decorator enforces four invariants at class-definition time:
1. Required ClassVars are present (CANONICAL_P_FIELD, P_VALUE_FIELDS).
2. Required members are present (canonical_p / verdict / diagnose /
   from_artifacts).
3. CANONICAL_P_FIELD is a member of P_VALUE_FIELDS.
4. No two classes register for the same FactorType.

If any of these ever silently regresses, BHY runs downstream would
accept classes that return meaningless p-values. The runtime check in
profile_set.multiple_testing_correct catches (3) but only after a
mutation — these tests lock the up-front guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

import pytest

from factrix._types import Diagnostic, FactorType, PValue, Verdict
from factrix.evaluation.profiles._base import (
    _PROFILE_REGISTRY,
    register_profile,
)


def _with_clean_registry(factor_type: FactorType):
    """Temporarily unregister a FactorType so the decorator can register
    a fresh test class for it without colliding with production."""

    class _Restore:
        def __enter__(self):
            self.prior = _PROFILE_REGISTRY.pop(factor_type, None)
            return self

        def __exit__(self, *exc):
            if self.prior is not None:
                _PROFILE_REGISTRY[factor_type] = self.prior
            else:
                _PROFILE_REGISTRY.pop(factor_type, None)

    return _Restore()


class TestMissingClassVars:
    def test_missing_canonical_p_field_raises(self):
        with _with_clean_registry(FactorType.CROSS_SECTIONAL):
            with pytest.raises(TypeError, match="CANONICAL_P_FIELD"):

                @register_profile(FactorType.CROSS_SECTIONAL)
                @dataclass(frozen=True, slots=True)
                class _Bad:
                    factor_name: str = ""
                    n_periods: int = 0
                    P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset()

                    @property
                    def canonical_p(self) -> PValue:
                        return PValue(1.0)

                    def verdict(self, threshold: float = 2.0) -> Verdict:
                        return "FAILED"

                    def diagnose(self) -> list[Diagnostic]:
                        return []

                    @classmethod
                    def from_artifacts(cls, artifacts) -> Self:  # noqa: ARG003
                        return cls()

    def test_missing_p_value_fields_raises(self):
        with _with_clean_registry(FactorType.CROSS_SECTIONAL):
            with pytest.raises(TypeError, match="P_VALUE_FIELDS"):

                @register_profile(FactorType.CROSS_SECTIONAL)
                @dataclass(frozen=True, slots=True)
                class _Bad:
                    factor_name: str = ""
                    n_periods: int = 0
                    CANONICAL_P_FIELD: ClassVar[str] = "ic_p"

                    @property
                    def canonical_p(self) -> PValue:
                        return PValue(1.0)

                    def verdict(self, threshold: float = 2.0) -> Verdict:
                        return "FAILED"

                    def diagnose(self) -> list[Diagnostic]:
                        return []

                    @classmethod
                    def from_artifacts(cls, artifacts) -> Self:  # noqa: ARG003
                        return cls()


class TestCanonicalNotInWhitelist:
    def test_raises(self):
        with _with_clean_registry(FactorType.CROSS_SECTIONAL):
            with pytest.raises(TypeError, match="not in P_VALUE_FIELDS"):

                @register_profile(FactorType.CROSS_SECTIONAL)
                @dataclass(frozen=True, slots=True)
                class _Bad:
                    factor_name: str = ""
                    n_periods: int = 0
                    CANONICAL_P_FIELD: ClassVar[str] = "missing_from_whitelist"
                    P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({"ic_p"})

                    @property
                    def canonical_p(self) -> PValue:
                        return PValue(1.0)

                    def verdict(self, threshold: float = 2.0) -> Verdict:
                        return "FAILED"

                    def diagnose(self) -> list[Diagnostic]:
                        return []

                    @classmethod
                    def from_artifacts(cls, artifacts) -> Self:  # noqa: ARG003
                        return cls()


class TestDoubleRegistration:
    def test_second_register_same_type_raises(self):
        # Don't clean the registry: the point is that the real production
        # class is already registered and a second registration under the
        # same FactorType must be refused.
        with pytest.raises(RuntimeError, match="already registered"):

            @register_profile(FactorType.CROSS_SECTIONAL)
            @dataclass(frozen=True, slots=True)
            class _Shadow:
                factor_name: str = ""
                n_periods: int = 0
                CANONICAL_P_FIELD: ClassVar[str] = "ic_p"
                P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({"ic_p"})
                ic_p: PValue = PValue(1.0)

                @property
                def canonical_p(self) -> PValue:
                    return self.ic_p

                def verdict(self, threshold: float = 2.0) -> Verdict:
                    return "FAILED"

                def diagnose(self) -> list[Diagnostic]:
                    return []

                @classmethod
                def from_artifacts(cls, artifacts) -> Self:  # noqa: ARG003
                    return cls()
