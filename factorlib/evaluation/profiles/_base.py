"""Base types for the FactorProfile architecture.

Per-factor-type profile dataclasses structurally match the ``FactorProfile``
Protocol defined here; they are registered via ``@register_profile(ft)`` so
that ``fl.evaluate()`` can dispatch on ``FactorType``.

See ``docs/gate_redesign_v2.md`` (ADR) and ``docs/plan_gate_redesign.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

from factorlib._types import Diagnostic, FactorType, PValue, Verdict

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts


@runtime_checkable
class FactorProfile(Protocol):
    """Structural interface for all typed factor profiles.

    Concrete profiles are frozen slotted dataclasses that match this shape;
    they do not inherit from the Protocol (avoids dataclass/ABC metaclass
    fights). Registration is done by the ``@register_profile`` decorator.

    Required class-level metadata:
        CANONICAL_P_FIELD: name of the p-value field used for BHY and the
            default ``verdict()`` decision. Must be one of P_VALUE_FIELDS.
        P_VALUE_FIELDS: frozenset of field names that hold genuine p-values.
            Whitelisted by ``ProfileSet.multiple_testing_correct`` to prevent
            composed-p abuse (e.g. ``min(ic_p, spread_p)`` being fed to BHY).

    Required members:
        canonical_p: property returning the canonical test p-value (single
            source of truth for BHY inputs).
        verdict(threshold): binary PASS/FAILED on canonical p only. Any
            "significant-but-with-caveats" nuance belongs in diagnose().
        diagnose(): contextual hints as ``list[Diagnostic]``.
        from_artifacts(artifacts): classmethod constructor from Artifacts.
    """

    factor_name: str
    n_periods: int

    CANONICAL_P_FIELD: ClassVar[str]
    P_VALUE_FIELDS: ClassVar[frozenset[str]]

    @property
    def canonical_p(self) -> PValue: ...

    def verdict(self, threshold: float = 2.0) -> Verdict: ...

    def diagnose(self) -> list[Diagnostic]: ...

    @classmethod
    def from_artifacts(cls, artifacts: "Artifacts") -> Self: ...


# Registry populated by @register_profile; consumed by fl.evaluate() dispatch.
_PROFILE_REGISTRY: dict[FactorType, type[FactorProfile]] = {}


def register_profile(factor_type: FactorType):
    """Class decorator: register a Profile dataclass under a FactorType.

    Validates required ClassVars and members at decoration time to catch
    mistakes before the class is ever instantiated.

    Usage::

        @register_profile(FactorType.CROSS_SECTIONAL)
        @dataclass(frozen=True, slots=True)
        class CrossSectionalProfile:
            ...
    """

    def decorator(cls: type) -> type:
        # Enforce ClassVars
        for attr in ("CANONICAL_P_FIELD", "P_VALUE_FIELDS"):
            if not hasattr(cls, attr):
                raise TypeError(
                    f"{cls.__name__} registered as {factor_type} profile but "
                    f"is missing required ClassVar {attr!r}."
                )

        # Enforce members
        for member in ("canonical_p", "verdict", "diagnose", "from_artifacts"):
            if not hasattr(cls, member):
                raise TypeError(
                    f"{cls.__name__} missing required member {member!r}."
                )

        # Enforce canonical ⊆ whitelist invariant
        canonical = cls.CANONICAL_P_FIELD
        whitelist = cls.P_VALUE_FIELDS
        if canonical not in whitelist:
            raise TypeError(
                f"{cls.__name__}.CANONICAL_P_FIELD={canonical!r} is not in "
                f"P_VALUE_FIELDS={whitelist!r}. Canonical must be whitelisted."
            )

        # Prevent silent double-registration
        if factor_type in _PROFILE_REGISTRY:
            prior = _PROFILE_REGISTRY[factor_type].__name__
            raise RuntimeError(
                f"Profile for {factor_type} already registered as {prior}; "
                f"refusing to overwrite with {cls.__name__}."
            )

        _PROFILE_REGISTRY[factor_type] = cls
        return cls

    return decorator


def get_profile_class(factor_type: FactorType) -> type[FactorProfile]:
    """Look up the Profile class registered under ``factor_type``.

    Raises KeyError with a helpful message listing registered types.
    """
    try:
        return _PROFILE_REGISTRY[factor_type]
    except KeyError:
        registered = sorted(ft.value for ft in _PROFILE_REGISTRY)
        raise KeyError(
            f"No profile registered for {factor_type!r}. "
            f"Registered: {registered}."
        ) from None
