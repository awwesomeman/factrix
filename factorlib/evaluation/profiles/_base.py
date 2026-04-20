"""Base types for the FactorProfile architecture.

Per-factor-type profile dataclasses structurally match the ``FactorProfile``
Protocol defined here; they are registered via ``@register_profile(ft)`` so
that ``fl.evaluate()`` can dispatch on ``FactorType``.

See ``docs/gate_redesign_v2.md`` (ADR) and ``docs/plan_gate_redesign.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self, runtime_checkable

from factorlib._types import Diagnostic, FactorType, MetricOutput, PValue, Verdict

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts


def _pv(m: MetricOutput) -> PValue:
    """Extract a p-value from a ``MetricOutput``.

    Metric functions store p-values under ``metadata["p_value"]``; when
    the metric is purely descriptive (e.g. ``ic_ir``) or short-circuits
    on insufficient data, no p-value is present. We default to 1.0 — the
    most conservative "definitely not significant" — so downstream BHY
    treats data-starved factors as rejected rather than crashing.
    """
    return PValue(float(m.metadata.get("p_value", 1.0)))


def _insufficient_metrics(metrics: "dict[str, MetricOutput]") -> tuple[str, ...]:
    """Return the names of metrics that short-circuited on insufficient data.

    A metric signals short-circuit by writing ``metadata["reason"]`` whose
    value starts with ``insufficient_`` or ``no_`` (the vocabulary agreed
    across ``factorlib/metrics/*``). Reasons like
    ``not_applicable_discrete_signal`` or ``degenerate_ic_variance`` are
    *intentional* skips, not data shortages, and are excluded so they
    don't trigger the ``data.insufficient`` diagnose rule.

    Accepts a ``{public_name: MetricOutput}`` map so the caller chooses
    which profile field name to expose — MetricOutput.name is the generic
    statistic identifier (``"ic"``, ``"caar"``), but the profile wants
    the user-facing field it populates.
    """
    _INSUFFICIENT_PREFIXES = ("insufficient_", "no_")
    hits: list[str] = []
    for public_name, m in metrics.items():
        reason = m.metadata.get("reason")
        if not isinstance(reason, str):
            continue
        if reason.startswith(_INSUFFICIENT_PREFIXES):
            hits.append(public_name)
    return tuple(hits)


def _stash(store: "dict[str, MetricOutput]", m: MetricOutput) -> MetricOutput:
    """Store ``m`` in ``store`` under ``m.name`` with metadata wrapped read-only.

    Returns ``m`` unchanged so callers can chain ``x = _stash(outputs, f(...))``.
    The stored copy's metadata is a ``MappingProxyType`` view — downstream
    mutation raises ``TypeError``. The returned ``m`` keeps its mutable
    metadata so helpers like ``_pv(m)`` can read it without indirection.
    """
    store[m.name] = MetricOutput(
        name=m.name,
        value=m.value,
        stat=m.stat,
        significance=m.significance,
        metadata=MappingProxyType(m.metadata),
    )
    return m


def _memoized(
    store: "dict[str, MetricOutput]",
    key: str,
    fn: Callable[..., MetricOutput],
    *args: Any,
    **kwargs: Any,
) -> MetricOutput:
    """Return ``store[key]`` if present, else compute ``fn(*args, **kwargs)``
    and ``_stash`` the result.

    Single source of truth for metric caching in ``from_artifacts``:
    whether an entry was pre-populated by ``_augment_level2_intermediates``
    (L2 opt-in metrics), by a ``Factor`` session method call, or by a prior
    ``from_artifacts`` pass, callers never recompute and never overwrite.

    Always returns the ``store``-side (proxy-wrapped) view — reads work
    identically to the fresh MetricOutput (`.metadata.get(...)` reads the
    same dict); writes to ``.metadata`` would raise TypeError, which is
    the intended guardrail (metric values are immutable after computation).

    ``key`` — not ``name`` — because primitives may take their own
    ``name=`` kwarg (e.g. ``ic_trend(series, name="caar_trend")``) that
    would collide with a positional alias.
    """
    if key not in store:
        _stash(store, fn(*args, **kwargs))
    return store[key]


def _run_profile_and_attach(profile_cls: type, artifacts: "Any") -> "Any":
    """Build a Profile via ``from_artifacts`` and attach its metric_outputs.

    ``from_artifacts`` is a pure function returning ``(profile,
    metric_outputs_dict)``. Both ``_evaluate_one`` and
    ``Factor.evaluate`` need the same two-step dance: call it, then
    write the resulting dict back onto ``artifacts.metric_outputs`` so
    subsequent standalone calls hit cache. Centralizing here keeps the
    purity contract visible while removing the duplication.
    """
    profile, metric_outputs = profile_cls.from_artifacts(artifacts)
    artifacts.metric_outputs = metric_outputs
    return profile


def _diagnose(profile: object) -> list[Diagnostic]:
    """Run the rule list registered for ``profile``'s concrete type.

    Thin delegator so every Profile class shares the identical method
    body rather than duplicating the import + call four times.
    """
    # Lazy import avoids a circular: diagnostics imports profiles
    # which imports _base during package init.
    from factorlib.evaluation.diagnostics import diagnose_profile
    return diagnose_profile(profile)


def _verdict_from_p(p: PValue, threshold: float, n_periods: int) -> Verdict:
    """Binary PASS/FAILED using a t-distribution threshold.

    ``threshold`` is in t-stat units (familiar to quants via Harvey,
    Liu & Zhu 2016's t > 3.0 recommendation for single-factor tests
    under multi-testing pressure — default 2.0 here is the classical
    single-test 95% boundary). All four canonical p-values are
    derived from t-distributions (IC / CAAR / FM lambda / TS beta),
    so the verdict threshold is translated through the *same* t CDF
    at ``n_periods - 1`` degrees of freedom (implemented inside
    ``_p_value_from_t``), not the Z approximation — the Z form would
    *under*-reject for small n because t tails are fatter.

    Two-sided: ``_p_value_from_t`` is symmetric in the sign of the
    input, so ``verdict(-t)`` and ``verdict(+t)`` return the same
    result. A caller who accidentally passes a negative threshold
    gets the well-defined answer they meant, not a crash.

    Degenerate case: when ``n_periods < 2`` the t-distribution has
    df ≤ 0 and is undefined; ``_p_value_from_t`` returns 1.0 as a
    fallback, which would cause ``p <= 1.0`` to always pass. We
    short-circuit to FAILED here so underpowered or empty inputs
    never masquerade as significant. Callers see the corroborating
    ``data.insufficient`` diagnose rule for context.

    ``verdict()`` is still a heuristic (one factor at a time); rigorous
    inference across a batch goes through
    ``ProfileSet.multiple_testing_correct`` (BHY).
    """
    if n_periods < 2:
        return "FAILED"
    # Lazy import to keep base module import light.
    from factorlib._stats import _p_value_from_t
    p_threshold = _p_value_from_t(threshold, n_periods)
    return "PASS" if p <= p_threshold else "FAILED"


def _verdict_with_warnings(profile: "FactorProfile", threshold: float) -> Verdict:
    """Upgrade PASS → PASS_WITH_WARNINGS when a warn-severity diagnostic
    names a whitelisted alternative p_source the user has not adopted.

    FAILED is never softened: a factor the canonical test rejects stays
    rejected regardless of warnings.
    """
    base = _verdict_from_p(profile.canonical_p, threshold, profile.n_periods)
    if base != "PASS":
        return base
    canonical = profile.CANONICAL_P_FIELD
    whitelist = profile.P_VALUE_FIELDS
    for d in profile.diagnose():
        if d.severity != "warn":
            continue
        rec = d.recommended_p_source
        if rec is None or rec == canonical:
            continue
        if rec not in whitelist:
            raise ValueError(
                f"Diagnostic {d.code!r} recommends p_source={rec!r} which "
                f"is not in {type(profile).__name__}.P_VALUE_FIELDS="
                f"{sorted(whitelist)}. Fix the Rule definition."
            )
        from factorlib._logging import get_evaluation_logger
        get_evaluation_logger().warning(
            "verdict=PASS_WITH_WARNINGS: factor=%s canonical=%s "
            "diagnostic=%s recommended_p_source=%s",
            getattr(profile, "factor_name", "?"), canonical, d.code, rec,
        )
        return "PASS_WITH_WARNINGS"
    return "PASS"


@runtime_checkable
class FactorProfile(Protocol):
    """Structural interface for all typed factor profiles.

    Concrete profiles are frozen slotted dataclasses that match this shape;
    they do not inherit from the Protocol (avoids dataclass/ABC metaclass
    fights). Registration is done by the ``@register_profile`` decorator.

    Required class-level metadata:
        CANONICAL_P_FIELD: name of the p-value field used for BHY and the
            default ``verdict()`` decision. Must be one of P_VALUE_FIELDS.
        P_VALUE_FIELDS: frozenset of field names that hold genuine p-values
            from the *same test family* across a batch of factors (IC
            across all CS factors, CAAR across all event factors, ...).
            Whitelisted by ``ProfileSet.multiple_testing_correct`` so BHY
            sees a coherent hypothesis family. The whitelist is NOT a
            within-factor list of interchangeable p-values: feeding BHY a
            mix of ``ic_p`` for some factors and ``spread_p`` for others
            would violate the same-test-family assumption and under-state
            FDR.

    Required members:
        canonical_p: property returning the canonical test p-value (single
            source of truth for BHY inputs).
        verdict(threshold): PASS / PASS_WITH_WARNINGS / FAILED on
            canonical p plus diagnose() hints. Raises ``ValueError`` if a
            registered ``Rule.recommended_p_source`` points outside
            ``P_VALUE_FIELDS`` — a developer error in rule authoring, not
            a user-data pathology.
        diagnose(): contextual hints as ``list[Diagnostic]``.
        from_artifacts(artifacts): classmethod constructor from Artifacts.
            Pure function: returns ``(profile, metric_outputs_dict)``
            without mutating its input, so subclasses can be reasoned
            about in isolation.
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
    def from_artifacts(
        cls, artifacts: "Artifacts",
    ) -> tuple[Self, dict[str, MetricOutput]]: ...


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
