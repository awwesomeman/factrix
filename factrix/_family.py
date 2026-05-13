"""Shared family resolution layer for multiple-testing verbs (#161, #170).

Every closed-form family verb (``bhy`` / ``bhy_hierarchical`` /
``partial_conjunction`` / ``bonferroni`` / ``holm``) and the
resampling-based ``romano_wolf`` runs through ``_resolve_family`` to
turn a list of :class:`~factrix._profile.FactorProfile` into flat
``_FamilyEntry`` records ready for the procedure-specific step-up math.

The four invariants enforced here are the family-layer extension of the
identity / context anti-shopping defense from #160: ``identity``
dimensions (``factor_id`` / ``forward_periods``) name *the hypothesis*,
``context`` dimensions name *the slicing condition*, and ``expand_over``
must stay in the latter — otherwise users would unwittingly let horizon
or factor name participate in family partitioning.

The ``estimator=`` override (#170) replaces the v0.10 ``p_stat=StatCode``
placeholder. An :class:`~factrix.stats.Estimator` instance names *the
inference method* whose p-value should drive the step-up math; the
instance reports its applicability per cell and dispatches to the
appropriate ``StatCode`` key in ``profile.stats``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from factrix._errors import UserInputError

if TYPE_CHECKING:
    from factrix._profile import FactorProfile
    from factrix.stats import Estimator


_IDENTITY_FIELDS: frozenset[str] = frozenset({"factor_id", "forward_periods"})

# Docs fragment anchor for ``estimator=`` error messages. Single source
# of truth so the family-resolution failure paths and any future docs
# generation agree on the URL shape.
_ESTIMATOR_DOCS_ANCHOR = "estimator"


@dataclass(frozen=True, slots=True)
class _FamilyEntry:
    """Flat record fed to family-verb procedures after invariant checks.

    Attributes:
        identity: ``(factor_id, forward_periods)`` from
            :attr:`FactorProfile.identity`.
        expand_over_values: ``tuple(profile.context[k] for k in expand_over)``
            in the order the caller passed ``expand_over``; empty tuple
            when ``expand_over`` is None / empty.
        p_value: Resolved per the ``estimator`` selection rule —
            ``profile.primary_p`` when ``estimator is None``, else the
            value at ``profile.stats[estimator.emits_for(...)]``.
        profile: Back-reference for survivor-renderer use; never read by
            the resolution layer itself.
    """

    identity: tuple[str, int]
    expand_over_values: tuple[Any, ...]
    p_value: float
    profile: FactorProfile


def _resolve_family(
    profiles: Sequence[FactorProfile],
    *,
    func_name: str,
    expand_over: Sequence[str] | None = None,
    estimator: Estimator | None = None,
) -> list[_FamilyEntry]:
    """Validate four invariants and return flat ``_FamilyEntry`` records.

    Steps (raise on failure, in order):

    1. ``expand_over`` names must exist in every profile's ``context``
       and must not collide with identity dimensions
       (``factor_id`` / ``forward_periods``).
    2. The partition key
       ``identity + tuple(profile.context[k] for k in expand_over)``
       must be unique across the input.
    3. When ``estimator`` is supplied, every profile's cell must be in
       its applicability set and the dispatched ``StatCode`` must be
       populated in ``profile.stats``.
    4. Resolve ``p_value`` per profile: ``primary_p`` when
       ``estimator is None``, else
       ``profile.stats[estimator.emits_for(scope, signal, metric)]``.

    Args:
        profiles: Input profiles. ``__hash__`` is disabled on
            ``FactorProfile``; dedup uses the partition-key tuple, never
            profile hashing.
        func_name: Calling function name for error rendering (e.g. ``"bhy"``).
        expand_over: Optional context keys to include in the partition
            key. ``None`` and ``[]`` are equivalent.
        estimator: Optional inference-method override. ``None`` falls
            back to ``primary_p``.

    Returns:
        One ``_FamilyEntry`` per input profile, in input order.

    Raises:
        UserInputError: On any of the named-set / availability /
            applicability failures.
    """
    keys = list(expand_over) if expand_over else []
    for name in keys:
        if name in _IDENTITY_FIELDS:
            raise UserInputError(
                func_name=func_name,
                field="expand_over",
                value=name,
                expected="context key, not an identity dimension (see #160)",
                docs_path=f"api/{func_name}#expand_over",
            )

    entries: list[_FamilyEntry] = []
    seen: dict[tuple[Any, ...], int] = {}

    for idx, profile in enumerate(profiles):
        values = _expand_over_values(profile, keys=keys, func_name=func_name)
        partition_key = (*profile.identity, *values)
        if partition_key in seen:
            raise UserInputError(
                func_name=func_name,
                field="profiles",
                value=partition_key,
                expected=(
                    "unique partition key across input; duplicate first "
                    f"seen at index {seen[partition_key]}, again at {idx}. "
                    "Stamp distinct factor_id per profile via "
                    "`evaluate(..., factor_col=<name>)` (canonical) or "
                    "`dataclasses.replace(profile, factor_id=<name>)` "
                    "(escape hatch when the column cannot be renamed); "
                    "or pass `expand_over=[<context key>]` to declare "
                    "per-bucket families"
                ),
                docs_path=f"api/{func_name}#partition-key",
            )
        seen[partition_key] = idx

        entries.append(
            _FamilyEntry(
                identity=profile.identity,
                expand_over_values=values,
                p_value=_resolve_p_value(
                    profile, estimator=estimator, func_name=func_name
                ),
                profile=profile,
            )
        )

    return entries


def _expand_over_values(
    profile: FactorProfile,
    *,
    keys: list[str],
    func_name: str,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for name in keys:
        if name not in profile.context:
            raise UserInputError(
                func_name=func_name,
                field="expand_over",
                value=name,
                candidates=sorted(profile.context)
                or ["<no context keys on this profile>"],
                docs_path=f"api/{func_name}#expand_over",
            )
        values.append(profile.context[name])
    return tuple(values)


def _resolve_p_value(
    profile: FactorProfile,
    *,
    estimator: Estimator | None,
    func_name: str,
) -> float:
    if estimator is None:
        return profile.primary_p

    cfg = profile.config
    if not estimator.applicable_to(cfg.scope, cfg.signal):
        raise UserInputError(
            func_name=func_name,
            field="estimator",
            value=estimator.name,
            expected=(
                f"estimator applicable to (scope={cfg.scope.value}, "
                f"signal={cfg.signal.value}) cell"
            ),
            docs_path=f"api/{func_name}#{_ESTIMATOR_DOCS_ANCHOR}",
        )

    code = estimator.emits_for(cfg.scope, cfg.signal, cfg.metric)
    try:
        return profile.stats[code]
    except KeyError:
        raise UserInputError(
            func_name=func_name,
            field="estimator",
            value=estimator.name,
            expected=(
                f"profile.stats to populate {code.name} when {estimator.name} "
                "is supplied; populated keys listed below"
            ),
            candidates=sorted(s.name for s in profile.stats),
            docs_path=f"api/{func_name}#{_ESTIMATOR_DOCS_ANCHOR}",
        ) from None
