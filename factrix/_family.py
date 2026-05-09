"""Shared family resolution layer for multiple-testing verbs (#161).

Every closed-form family verb (``bhy`` / ``bhy_hierarchical`` /
``partial_conjunction`` / ``bonferroni`` / ``holm``) and the
resampling-based ``romano_wolf`` runs through ``_resolve_family`` to
turn a list of :class:`~factrix._profile.FactorProfile` into flat
``FamilyEntry`` records ready for the procedure-specific step-up math.

The four invariants enforced here are the family-layer extension of the
identity / context anti-shopping defense from #160: ``identity``
dimensions (``factor_id`` / ``forward_periods``) name *the hypothesis*,
``context`` dimensions name *the slicing condition*, and ``expand_over``
must stay in the latter — otherwise users would unwittingly let horizon
or factor name participate in family partitioning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from factrix._codes import StatCode
from factrix._errors import UserInputError

if TYPE_CHECKING:
    from factrix._profile import FactorProfile


_IDENTITY_FIELDS: frozenset[str] = frozenset({"factor_id", "forward_periods"})


@dataclass(frozen=True, slots=True)
class FamilyEntry:
    """Flat record fed to family-verb procedures after invariant checks.

    Attributes:
        identity: ``(factor_id, forward_periods)`` from
            :attr:`FactorProfile.identity`.
        expand_over_values: ``tuple(profile.context[k] for k in expand_over)``
            in the order the caller passed ``expand_over``; empty tuple
            when ``expand_over`` is None / empty.
        p_value: Resolved per the ``p_stat`` selection rule —
            ``profile.primary_p`` when ``p_stat is None``, else
            ``profile.stats[p_stat]``.
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
    verb: str,
    expand_over: Sequence[str] | None = None,
    p_stat: StatCode | None = None,
) -> list[FamilyEntry]:
    """Validate four invariants and return flat ``FamilyEntry`` records.

    Steps (raise on failure, in order):

    1. ``expand_over`` names must exist in every profile's ``context``
       and must not collide with identity dimensions
       (``factor_id`` / ``forward_periods``).
    2. The partition key
       ``identity + tuple(profile.context[k] for k in expand_over)``
       must be unique across the input.
    3. When ``p_stat`` is supplied, every profile must populate it in
       ``stats``.
    4. Resolve ``p_value`` per profile: ``primary_p`` when
       ``p_stat is None``, else ``profile.stats[p_stat]``.

    Args:
        profiles: Input profiles. ``__hash__`` is disabled on
            ``FactorProfile``; dedup uses the partition-key tuple, never
            profile hashing.
        verb: Calling verb name for error rendering (e.g. ``"bhy"``).
        expand_over: Optional context keys to include in the partition
            key. ``None`` and ``[]`` are equivalent.
        p_stat: Optional alternate p-value selector. ``None`` falls back
            to ``primary_p``.

    Returns:
        One ``FamilyEntry`` per input profile, in input order.

    Raises:
        UserInputError: On any of the three named-set / availability
            failures (unknown ``expand_over`` / hits identity / duplicate
            partition key / missing ``p_stat``).
    """
    if p_stat is not None and not p_stat.is_p_value:
        raise UserInputError(
            verb=verb,
            field="p_stat",
            value=p_stat.name,
            expected=(
                f"p-value StatCode (is_p_value=True); {p_stat.name} is "
                "not a probability and family step-up math would be "
                "incoherent on it"
            ),
            docs_path=f"api/{verb}#p_stat",
        )

    keys = list(expand_over) if expand_over else []
    for name in keys:
        if name in _IDENTITY_FIELDS:
            raise UserInputError(
                verb=verb,
                field="expand_over",
                value=name,
                expected="context key, not an identity dimension (see #160)",
                docs_path=f"api/{verb}#expand_over",
            )

    entries: list[FamilyEntry] = []
    seen: dict[tuple[Any, ...], int] = {}

    for idx, profile in enumerate(profiles):
        values = _expand_over_values(profile, keys=keys, verb=verb)
        partition_key = (*profile.identity, *values)
        if partition_key in seen:
            raise UserInputError(
                verb=verb,
                field="profiles",
                value=partition_key,
                expected=(
                    "unique partition key across input; duplicate first "
                    f"seen at index {seen[partition_key]}, again at {idx}. "
                    "Set distinct factor_id per profile, or pass "
                    "expand_over=[<context key>] to declare per-bucket "
                    "families"
                ),
                docs_path=f"api/{verb}#partition-key",
            )
        seen[partition_key] = idx

        entries.append(
            FamilyEntry(
                identity=profile.identity,
                expand_over_values=values,
                p_value=_resolve_p_value(profile, p_stat=p_stat, verb=verb),
                profile=profile,
            )
        )

    return entries


def _expand_over_values(
    profile: FactorProfile,
    *,
    keys: list[str],
    verb: str,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for name in keys:
        if name not in profile.context:
            raise UserInputError(
                verb=verb,
                field="expand_over",
                value=name,
                candidates=sorted(profile.context)
                or ["<no context keys on this profile>"],
                docs_path=f"api/{verb}#expand_over",
            )
        values.append(profile.context[name])
    return tuple(values)


def _resolve_p_value(
    profile: FactorProfile,
    *,
    p_stat: StatCode | None,
    verb: str,
) -> float:
    if p_stat is None:
        return profile.primary_p
    try:
        return profile.stats[p_stat]
    except KeyError:
        raise UserInputError(
            verb=verb,
            field="p_stat",
            value=p_stat.name,
            candidates=sorted(s.name for s in profile.stats),
            docs_path=f"api/{verb}#p_stat",
        ) from None
