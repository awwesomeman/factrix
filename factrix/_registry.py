"""v0.5 dispatch registry — SSOT for "which analysis cells exist" (§4.4 A1).

Every component that needs to know whether a given cell is legal
(``_validate_axis_compat``, ``describe_analysis_modes``,
``suggest_config``, BHY family-key partitioning) reverse-queries this
module rather than maintaining a parallel rule table. Adding a new
cell touches one ``register(...)`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from factrix._axis import FactorScope, Metric, Mode, Signal

if TYPE_CHECKING:
    from factrix._procedures import FactorProcedure


class _ScopeCollapsedSentinel:
    """Marker for the collapsed scope axis under Mode B sparse (§5.4.1).

    A single sentinel instance — not an enum value — keeps the
    ``FactorScope | _ScopeCollapsedSentinel`` union narrow and avoids
    polluting the user-facing ``FactorScope`` enum with an internal
    routing token.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "_SCOPE_COLLAPSED"


_SCOPE_COLLAPSED: _ScopeCollapsedSentinel = _ScopeCollapsedSentinel()


@dataclass(frozen=True, slots=True)
class _DispatchKey:
    """``(scope, signal, metric, mode)`` cell coordinate."""

    scope: FactorScope | _ScopeCollapsedSentinel
    signal: Signal
    metric: Metric | None
    mode: Mode


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    """Procedure + metadata for one cell.

    ``canonical_use_case`` and ``references`` feed
    ``describe_analysis_modes()`` directly — no parallel docs table.
    """

    key: _DispatchKey
    procedure: "FactorProcedure"
    canonical_use_case: str
    references: tuple[str, ...] = field(default_factory=tuple)


_DISPATCH_REGISTRY: dict[_DispatchKey, _RegistryEntry] = {}


def register(
    key: _DispatchKey,
    procedure: "FactorProcedure",
    *,
    use_case: str,
    refs: tuple[str, ...] = (),
) -> None:
    """Register a procedure under ``key``.

    Raises ``ValueError`` on duplicate keys — registry is append-only
    and one cell maps to exactly one procedure.
    """
    if key in _DISPATCH_REGISTRY:
        raise ValueError(f"DispatchKey already registered: {key}")
    _DISPATCH_REGISTRY[key] = _RegistryEntry(
        key=key,
        procedure=procedure,
        canonical_use_case=use_case,
        references=refs,
    )


def matches_user_axis(
    scope: FactorScope,
    signal: Signal,
    metric: Metric | None,
) -> bool:
    """Does any registry entry accept this user-facing triple?

    Used by ``_validate_axis_compat``. Sentinel-keyed entries match
    either ``FactorScope`` value because ``(*, SPARSE, N=1)`` collapses
    to the sentinel at evaluate-time (§5.4.1) but is constructible
    under either user-facing scope.
    """
    for entry in _DISPATCH_REGISTRY.values():
        if entry.key.signal != signal or entry.key.metric != metric:
            continue
        if entry.key.scope == scope or entry.key.scope is _SCOPE_COLLAPSED:
            return True
    return False


# Import-time bootstrap: ``_procedures`` calls ``register(...)`` at
# module bottom, populating ``_DISPATCH_REGISTRY`` before any first
# query lands. ``_procedures`` imports back only ``register`` and
# ``_DispatchKey`` from us, both defined above this line.
from factrix import _procedures as _procedures  # noqa: E402, F401
