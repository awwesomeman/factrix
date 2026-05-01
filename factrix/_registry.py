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
    # A-4 from review: sentinel-keyed entries route both INDIVIDUAL and
    # COMMON user axes to the same procedure, so a non-None metric on
    # such an entry would silently let an illegal (e.g. SPARSE+IC)
    # triple pass `matches_user_axis`. Lock it out at registration.
    if key.scope is _SCOPE_COLLAPSED and key.metric is not None:
        raise ValueError(
            "_SCOPE_COLLAPSED entries must have metric=None — "
            "non-None metric on a collapsed-scope cell would let "
            "matches_user_axis admit illegal user triples.",
        )
    _DISPATCH_REGISTRY[key] = _RegistryEntry(
        key=key,
        procedure=procedure,
        canonical_use_case=use_case,
        references=refs,
    )


def _route_scope(
    scope: FactorScope, signal: Signal, mode: Mode,
) -> FactorScope | _ScopeCollapsedSentinel:
    """Apply the §5.4.1 scope-collapse rule for sparse Mode B routing.

    Single source of truth for "when does the scope axis collapse to
    the sentinel?" — `_evaluate`, `_describe`, and `_multi_factor.bhy`
    all call this so the rule cannot drift across sites.
    """
    if signal is Signal.SPARSE and mode is Mode.TIMESERIES:
        return _SCOPE_COLLAPSED
    return scope


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


# IMPORT-ORDER LOAD-BEARING — do not move this import.
#
# ``_procedures`` calls ``register(...)`` at module bottom, populating
# ``_DISPATCH_REGISTRY`` before any first query lands. ``_procedures``
# imports from us only the names defined above this line (``register``,
# ``_DispatchKey``, ``_route_scope``, ``_SCOPE_COLLAPSED``,
# ``_ScopeCollapsedSentinel``). Adding a top-level ``_procedures``
# usage of any helper defined below would create a circular-import
# deadlock that will not surface until import time of a downstream
# package — fail loudly here instead with the post-import assert below.
from factrix import _procedures as _procedures  # noqa: E402, F401

# Post-bootstrap invariant: 7 cells (5 PANEL + 2 TIMESERIES). Catches
# accidental deletion of a register(...) call or a circular-import
# regression that prevents _procedures from running to completion.
_EXPECTED_REGISTRY_SIZE = 7
assert len(_DISPATCH_REGISTRY) == _EXPECTED_REGISTRY_SIZE, (
    f"registry bootstrap incomplete: {len(_DISPATCH_REGISTRY)} of "
    f"{_EXPECTED_REGISTRY_SIZE} cells registered"
)
