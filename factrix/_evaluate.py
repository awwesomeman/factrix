"""v0.5 ``_evaluate`` — config + raw → registry dispatch → ``FactorProfile``.

Implements the four-step routing flow of refactor_api.md §4.4.2:

1. derive ``mode`` from raw data (``N == 1`` → ``TIMESERIES``, else ``PANEL``)
2. if ``signal == SPARSE`` and ``mode == TIMESERIES`` → rewrite scope to
   ``_SCOPE_COLLAPSED`` (§5.4.1) and tag the result with
   ``InfoCode.SCOPE_AXIS_COLLAPSED``
3. assemble ``_DispatchKey`` and look up the registry; missing → raise
   ``ModeAxisError`` with the nearest legal fallback (§5.5 / §4.5 A4)
4. ``entry.procedure.compute(raw, config)`` → ``FactorProfile``

Underscore-prefixed: this is the private dispatch entry. The public
``factrix.evaluate`` binding owns the user-facing surface and delegates
here once it adopts the v0.5 contract.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from factrix._analysis_config import _FALLBACK_MAP
from factrix._axis import Mode
from factrix._codes import InfoCode
from factrix._errors import ModeAxisError
from factrix._registry import (
    _DISPATCH_REGISTRY,
    _DispatchKey,
    _route_scope,
)

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig
    from factrix._profile import FactorProfile


def _derive_mode(raw: Any) -> Mode:
    """Return ``TIMESERIES`` if the panel has a single asset, else ``PANEL``.

    Reads ``asset_id`` directly off the raw frame; callers are expected
    to have validated the schema against ``procedure.INPUT_SCHEMA``
    before reaching this point.
    """
    return Mode.TIMESERIES if raw["asset_id"].n_unique() <= 1 else Mode.PANEL


def _evaluate(raw: Any, config: "AnalysisConfig") -> "FactorProfile":
    """Dispatch ``config + raw`` to the registered procedure.

    Mode is derived from ``raw`` (``N == 1`` → ``TIMESERIES``, else
    ``PANEL``). Sparse signals at ``N == 1`` collapse the scope axis
    so ``individual_sparse`` and ``common_sparse`` route to the same
    cell, tagged with ``InfoCode.SCOPE_AXIS_COLLAPSED`` on the
    returned profile.

    Args:
        raw: Canonical-column panel (``date, asset_id, factor,
            forward_return``). Schema is validated downstream by the
            registered procedure.
        config: Validated ``AnalysisConfig`` produced by one of the
            four factory methods.

    Returns:
        A ``FactorProfile`` populated by the procedure registered for
        the routed dispatch cell.

    Raises:
        ModeAxisError: If the routed cell has no registered procedure
            under the derived mode (e.g. ``(INDIVIDUAL, CONTINUOUS, *)``
            at ``N == 1``); the error carries a nearest-legal
            ``suggested_fix``.
    """
    mode = _derive_mode(raw)
    routed_scope = _route_scope(config.scope, config.signal, mode)
    extra_info: frozenset[InfoCode] = (
        frozenset({InfoCode.SCOPE_AXIS_COLLAPSED})
        if routed_scope is not config.scope
        else frozenset()
    )
    key = _DispatchKey(routed_scope, config.signal, config.metric, mode)
    entry = _DISPATCH_REGISTRY.get(key)
    if entry is None:
        fallback = _FALLBACK_MAP.get((config.scope, config.signal, mode))
        suggested = fallback() if fallback is not None else None
        suffix = f" Suggested fix: {suggested!r}" if suggested else ""
        raise ModeAxisError(
            f"({config.scope.value}, {config.signal.value}, "
            f"{config.metric.value if config.metric else None}) is "
            f"undefined under mode={mode.value}.{suffix}",
            suggested_fix=suggested,
        )

    profile = entry.procedure.compute(raw, config)
    if extra_info:
        profile = dataclasses.replace(
            profile,
            info_notes=profile.info_notes | extra_info,
        )
    return profile
