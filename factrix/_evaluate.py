"""v0.5 ``_evaluate`` — config + panel → registry dispatch → ``FactorProfile``.

Implements the four-step routing flow of refactor_api.md §4.4.2:

1. derive ``mode`` from the panel (``N == 1`` → ``TIMESERIES``, else ``PANEL``)
2. if ``signal == SPARSE`` and ``mode == TIMESERIES`` → rewrite scope to
   ``_SCOPE_COLLAPSED`` (§5.4.1) and tag the result with
   ``InfoCode.SCOPE_AXIS_COLLAPSED``
3. assemble ``_DispatchKey`` and look up the registry; missing → raise
   ``ModeAxisError`` with the nearest legal fallback (§5.5 / §4.5 A4)
4. ``entry.procedure.compute(panel, config)`` → ``FactorProfile``

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
    _SCOPE_COLLAPSED,
    _dispatch_key_for,
)

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig
    from factrix._profile import FactorProfile


def _derive_mode(panel: Any) -> Mode:
    """Return ``TIMESERIES`` if the panel has a single asset, else ``PANEL``.

    Reads ``asset_id`` directly off the panel; callers are expected
    to have validated the schema against ``procedure.INPUT_SCHEMA``
    before reaching this point.
    """
    return Mode.TIMESERIES if panel["asset_id"].n_unique() <= 1 else Mode.PANEL


def _evaluate(
    panel: Any,
    config: AnalysisConfig,
    *,
    factor_col: str = "factor",
) -> FactorProfile:
    """Dispatch ``config + panel`` to the registered procedure.

    Mode is derived from ``panel`` (``N == 1`` → ``TIMESERIES``, else
    ``PANEL``). Sparse signals at ``N == 1`` collapse the scope axis
    so ``individual_sparse`` and ``common_sparse`` route to the same
    cell, tagged with ``InfoCode.SCOPE_AXIS_COLLAPSED`` on the
    returned profile.

    Args:
        panel: Canonical-column long panel (``date, asset_id, factor,
            forward_return``). Schema is validated downstream by the
            registered procedure.
        config: Validated ``AnalysisConfig`` produced by one of the
            four factory methods.
        factor_col: Name of the signal column on ``panel``. Default
            ``"factor"``; pass any other column name when the panel's
            signal is named differently, or when looping over multiple
            candidate signals on a wide panel. The column is renamed
            to ``"factor"`` internally before dispatch so procedures
            keep their canonical schema. Each call repays the per-date
            cross-section work; ``factrix.multi_factor.bhy`` controls
            FDR on the resulting profile list but does not share
            computation across signals.

    Returns:
        A ``FactorProfile`` populated by the procedure registered for
        the routed dispatch cell.

    Raises:
        ValueError: If ``factor_col`` is not present on ``panel``, or if
            ``factor_col != "factor"`` while ``panel`` already carries a
            different ``"factor"`` column (ambiguous which is the
            signal).
        ModeAxisError: If the routed cell has no registered procedure
            under the derived mode (e.g. ``(INDIVIDUAL, CONTINUOUS, *)``
            at ``N == 1``); the error carries a nearest-legal
            ``suggested_fix``.
    """
    if factor_col != "factor":
        if factor_col not in panel.columns:
            raise ValueError(
                f"factor_col={factor_col!r} not found in panel columns: "
                f"{list(panel.columns)}. Pass the actual signal column "
                f"name, or rename the column to 'factor' before calling."
            )
        if "factor" in panel.columns:
            raise ValueError(
                f"panel carries both 'factor' and {factor_col!r} columns; "
                f"ambiguous which is the signal under test. Drop the "
                f"unused column before calling evaluate."
            )
        panel = panel.rename({factor_col: "factor"})
    mode = _derive_mode(panel)
    key = _dispatch_key_for(config.scope, config.signal, config.metric, mode)
    extra_info: frozenset[InfoCode] = (
        frozenset({InfoCode.SCOPE_AXIS_COLLAPSED})
        if key.scope is _SCOPE_COLLAPSED
        else frozenset()
    )
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

    profile = entry.procedure.compute(panel, config)
    profile = dataclasses.replace(
        profile,
        factor_id=factor_col,
        info_notes=profile.info_notes | extra_info,
    )
    return profile
