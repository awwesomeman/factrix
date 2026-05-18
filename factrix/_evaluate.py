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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from factrix._analysis_config import _FALLBACK_MAP
from factrix._axis import Mode
from factrix._codes import InfoCode
from factrix._errors import ModeAxisError, UserInputError
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


def _raise_factor_cols_error(*, value: object, expected: str) -> None:
    raise UserInputError(
        func_name="evaluate",
        field="factor_cols",
        value=value,
        expected=expected,
        docs_path="api/evaluate#factor_cols",
    )


def _validate_factor_cols(factor_cols: Sequence[str], panel: Any) -> list[str]:
    """Eager non-empty / no-dup / all-present check on ``factor_cols``.

    Sibling of ``_run_metrics._validate_factor_cols`` — kept separate
    because this variant also validates column presence on ``panel``
    so ``evaluate`` fails fast at the API boundary; ``run_metrics``
    defers schema validation to per-metric dispatch where each
    primitive's call surfaces a column-specific error.
    """
    cols = list(factor_cols)
    if not cols:
        _raise_factor_cols_error(
            value=cols, expected="a non-empty list of factor column names"
        )
    if len(set(cols)) != len(cols):
        _raise_factor_cols_error(value=cols, expected="factor_cols with no duplicates")
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        _raise_factor_cols_error(
            value=missing,
            expected=(
                f"every name in factor_cols to exist on panel; "
                f"got columns {list(panel.columns)!r}"
            ),
        )
    return cols


def _evaluate(
    panel: Any,
    config: AnalysisConfig,
    *,
    factor_cols: Sequence[str] = ("factor",),
) -> dict[str, FactorProfile]:
    """Dispatch ``config + panel`` to the registered procedure for each factor.

    All factors in ``factor_cols`` share the same dispatch cell and
    mode **by design**: ``config`` uniquely determines the cell
    (``scope × signal × metric``); the panel's
    ``asset_id.n_unique()`` uniquely determines the mode
    (``N == 1`` → ``TIMESERIES``, else ``PANEL``). Batching factors
    only makes sense when the resulting profiles are comparable, and
    comparability requires identical cell × mode — so this layer
    pins both at the batch level instead of deriving per-factor.
    Sparse signals at ``N == 1`` collapse the scope axis so
    ``individual_sparse`` and ``common_sparse`` route to the same
    cell, tagged with ``InfoCode.SCOPE_AXIS_COLLAPSED`` on each
    returned profile.

    Cross-factor compute sharing (IC stage-1 reuse, batch primitives)
    is **not** done at this layer yet — each factor is dispatched
    independently via a thin column projection. Adding shared stage-1
    is tracked as a follow-up using the same protocol-class registry
    that ``run_metrics`` adopted in #418.

    Args:
        panel: Canonical-column long panel (``date, asset_id, *factor_cols,
            forward_return``). Schema is validated downstream by the
            registered procedure on the per-factor projection.
        config: Validated ``AnalysisConfig`` produced by one of the
            four factory methods.
        factor_cols: Names of the signal columns on ``panel`` to
            evaluate. Each column is projected and renamed to
            ``"factor"`` before being dispatched to the procedure, so
            procedures keep their canonical schema. The return dict is
            keyed by the original ``factor_cols`` name; each profile's
            ``factor_id`` is also stamped to match. Default
            ``("factor",)`` keeps the single-factor case ergonomic
            (caller indexes via ``["factor"]``).

    Returns:
        ``dict[factor_name, FactorProfile]`` — one profile per input
        column, keyed by the original ``factor_cols`` name.

    Raises:
        UserInputError: ``factor_cols`` empty, contains duplicates, or
            references a column not present on ``panel``.
        ModeAxisError: If the routed cell has no registered procedure
            under the derived mode (e.g. ``(INDIVIDUAL, CONTINUOUS, *)``
            at ``N == 1``); the error carries a nearest-legal
            ``suggested_fix``.
    """
    cols = _validate_factor_cols(factor_cols, panel)

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

    other_factor_cols = frozenset(cols)
    profiles: dict[str, FactorProfile] = {}
    for col in cols:
        sub_panel = _project_factor(panel, col, other_factor_cols)
        profile = entry.procedure.compute(sub_panel, config)
        profiles[col] = dataclasses.replace(
            profile,
            factor_id=col,
            info_notes=profile.info_notes | extra_info,
        )
    return profiles


def _project_factor(panel: Any, col: str, all_factor_cols: frozenset[str]) -> Any:
    """Project ``panel`` to canonical schema with ``col`` aliased to ``"factor"``.

    Drops the sibling factor columns in ``all_factor_cols - {col}``
    and an incumbent ``"factor"`` column (when ``col != "factor"``)
    so the procedure receives a panel whose only signal column is
    named ``"factor"``. Every other column passes through unchanged so
    optional schema columns the procedure may consume (e.g. ``price``
    for event-window metrics) stay available.
    """
    import polars as pl

    drop = set(all_factor_cols - {col})
    if col != "factor":
        drop.add("factor")
    selects = []
    for c in panel.columns:
        if c in drop:
            continue
        if c == col and col != "factor":
            selects.append(pl.col(col).alias("factor"))
        else:
            selects.append(pl.col(c))
    return panel.select(selects)
