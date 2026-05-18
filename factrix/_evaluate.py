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
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

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


_DEFAULT_BASE_COLS: tuple[str, ...] = ("date", "asset_id", "forward_return")


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

    Cross-factor compute sharing rides on each procedure's
    ``compute_batch`` hook (#426): the default
    :class:`factrix._procedures._PerFactorBatchMixin` keeps the
    per-factor loop behaviour, and the IC-cell procedure overrides it
    to call ``compute_ic(panel, factor_cols=cols)`` once and stitch
    per-factor HAC inference. Adding shared stage-1 to other cells is
    a per-procedure decision; this dispatch layer is agnostic.

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

    profiles = entry.procedure.compute_batch(panel, config, cols)
    if extra_info:
        profiles = {
            col: dataclasses.replace(p, info_notes=p.info_notes | extra_info)
            for col, p in profiles.items()
        }
    return profiles


def evaluate_chunked(
    panel: pl.DataFrame | pl.LazyFrame,
    config: AnalysisConfig,
    *,
    factor_cols: Sequence[str],
    chunk_size: int | None = None,
    base_cols: Sequence[str] = _DEFAULT_BASE_COLS,
) -> Iterator[dict[str, FactorProfile]]:
    """Yield :func:`evaluate` output one chunk of factors at a time.

    Splits ``factor_cols`` into chunks, narrows ``panel`` to
    ``base_cols + chunk`` per iteration, calls :func:`evaluate`, and
    yields each chunk's ``dict[factor_id, FactorProfile]``. Peak RSS is
    bounded by the chunk size rather than ``len(factor_cols)`` — the
    evaluate-side mirror of :func:`factrix.run_metrics_chunked`.

    Within a chunk the cell's procedure runs its normal
    ``compute_batch`` path (#426), so any cross-factor sharing on that
    cell (currently: IC stage-1 reuse) applies inside the chunk and
    is recomputed across chunks. Very small ``chunk_size`` (e.g. 1)
    therefore pays the per-chunk overhead without the share — pick
    ``chunk_size`` to fit your RAM budget, not to micromanage the
    share / no-share trade.

    Args:
        panel: ``pl.DataFrame`` or ``pl.LazyFrame``. When passed a
            ``LazyFrame``, the height is sampled via
            ``select(pl.len()).collect()`` (one row) and each chunk
            does a fresh ``panel.select([...]).collect()`` so
            projection pushdown applies per chunk — only the chunk's
            factor columns get scanned from the source.
        config: Same as :func:`evaluate`.
        factor_cols: Factor columns to chunk over. Must be non-empty
            and contain no duplicates. ``base_cols`` plus every factor
            in this list must exist on ``panel`` — schema is checked
            eagerly before the first chunk yields.
        chunk_size: Number of factors per chunk. ``None`` (default)
            picks a chunk size targeting ~25% of available RAM, which
            requires ``psutil`` (optional dependency — install via
            ``pip install psutil`` or ``pip install 'factrix[bench]'``).
            Pass an explicit value to override. An explicit
            ``chunk_size`` larger than ``len(factor_cols)`` is accepted
            and degenerates to a single chunk.
        base_cols: Panel columns required by every chunk regardless of
            which factor subset is active. Default
            ``("date", "asset_id", "forward_return")`` matches
            :func:`evaluate`'s base contract. Override when an extra
            column is required (e.g. a weight column for a
            future weighted procedure).

    Yields:
        ``dict[factor_id, FactorProfile]`` — same shape as
        :func:`evaluate`, scoped to one chunk's factors. Iterate the
        generator to consume chunks sequentially; each chunk's
        profiles can be written to a sink and released before the
        next chunk is produced.

    Raises:
        UserInputError: ``factor_cols`` empty / contains duplicates,
            or ``panel`` missing a ``base_cols`` / factor column,
            or ``chunk_size=None`` and ``psutil`` is not installed.
        ValueError: ``chunk_size`` non-positive.
        TypeError: ``panel`` not ``pl.DataFrame`` or ``pl.LazyFrame``.

    Examples:
        Stream 1000 factors through a parquet sink, 100 per chunk:

        >>> import factrix as fx                                   # doctest: +SKIP
        >>> for profiles in fx.evaluate_chunked(                   # doctest: +SKIP
        ...     panel, cfg, factor_cols=cols, chunk_size=100,
        ... ):
        ...     for fid, profile in profiles.items():
        ...         sink.write(fid, profile)

        Auto-sized chunks (default):

        >>> for profiles in fx.evaluate_chunked(                   # doctest: +SKIP
        ...     panel, cfg, factor_cols=cols,
        ... ):
        ...     ...
    """
    from factrix._chunk_size import chunk_panel

    for sub_panel, chunk in chunk_panel(
        panel,
        factor_cols,
        chunk_size=chunk_size,
        base_cols=base_cols,
        func_name="evaluate_chunked",
        docs_path="api/evaluate_chunked#chunk_size",
    ):
        yield _evaluate(sub_panel, config, factor_cols=chunk)
