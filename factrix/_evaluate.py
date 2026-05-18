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

    Thin eager wrapper — ``dict(evaluate_iter(panel, config,
    factor_cols=factor_cols))`` — so eager and streaming paths share
    one dispatcher (mirrors the ``run_metrics`` / ``run_metrics_iter``
    pattern from #423). See :func:`evaluate_iter` for the full
    contract; this docstring only covers what differs.

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
    return dict(evaluate_iter(panel, config, factor_cols=factor_cols))


def evaluate_iter(
    panel: Any,
    config: AnalysisConfig,
    *,
    factor_cols: Sequence[str] = ("factor",),
) -> Iterator[tuple[str, FactorProfile]]:
    """Stream ``(factor_id, FactorProfile)`` pairs as each factor completes.

    Per-factor streaming sibling of :func:`factrix.evaluate`. Cross-
    factor work that the dispatcher would otherwise share is still
    amortised once: the registered procedure's
    :meth:`~factrix._procedures.FactorProcedure.bind_batch` runs any
    eager stage-1 (currently the IC cell's ``compute_ic`` across the
    batch) **before the first yield** and returns a per-factor
    closure. The yield loop then calls the closure once per factor —
    so the first ``next(gen)`` lands after one factor's inference,
    not all ``N`` factors', letting callers write to a sink / update
    a progress bar / break early without paying the full-batch
    latency.

    :func:`factrix.evaluate` is a one-line wrapper —
    ``dict(evaluate_iter(...))`` — so eager and streaming paths share
    the same dispatcher and the same cell × mode invariants. The same
    cell + mode pinning applies (``config`` locks the cell;
    ``panel["asset_id"].n_unique()`` locks the mode; ``SPARSE`` at
    ``N == 1`` collapses to a single procedure and each yielded
    profile is tagged with ``InfoCode.SCOPE_AXIS_COLLAPSED``).

    Args:
        panel: Same contract as :func:`factrix.evaluate`.
        config: Same as :func:`factrix.evaluate`.
        factor_cols: Same as :func:`factrix.evaluate`. Yield order
            matches this sequence.

    Yields:
        ``(factor_id, FactorProfile)`` pairs in ``factor_cols`` order.
        Each profile is identical to the matching entry of
        ``evaluate(...)`` — same ``factor_id`` / ``stats`` /
        ``info_notes`` content.

    Raises:
        UserInputError: Same conditions as :func:`factrix.evaluate`.
            Raised eagerly (validation runs before the generator
            emits anything).
        ModeAxisError: Same conditions as :func:`factrix.evaluate`.
            Raised eagerly.
        InsufficientSampleError: Propagated from the per-factor
            closure — surfaces mid-stream after earlier factors have
            already been emitted (the trade-off any streaming
            iterator carries; callers consuming the generator into a
            sink will see a partial stream).

    Examples:
        Stream 1000 factors to a parquet sink without holding the
        full result dict in memory:

        >>> import factrix as fx                                   # doctest: +SKIP
        >>> for fid, profile in fx.evaluate_iter(                  # doctest: +SKIP
        ...     panel, cfg, factor_cols=cols,
        ... ):
        ...     sink.write(fid, profile)
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

    return _evaluate_iter_inner(
        procedure=entry.procedure,
        panel=panel,
        config=config,
        cols=cols,
        extra_info=extra_info,
    )


def _evaluate_iter_inner(
    *,
    procedure: Any,
    panel: Any,
    config: AnalysisConfig,
    cols: list[str],
    extra_info: frozenset[InfoCode],
) -> Iterator[tuple[str, FactorProfile]]:
    """Yield ``(factor_id, FactorProfile)`` after cross-factor bind.

    Split out from :func:`evaluate_iter` so input validation and
    dispatch resolution raise eagerly (before the first ``next()``)
    while the per-factor yield loop is a true generator — yields land
    between factors, not all at the end.
    """
    getter = procedure.bind_batch(panel, config, cols)
    for col in cols:
        profile = getter(col)
        if extra_info:
            profile = dataclasses.replace(
                profile, info_notes=profile.info_notes | extra_info
            )
        yield col, profile


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
