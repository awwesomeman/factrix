"""Shared helpers for the ``*_chunked`` API family.

Used by :func:`factrix._run_metrics.run_metrics_chunked` and
:func:`factrix._evaluate.evaluate_chunked`. The two callers share
identical validation + chunk-loop boilerplate; the only per-caller
differences are the underlying ``run_metrics`` / ``_evaluate`` call
and the ``func_name`` / ``docs_path`` stamped into any raised
``UserInputError``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import batched

import polars as pl

from factrix._errors import UserInputError

# Per-factor in-memory amplification factor used by the chunk-size
# heuristic. Each chunk's peak RSS is dominated by panel materialise +
# polars query intermediates (``with_columns(rank_exprs)`` adds one
# ``_rank__<f>`` column per factor); ``4 ×`` an 8-byte-per-row column
# slice empirically tracks observed M-ic / S2 peak_rss within ±20%
# across the small / large presets — close enough for a budget knob.
_AUTO_CHUNK_OVERHEAD_FACTOR = 4

# Divisor applied to ``psutil.virtual_memory().available`` to derive
# the per-chunk peak budget. Dividing by 4 (i.e. targeting ~25%) leaves
# slack for OS, BLAS arenas, and the caller's downstream sink without
# forcing every batch back to chunk_size=1 on tight machines.
_AUTO_CHUNK_RSS_DIVISOR = 4


def auto_chunk_size(
    n_rows: int,
    n_factors: int,
    *,
    func_name: str,
    docs_path: str,
) -> int:
    """Pick chunk size targeting ~25% of available RAM as per-chunk peak.

    ``psutil`` is an optional dependency (``factrix[bench]`` extras);
    when absent, callers must pass ``chunk_size`` explicitly — the
    function raises so the user reaches for the right knob rather
    than silently getting a degenerate default. ``func_name`` /
    ``docs_path`` are stamped into the raised ``UserInputError`` so
    the hint points the user back at the API they called.
    """
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError as exc:
        raise UserInputError(
            func_name=func_name,
            field="chunk_size",
            value=None,
            expected=(
                "an explicit positive integer when psutil is not installed; "
                "auto-sizing requires `pip install psutil` (or "
                "`pip install 'factrix[bench]'`)"
            ),
            docs_path=docs_path,
        ) from exc

    available = psutil.virtual_memory().available
    per_factor_bytes = max(n_rows * 8 * _AUTO_CHUNK_OVERHEAD_FACTOR, 1)
    budget = max(available // _AUTO_CHUNK_RSS_DIVISOR, per_factor_bytes)
    return max(1, min(n_factors, budget // per_factor_bytes))


def _raise_factor_cols_error(
    *, value: object, expected: str, func_name: str, docs_path: str
) -> None:
    raise UserInputError(
        func_name=func_name,
        field="factor_cols",
        value=value,
        expected=expected,
        docs_path=docs_path,
    )


def chunk_panel(
    panel: pl.DataFrame | pl.LazyFrame,
    factor_cols: Sequence[str],
    *,
    chunk_size: int | None,
    base_cols: Sequence[str],
    func_name: str,
    docs_path: str,
) -> Iterator[tuple[pl.DataFrame, list[str]]]:
    """Validate + chunk a ``(panel, factor_cols)`` pair for the ``*_chunked`` APIs.

    Yields ``(sub_panel, chunk_factor_cols)`` per iteration. All
    eager checks (``factor_cols`` empty / duplicate, ``chunk_size``
    sign, ``panel`` type, ``base_cols + factor_cols`` schema) run
    before the first yield so the caller-facing ``UserInputError``
    lands at call time rather than mid-iteration.

    ``func_name`` / ``docs_path`` are stamped into every raised
    ``UserInputError`` so the hint points back at the API the user
    actually called.
    """
    cols = list(factor_cols)
    if not cols:
        _raise_factor_cols_error(
            value=cols,
            expected="a non-empty list of factor column names",
            func_name=func_name,
            docs_path=docs_path,
        )
    if len(set(cols)) != len(cols):
        _raise_factor_cols_error(
            value=cols,
            expected="factor_cols with no duplicates",
            func_name=func_name,
            docs_path=docs_path,
        )

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")

    if not isinstance(panel, pl.DataFrame | pl.LazyFrame):
        raise TypeError(
            f"panel must be pl.DataFrame or pl.LazyFrame; got {type(panel).__name__}"
        )

    schema_cols = (
        set(panel.collect_schema().names())
        if isinstance(panel, pl.LazyFrame)
        else set(panel.columns)
    )
    base = list(base_cols)
    missing = (set(base) | set(cols)) - schema_cols
    if missing:
        _raise_factor_cols_error(
            value=cols,
            expected=(
                f"panel with all of base_cols + factor_cols present; "
                f"missing {sorted(missing)!r}"
            ),
            func_name=func_name,
            docs_path=docs_path,
        )

    if chunk_size is None:
        n_rows = (
            panel.select(pl.len()).collect().item()
            if isinstance(panel, pl.LazyFrame)
            else panel.height
        )
        cs = auto_chunk_size(
            n_rows, len(cols), func_name=func_name, docs_path=docs_path
        )
    else:
        cs = chunk_size

    for chunk_tuple in batched(cols, cs):
        chunk = list(chunk_tuple)
        projection = [*base, *chunk]
        sub_panel = (
            panel.select(projection).collect()
            if isinstance(panel, pl.LazyFrame)
            else panel.select(projection)
        )
        yield sub_panel, chunk
