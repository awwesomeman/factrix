"""Shared ``chunk_size`` auto-sizing helper for the ``*_chunked`` family.

Used by :func:`factrix._run_metrics.run_metrics_chunked` and
:func:`factrix._evaluate.evaluate_chunked`. The two callers want the
same RAM-budget heuristic but raise their own ``UserInputError`` with
caller-specific ``func_name`` / ``docs_path`` so the missing-psutil
hint points the user back at the API they called.
"""

from __future__ import annotations

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
