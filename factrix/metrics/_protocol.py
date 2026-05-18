"""Dispatch-role markers for the ``run_metrics`` dispatcher.

Two roles are recognised by :mod:`factrix._run_metrics`:

- **batch primitive** — accepts ``factor_cols: list[str]`` and returns
  ``dict[str, ResultT]``. The dispatcher hands the whole factor batch
  to the function in one call so cross-factor compute (single polars
  query plan, shared IC stage-1) can be amortised.
- **IC stage-1 consumer** — accepts a single per-factor IC frame
  ``(date, ic, tie_ratio)`` produced once by :func:`compute_ic` and
  shared across every consumer in the batch.

Both roles were previously detected implicitly — batch primitive by
signature inspection (``"factor_cols" in inspect.signature(fn).parameters``),
IC consumer by membership in a hand-maintained frozenset
(``_IC_CONSUMERS = {"ic", "ic_newey_west", "ic_ir"}``). Either path
drifted silently: renaming ``factor_cols`` to ``factor_columns``
demoted a batch primitive to a per-factor loop with no warning;
adding a fourth IC consumer required remembering to update the
central constant. Explicit markers make the role part of the
function's own metadata so the dispatcher routes correctly without
relying on naming conventions or central registries.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

_BATCH_ATTR = "_factrix_batch_primitive"
_IC_ATTR = "_factrix_ic_consumer"


def batch_primitive[F: Callable](fn: F) -> F:
    """Mark ``fn`` as a batch-native metric primitive.

    The dispatcher will call it once across the whole factor batch via
    ``fn(panel, factor_cols=cols, ...)`` and expect a ``dict`` keyed by
    factor name. Raises ``TypeError`` at decoration time if the
    function signature does not accept a ``factor_cols`` keyword — the
    marker and the signature must agree.
    """
    if "factor_cols" not in inspect.signature(fn).parameters:
        raise TypeError(
            f"@batch_primitive on {fn.__qualname__!r}: signature must accept "
            f"a `factor_cols` keyword (the dispatcher passes the whole batch "
            f"via `factor_cols=cols`)."
        )
    setattr(fn, _BATCH_ATTR, True)
    return fn


def ic_consumer[F: Callable](fn: F) -> F:
    """Mark ``fn`` as a consumer of the IC stage-1 frame.

    The dispatcher will compute ``compute_ic(panel, factor_cols=cols)``
    once per batch, then call ``fn(ic_by_factor[c], ...)`` per factor
    with the per-factor IC frame instead of the raw panel. Raises
    ``TypeError`` at decoration time if the function's first
    positional parameter is not named ``ic_df`` — the marker and the
    signature must agree on what the first argument represents (the
    IC frame, not a panel).
    """
    params = list(inspect.signature(fn).parameters.values())
    if not params or params[0].name != "ic_df":
        first = params[0].name if params else "<no params>"
        raise TypeError(
            f"@ic_consumer on {fn.__qualname__!r}: first positional parameter "
            f"must be named `ic_df` (the dispatcher injects the per-factor IC "
            f"frame as the first argument); got {first!r}."
        )
    setattr(fn, _IC_ATTR, True)
    return fn


def is_batch_primitive(fn: Callable) -> bool:
    """Return True iff ``fn`` is marked ``@batch_primitive``."""
    return bool(getattr(fn, _BATCH_ATTR, False))


def is_ic_consumer(fn: Callable) -> bool:
    """Return True iff ``fn`` is marked ``@ic_consumer``."""
    return bool(getattr(fn, _IC_ATTR, False))


__all__ = [
    "batch_primitive",
    "ic_consumer",
    "is_batch_primitive",
    "is_ic_consumer",
]
