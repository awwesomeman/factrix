"""``compare`` — multi-metric leaderboard for :class:`EvaluationResult` lists.

Pure projection: stacks per-factor per-metric values into a wide
``pl.DataFrame`` for sorting and visual diff. No metric is recomputed.

Heterogeneous context keys follow ``pl.concat(how="diagonal")`` —
union + null-fill — so a result missing ``region`` surfaces as a
``null`` cell rather than a silent drop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl

from factrix._errors import UserInputError
from factrix._multi_factor import _require_non_empty_results, _validate_metric_list
from factrix._results import EvaluationResult, _float_or_none


def compare(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    sort_by: str | None = None,
    descending: bool = True,
) -> pl.DataFrame:
    """Render a wide leaderboard ``pl.DataFrame`` for multiple metrics.

    One row per :class:`EvaluationResult`; two columns per metric —
    ``<metric_name>`` (``MetricResult.value``) and ``<metric_name>_p``
    (``MetricResult.p_value`` when present, else ``null``).

    Args:
        results: Non-empty list of :class:`EvaluationResult`. Each must
            carry every spec in ``metrics``.
        metrics: ``list[str]`` — list-only canonical form
            (element type strictly :class:`str`). Single-metric
            callers still pass a one-element list; mirrors the
            ``metrics`` contract on ``fx.multi_factor.bhy`` so the
            whole multi-factor API surface uses one shape.
        sort_by: Optional :class:`str` (must be in ``metrics``)
            naming the sort key. ``None`` keeps input order and omits
            the ``rank`` column.
        descending: Sort direction applied to ``sort_by``. Default
            ``True`` (higher-is-better, the common case for ``ic`` /
            ``alpha`` / information ratio). Pass ``descending=False``
            for lower-is-better metrics such as ``turnover`` or any
            cost / drag metric. :class:`str` deliberately does
            not carry a ``higher_is_better`` flag — encoding sort
            direction in the type system bakes in a default that
            silently mis-ranks when wrong. No-op when ``sort_by`` is
            ``None``.

    Returns:
        ``pl.DataFrame`` with column order ``factor``,
        ``forward_periods``, context keys (union across results,
        first-seen order), then ``<m_name>`` / ``<m_name>_p`` pairs
        in ``metrics`` order, then ``rank`` when ``sort_by`` is set.

    Raises:
        UserInputError: Empty ``results``; ``metrics`` not a non-empty
            ``list[str]``; any metric absent from any result's
            outputs; ``sort_by`` not present in ``metrics``.

    Examples:
        Multi-metric wide leaderboard sorted on IC:

        >>> ic = fx.metrics.spec_by_name()["ic"]  # doctest: +SKIP
        >>> sharpe = fx.metrics.spec_by_name()["sharpe"]  # doctest: +SKIP
        >>> board = fx.compare(  # doctest: +SKIP
        ...     results, metrics=[ic, sharpe], sort_by=ic
        ... )

        Lower-is-better metric (``descending=False``):

        >>> turnover = fx.metrics.spec_by_name()["turnover"]  # doctest: +SKIP
        >>> board = fx.compare(  # doctest: +SKIP
        ...     results, metrics=[turnover], sort_by=turnover, descending=False
        ... )
    """
    metric_list = _validate_metric_list(metrics, func_name="compare", field="metrics")
    _require_non_empty_results(results, func_name="compare")
    if sort_by is not None and sort_by not in {m for m in metric_list}:
        raise UserInputError(
            func_name="compare",
            field="sort_by",
            value=sort_by,
            expected="str that appears in `metrics`",
            candidates=[m for m in metric_list],
            docs_path="api/compare#sort_by",
        )

    context_keys = _ordered_keys(r.context for r in results)
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "factor": r.factor,
            "forward_periods": r.forward_periods,
        }
        for k in context_keys:
            row[k] = r.context.get(k)
        for spec in metric_list:
            if spec not in r.metrics:
                raise UserInputError(
                    func_name="compare",
                    field="metrics",
                    value=spec,
                    expected=(
                        f"every result to carry metric {spec!r}; "
                        f"missing on factor={r.factor!r}"
                    ),
                    candidates=sorted(r.metrics),
                    docs_path="api/compare#metrics",
                )
            out = r.metrics[spec]
            row[spec] = _float_or_none(out.value)
            row[f"{spec}_p_value"] = _float_or_none(out.p_value)
        rows.append(row)

    df = pl.DataFrame(rows)
    if sort_by is not None:
        df = df.sort(sort_by, descending=descending, nulls_last=True)
        df = df.with_columns(
            pl.int_range(1, df.height + 1, dtype=pl.Int64).alias("rank")
        )
    return df


def _ordered_keys(maps: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for m in maps:
        for k in m:
            seen.setdefault(k, None)
    return list(seen)
