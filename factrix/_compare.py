"""``compare`` — leaderboard renderer for :class:`EvaluationResult` lists.

Pure projection: stacks per-factor metric values into a wide
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
from factrix._metric_index import MetricSpec
from factrix._results import EvaluationResult, _float_or_none


def compare(
    results: list[EvaluationResult],
    *,
    metric: MetricSpec,
    descending: bool = True,
) -> pl.DataFrame:
    """Render a leaderboard ``pl.DataFrame`` for one metric across results.

    Args:
        results: Non-empty list of :class:`EvaluationResult`. Each must
            carry ``metric`` in its outputs.
        metric: :class:`MetricSpec` whose ``MetricOutput.value`` (and
            ``metadata['p_value']`` when present) is projected per row.
            ``str`` / ``Callable`` not accepted — pick the spec from
            ``fx.metrics.spec_by_name()``.
        descending: Sort direction on ``value``. Default ``True``
            (higher-is-better, the common case for ``ic`` / ``alpha`` /
            information ratio). Pass ``descending=False`` for
            lower-is-better metrics such as ``turnover`` or any
            cost / drag metric. ``MetricSpec`` deliberately does not
            carry a ``higher_is_better`` flag — encoding sort direction
            in the type system bakes in a default that silently
            mis-ranks when wrong.

    Returns:
        ``pl.DataFrame`` with columns laid out as ``factor``,
        ``forward_periods``, context keys (union across results,
        first-seen order), ``value``, ``p``, ``rank``.

    Raises:
        UserInputError: Empty ``results``; ``metric`` not a
            :class:`MetricSpec`; metric absent from any result's outputs.

    Examples:
        Higher-is-better (default):

        >>> board = fx.compare(  # doctest: +SKIP
        ...     results, metric=fx.metrics.spec_by_name()["ic"]
        ... )

        Lower-is-better (``descending=False``):

        >>> board = fx.compare(  # doctest: +SKIP
        ...     results,
        ...     metric=fx.metrics.spec_by_name()["turnover"],
        ...     descending=False,
        ... )
    """
    if not isinstance(metric, MetricSpec):
        raise UserInputError(
            func_name="compare",
            field="metric",
            value=type(metric).__name__,
            expected=(
                "MetricSpec instance (str / Callable not accepted — pick "
                "the spec from fx.metrics.spec_by_name() or the metric "
                "module's __metric_specs__ tuple)"
            ),
            docs_path="api/compare/",
        )
    if not results:
        raise UserInputError(
            func_name="compare",
            field="results",
            value=results,
            expected="non-empty list[EvaluationResult]",
            docs_path="api/compare/",
        )

    context_keys = _ordered_keys(r.context for r in results)
    rows: list[dict[str, Any]] = []
    for r in results:
        if metric.name not in r.metrics.outputs:
            raise UserInputError(
                func_name="compare",
                field="metric",
                value=metric.name,
                expected=(
                    f"every result to carry metric {metric.name!r}; "
                    f"missing on factor={r.factor!r}"
                ),
                candidates=sorted(r.metrics.outputs),
                docs_path="api/compare/",
            )
        out = r.metrics.outputs[metric.name]
        row: dict[str, Any] = {
            "factor": r.factor,
            "forward_periods": r.forward_periods,
        }
        for k in context_keys:
            row[k] = r.context.get(k)
        row["value"] = _float_or_none(out.value)
        row["p"] = _float_or_none(out.metadata.get("p_value"))
        rows.append(row)

    df = pl.DataFrame(rows)
    df = df.sort("value", descending=descending, nulls_last=True)
    rank_col = pl.int_range(1, df.height + 1, dtype=pl.Int64).alias("rank")
    df = df.with_columns(rank_col)
    return df


def _ordered_keys(maps: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for m in maps:
        for k in m:
            seen.setdefault(k, None)
    return list(seen)
