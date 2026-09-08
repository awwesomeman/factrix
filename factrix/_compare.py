"""``compare`` — multi-metric leaderboard for :class:`EvaluationResult` lists.

Pure projection: stacks per-factor per-metric values into a wide
``pl.DataFrame`` for sorting and visual diff. No metric is recomputed.

Heterogeneous ``params`` keys follow ``pl.concat(how="diagonal")`` —
union + null-fill — so a result missing ``region`` surfaces as a
``null`` cell rather than a silent drop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import polars as pl

from factrix._errors import UserInputError
from factrix._multi_factor import _require_non_empty_results, _validate_metric_list
from factrix._results import EvaluationResult, _float_or_none

RankMethod = Literal["min", "dense", "ordinal"]

_RANK_METHODS: tuple[RankMethod, ...] = ("min", "dense", "ordinal")

# Sort direction for metric columns whose direction follows from the
# metric's own definition: a return-like or evidence-like quantity that
# is better when larger, or a cost driver that is better when smaller.
# Signed metrics (``predictive_beta``, ``fm_beta``, ``spanning_alpha``,
# ``caar``, ``ic_trend`` …) are deliberately absent — "largest positive
# value" is not the same question as "strongest effect", so those keys
# require an explicit ``descending``.
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "ic",
        "ic_ir",
        "quantile_spread",
        "quantile_spread_vw",
        "common_quantile_spread",
        "net_spread",
        "breakeven_cost",
    }
)
_LOWER_IS_BETTER: frozenset[str] = frozenset({"rank_turnover", "notional_turnover"})

# Identity columns that also serve as deterministic tiebreakers.
_IDENTITY_COLS: tuple[str, ...] = ("factor", "forward_periods")

# Hidden ordering key: the sort column with NaN folded into null so that
# "missing last" holds in both directions (polars orders NaN as the
# largest float, which would otherwise put it first under descending).
_SORT_KEY = "__factrix_sort_key"


def compare(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    sort_by: str | None = None,
    descending: bool | None = None,
    rank_method: RankMethod = "min",
) -> pl.DataFrame:
    """Render a wide leaderboard ``pl.DataFrame`` for multiple metrics.

    One row per :class:`EvaluationResult`; two columns per requested metric
    label — ``<metric_label>`` (``MetricResult.value``) and
    ``<metric_label>_p_value`` (``MetricResult.p_value`` when present,
    else ``null``). The label is the key in ``EvaluationResult.metrics``,
    usually the user-supplied key from ``evaluate(metrics={...})``.

    Args:
        results: Non-empty list of :class:`EvaluationResult`. Each must
            carry every spec in ``metrics``.
        metrics: ``list[str]`` — list-only canonical form
            (element type strictly :class:`str`). Single-metric
            callers still pass a one-element list; mirrors the
            ``metrics`` contract on ``fx.multi_factor.bhy`` so the
            whole multi-factor API surface uses one shape.
        sort_by: Optional :class:`str` naming any output column produced
            before ranking: identity columns, params keys, metric value
            columns, or ``<metric_label>_p_value`` columns. ``None`` keeps
            input order and omits the ``rank`` column.
        descending: Sort direction applied to ``sort_by``. ``None`` (the
            default) resolves the direction from the column itself, under
            the rule below; pass ``True`` / ``False`` to state it
            outright, which always wins over the rule. There is no global
            direction default — a lower-is-better key must never inherit
            a higher-is-better sort. No-op when ``sort_by`` is ``None``.
        rank_method: How equal ``sort_by`` values are numbered in the
            ``rank`` column. ``"min"`` (default) gives tied rows the same,
            best rank and leaves a gap (``1, 1, 3``); ``"dense"`` gives the
            same rank without a gap (``1, 1, 2``); ``"ordinal"`` numbers
            every row ``1..N``, ties broken by the row order described
            below. Names and semantics are Polars'
            ``Expr.rank`` methods.

    Direction rule (``descending=None``):

    - a ``<metric_label>_p_value`` column sorts **ascending** — a smaller
      p-value is stronger evidence;
    - ``factor``, ``forward_periods`` and params keys sort **ascending**;
      they label a row rather than score it, so natural order applies;
    - a metric value column sorts by the direction its metric is defined
      with: ``ic``, ``ic_ir``, ``quantile_spread``, ``quantile_spread_vw``,
      ``common_quantile_spread``, ``net_spread`` and ``breakeven_cost``
      descending; ``rank_turnover`` and ``notional_turnover`` ascending;
    - any other metric label — a custom evaluation label such as
      ``ic_nw``, or a signed metric such as ``predictive_beta`` — raises
      :class:`UserInputError`. Pass ``descending`` explicitly there.

    Row order and ties: rows are sorted on ``sort_by``, then on ``factor``
    and ``forward_periods`` ascending, then on params and output columns.
    Columns Polars cannot sort natively use a canonical, type-tagged string
    only as an internal ordering key; their returned values are untouched.
    That key orders lexicographically on the string, not on the value's own
    semantics — ``[10, 2]`` sorts before ``[2, 1]`` — so read such an order
    as stable, not as meaningful.
    The output therefore does not depend on the order of ``results``. Rows
    equal on every one of those columns are indistinguishable and keep input
    order among themselves.

    Missing values: a ``null`` or ``NaN`` ``sort_by`` value sorts **last**
    in both directions and carries a ``null`` rank under every
    ``rank_method`` — a row with no value has no place in the ranking.

    Returns:
        ``pl.DataFrame`` with column order ``factor``,
        ``forward_periods``, params keys (union across results,
        first-seen order), then ``<metric_label>`` /
        ``<metric_label>_p_value`` pairs
        in ``metrics`` order, then ``rank`` when ``sort_by`` is set.

    Raises:
        UserInputError: Empty ``results``; ``metrics`` not a non-empty
            ``list[str]``; any metric absent from any result's
            outputs; ``rank_method`` not one of ``min`` / ``dense`` /
            ``ordinal``; ``sort_by`` not present in the output columns;
            ``sort_by`` naming a metric column of unknown direction while
            ``descending`` is ``None``.

    Examples:
        Alpha / information-ratio style metric — higher is better, and
        the direction rule knows it:

        >>> board = fx.compare(  # doctest: +SKIP
        ...     results, metrics=["ic", "ic_ir"], sort_by="ic_ir"
        ... )

        Turnover — lower is better, resolved the same way:

        >>> board = fx.compare(  # doctest: +SKIP
        ...     results, metrics=["rank_turnover"], sort_by="rank_turnover"
        ... )

        Significance screen on a p-value column, ties sharing one rank:

        >>> board = fx.compare(  # doctest: +SKIP
        ...     results, metrics=["ic"], sort_by="ic_p_value", rank_method="min"
        ... )

        A custom label carries no direction, so state one:

        >>> board = fx.compare(  # doctest: +SKIP
        ...     results, metrics=["ic_nw"], sort_by="ic_nw", descending=True
        ... )
    """
    metric_list = _validate_metric_list(metrics, func_name="compare", field="metrics")
    _require_non_empty_results(results, func_name="compare")
    if rank_method not in _RANK_METHODS:
        raise UserInputError(
            func_name="compare",
            field="rank_method",
            value=rank_method,
            candidates=_RANK_METHODS,
            docs_path="api/compare#parameter-details",
        )
    param_keys = _ordered_keys(r.params for r in results)
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "factor": r.factor,
            "forward_periods": r.forward_periods,
        }
        for k in param_keys:
            row[k] = r.params.get(k)
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
                    docs_path="api/compare#parameter-details",
                )
            out = r.metrics[spec]
            row[spec] = _float_or_none(out.value)
            row[f"{spec}_p_value"] = _float_or_none(out.p_value)
        rows.append(row)

    data = pl.DataFrame(rows)
    if sort_by is None:
        return data

    sort_candidates = list(data.columns)
    if sort_by not in sort_candidates:
        raise UserInputError(
            func_name="compare",
            field="sort_by",
            value=sort_by,
            expected="one of the columns produced by compare()",
            candidates=sort_candidates,
            docs_path="api/compare#parameter-details",
        )
    resolved = _resolve_descending(sort_by, descending, metric_list=metric_list)
    return _rank(data, sort_by=sort_by, descending=resolved, rank_method=rank_method)


def _resolve_descending(
    sort_by: str, descending: bool | None, *, metric_list: list[str]
) -> bool:
    """Return the sort direction actually applied to ``sort_by``.

    An explicit ``descending`` is used as given. Otherwise the direction
    comes from the column's kind — p-value column, identity/params
    column, or a metric label of known direction — and an unknown metric
    label is an error rather than a silent default.
    """
    if descending is not None:
        return descending
    if sort_by in metric_list:
        if sort_by in _HIGHER_IS_BETTER:
            return True
        if sort_by in _LOWER_IS_BETTER:
            return False
        raise UserInputError(
            func_name="compare",
            field="descending",
            value=descending,
            expected=(
                f"an explicit True / False for sort_by={sort_by!r}: compare() "
                "resolves a direction only for p-value columns, identity and "
                "params columns, and metric labels of known direction "
                f"(higher is better: {sorted(_HIGHER_IS_BETTER)}; lower is "
                f"better: {sorted(_LOWER_IS_BETTER)})"
            ),
            docs_path="api/compare#parameter-details",
        )
    # p-value columns, identity columns and params keys all sort ascending.
    return False


def _is_natively_sortable(dtype: pl.DataType) -> bool:
    """Whether Polars can use ``dtype`` directly as a leaderboard key."""
    return dtype.is_numeric() or dtype.is_temporal() or dtype in (pl.Boolean, pl.String)


def _canonical_sort_value(value: object) -> str | None:
    """Return a stable, type-preserving ordering token for a nested value."""
    if value is None:
        return None
    if isinstance(value, pl.Series):
        value = value.to_list()
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, Mapping):
        mapping_items = sorted(
            (
                _canonical_sort_value(key) or "builtins.NoneType:None",
                _canonical_sort_value(item) or "builtins.NoneType:None",
            )
            for key, item in value.items()
        )
        return f"{type_name}:{mapping_items!r}"
    if isinstance(value, list | tuple):
        sequence_items = [
            _canonical_sort_value(item) or "builtins.NoneType:None" for item in value
        ]
        return f"{type_name}:{sequence_items!r}"
    if isinstance(value, set | frozenset):
        set_items = sorted(
            _canonical_sort_value(item) or "builtins.NoneType:None" for item in value
        )
        return f"{type_name}:{set_items!r}"
    return f"{type_name}:{value!r}"


def _deterministic_tiebreaks(
    data: pl.DataFrame, sort_by: str
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Attach canonical keys where needed and return all secondary keys."""
    keys = [c for c in _IDENTITY_COLS if c != sort_by]
    hidden: list[str] = []
    for name, dtype in data.schema.items():
        if name in _IDENTITY_COLS or name == sort_by:
            continue
        if _is_natively_sortable(dtype):
            keys.append(name)
            continue
        hidden_name = _unused_column_name(
            [*data.columns, *hidden], f"__factrix_tiebreak_{len(hidden)}"
        )
        data = data.with_columns(
            pl.Series(
                hidden_name,
                [_canonical_sort_value(value) for value in data[name].to_list()],
                dtype=pl.String,
            )
        )
        keys.append(hidden_name)
        hidden.append(hidden_name)
    return data, keys, hidden


def _rank(
    data: pl.DataFrame, *, sort_by: str, descending: bool, rank_method: RankMethod
) -> pl.DataFrame:
    """Sort ``data`` on ``sort_by`` and attach the ``rank`` column."""
    data, tiebreaks, hidden_tiebreaks = _deterministic_tiebreaks(data, sort_by)
    sort_key_name = _unused_column_name(data.columns, _SORT_KEY)
    dtype = data.schema[sort_by]
    key = pl.col(sort_by)
    if dtype in (pl.Float32, pl.Float64):
        # Fold NaN into null so "missing last" holds in both directions.
        key = pl.when(key.is_nan()).then(None).otherwise(key)
        data = data.with_columns(key.alias(sort_key_name))
    elif _is_natively_sortable(dtype):
        data = data.with_columns(key.alias(sort_key_name))
    else:
        data = data.with_columns(
            pl.Series(
                sort_key_name,
                [_canonical_sort_value(value) for value in data[sort_by].to_list()],
                dtype=pl.String,
            )
        )
    data = data.sort(
        [sort_key_name, *tiebreaks],
        descending=[descending, *[False] * len(tiebreaks)],
        nulls_last=True,
    )
    if rank_method == "ordinal":
        ranks = pl.int_range(1, data.height + 1, dtype=pl.Int64)
    else:
        ranks = pl.col(sort_key_name).rank(method=rank_method, descending=descending)
    data = data.with_columns(
        pl.when(pl.col(sort_key_name).is_null())
        .then(None)
        .otherwise(ranks)
        .cast(pl.Int64)
        .alias("rank")
    )
    return data.drop(sort_key_name, *hidden_tiebreaks)


def _unused_column_name(columns: Iterable[str], preferred: str) -> str:
    """Return an internal column name that cannot shadow caller data."""
    occupied = set(columns)
    candidate = preferred
    while candidate in occupied:
        candidate = f"_{candidate}"
    return candidate


def _ordered_keys(maps: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for m in maps:
        for k in m:
            seen.setdefault(k, None)
    return list(seen)
