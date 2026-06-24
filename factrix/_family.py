"""Shared family resolution layer for multiple-testing functions.

Every closed-form family function (``bhy`` / ``bhy_hierarchical`` /
``partial_conjunction``) runs through ``_resolve_family`` to turn a
list of :class:`~factrix._results.EvaluationResult` into flat
``_FamilyEntry`` records ready for the procedure-specific step-up math.

The invariants enforced here are the family-layer extension of the
anti-shopping defense: the hypothesis identity is
``(factor, *expand_over_values)``; ``expand_over`` names are read
from ``forward_periods`` (the lone non-context built-in slicing axis)
or from ``EvaluationResult.context``. The estimator-override hook is
gone — callers select inference at metric-construction time (e.g.
``ic(inference=fx.inference.NEWEY_WEST)``) and pick the result by
passing the corresponding ``metric`` label.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from factrix._errors import UserInputError

if TYPE_CHECKING:
    from factrix._results import EvaluationResult


_BUILTIN_EXPAND_OVER_FIELDS: frozenset[str] = frozenset({"forward_periods"})


@dataclass(frozen=True, slots=True)
class _FamilyEntry:
    """Flat record carrying one hypothesis through the family pipeline.

    Spans both stages: ``_partition`` emits entries with ``p_value=None``
    (identity resolved, p-value not yet attached); ``_attach_p_values``
    re-emits them per metric with ``p_value`` populated. Procedures read
    ``p_value`` only after the attach stage, where it is always non-None.

    Attributes:
        identifier: ``(factor, *expand_over_values)`` — the hypothesis
            key. With no ``expand_over``, collapses to ``(factor,)``.
        expand_over_values: ``tuple`` in caller-supplied key order;
            empty when ``expand_over`` is empty.
        result: Back-reference for survivor rendering; not read by the
            resolution layer itself.
        p_value: ``MetricResult.p_value`` for the resolved ``metric`` spec.
            ``None`` until ``_attach_p_values`` runs.
    """

    identifier: tuple[Any, ...]
    expand_over_values: tuple[Any, ...]
    result: EvaluationResult
    p_value: float | None = None


def _partition(
    results: Sequence[EvaluationResult],
    *,
    func_name: str,
    expand_over: Sequence[str] = (),
) -> list[_FamilyEntry]:
    """Validate ``expand_over`` keys and partition-key uniqueness.

    Pulled out of ``_resolve_family`` so multi-metric callers run
    the per-result walk once, then attach per-metric p-values via
    ``_attach_p_values``. Returns entries with ``p_value=None``.
    """
    keys = list(expand_over)
    for name in keys:
        if name == "factor":
            raise UserInputError(
                func_name=func_name,
                field="expand_over",
                value=name,
                expected="slicing axis, not the hypothesis identifier 'factor'",
                docs_path=f"api/{func_name}#expand_over",
            )

    _check_context_keys(results, keys=keys, func_name=func_name)

    entries: list[_FamilyEntry] = []
    seen: dict[tuple[Any, ...], int] = {}
    for idx, result in enumerate(results):
        values = _expand_over_values(result, keys=keys)
        identifier = (result.factor, *values)
        if identifier in seen:
            raise UserInputError(
                func_name=func_name,
                field="results",
                value=identifier,
                expected=(
                    "unique (factor, *expand_over_values) identifier across "
                    f"input; duplicate first seen at index {seen[identifier]}, "
                    f"again at {idx}. Either stamp a distinct factor column "
                    "per evaluation, or pass `expand_over=[<key>]` to declare "
                    "per-bucket families (e.g. `expand_over=('forward_periods',)` "
                    "for multi-horizon screening)"
                ),
                docs_path=f"api/{func_name}#partition-key",
            )
        seen[identifier] = idx
        entries.append(
            _FamilyEntry(
                identifier=identifier,
                expand_over_values=values,
                result=result,
            )
        )
    return entries


def _attach_p_values(
    partition: Sequence[_FamilyEntry],
    *,
    func_name: str,
    metric: str,
) -> list[_FamilyEntry]:
    """Resolve per-metric p-value for each partition entry."""
    if partition:
        try:
            all_none = all(p.result.metrics[metric].p_value is None for p in partition)
        except KeyError:
            all_none = False

        if all_none:
            raise UserInputError(
                func_name=func_name,
                field="metrics",
                value=metric,
                expected=(
                    f"target metric {metric!r} has no p-values (all results returned None). "
                    "Descriptive metrics without formal hypothesis tests cannot be used for FDR control"
                ),
                docs_path=f"api/{func_name}#metrics",
            )

    return [
        _FamilyEntry(
            identifier=p.identifier,
            expand_over_values=p.expand_over_values,
            result=p.result,
            p_value=_resolve_p_value(p.result, metric=metric, func_name=func_name),
        )
        for p in partition
    ]


def _resolve_family(
    results: Sequence[EvaluationResult],
    *,
    func_name: str,
    metric: str,
    expand_over: Sequence[str] = (),
) -> list[_FamilyEntry]:
    """Single-metric convenience wrapper around ``_partition`` +
    ``_attach_p_values``.

    Steps (raise on failure, in order):

    1. ``expand_over`` names must exist either as a built-in slicing
       field (``forward_periods``) or as a key in every result's
       ``context``. ``factor`` is rejected (it is the hypothesis
       identifier, not a slicing axis).
    2. The partition key ``(factor, *expand_over_values)`` must be
       unique across the input.
    3. Each result must produce the ``metric``'s p-value;
       the p must be present and non-NaN.
    """
    partition = _partition(results, func_name=func_name, expand_over=expand_over)
    return _attach_p_values(partition, func_name=func_name, metric=metric)


def _check_context_keys(
    results: Sequence[EvaluationResult],
    *,
    keys: list[str],
    func_name: str,
) -> None:
    """Raise once listing every ``(factor, missing_key)`` across the input.

    Built-in slicing fields (``forward_periods``) are read off the result
    and never missing; only ``context`` keys are checked. A single failed
    result no longer short-circuits the whole screen — integrating results
    from several sources surfaces all context gaps in one pass, matching
    the aggregate-then-report idiom of ``evaluate``'s strict / column
    guards. fail-loud is preserved: any gap still raises (silently
    dropping a result would alter family composition and the FDR
    denominator).
    """
    context_keys = [k for k in keys if k not in _BUILTIN_EXPAND_OVER_FIELDS]
    if not context_keys:
        return
    missing = [
        (result.factor, name)
        for result in results
        for name in context_keys
        if name not in result.context
    ]
    if not missing:
        return
    missing_keys = sorted({k for _, k in missing})
    available = sorted({k for result in results for k in result.context})
    detail = "; ".join(f"factor={factor!r} missing {key!r}" for factor, key in missing)
    raise UserInputError(
        func_name=func_name,
        field="expand_over",
        value=missing_keys,
        expected=(
            f"expand_over key(s) {missing_keys!r} present in every result's "
            f"context. Missing — {detail}. Stamp the key on every "
            f"EvaluationResult.context, or drop it from expand_over. "
            f"Context keys seen across the input: "
            f"{available or ['<none>']!r}"
        ),
        docs_path=f"api/{func_name}#expand_over",
    )


def _expand_over_values(
    result: EvaluationResult,
    *,
    keys: list[str],
) -> tuple[Any, ...]:
    """Read the ``expand_over`` value tuple off one result.

    Presence of every context key is guaranteed by a prior
    ``_check_context_keys`` sweep, so this reads without re-validating.
    """
    return tuple(
        getattr(result, name)
        if name in _BUILTIN_EXPAND_OVER_FIELDS
        else result.context[name]
        for name in keys
    )


def _resolve_p_value(
    result: EvaluationResult,
    *,
    metric: str,
    func_name: str,
) -> float:
    try:
        out = result.metrics[metric]
    except KeyError:
        raise UserInputError(
            func_name=func_name,
            field="metrics",
            value=metric,
            expected=(
                f"every result to carry the metric "
                f"{metric!r}; missing on factor={result.factor!r}"
            ),
            candidates=sorted(result.metrics),
            docs_path=f"api/{func_name}#metrics",
        ) from None

    p = out.p_value
    if p is None:
        raise UserInputError(
            func_name=func_name,
            field="metrics",
            value=metric,
            expected=(
                f"a p-value (`.p_value`) populated on every result; "
                f"factor={result.factor!r} has no p-value for "
                f"metric {metric!r}"
            ),
            docs_path=f"api/{func_name}#metrics",
        )

    p_float = float(p)
    if math.isnan(p_float):
        raise UserInputError(
            func_name=func_name,
            field="metrics",
            value=metric,
            expected=(
                f"finite p-value on every result; factor={result.factor!r} "
                f"has NaN p for metric {metric!r} — drop the result "
                "or pick a different metric"
            ),
            docs_path=f"api/{func_name}#metrics",
        )
    return p_float
