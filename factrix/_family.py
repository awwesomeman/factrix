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
gone — the new ``fx.estimators`` namespace bakes the inference choice
into each MetricSpec name (e.g. ``ic_newey_west``), so callers pick
the estimator by passing the corresponding ``primary`` MetricSpec.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from factrix._errors import UserInputError
from factrix._metric_index import MetricSpec

if TYPE_CHECKING:
    from factrix._results import EvaluationResult


_BUILTIN_EXPAND_OVER_FIELDS: frozenset[str] = frozenset({"forward_periods"})


@dataclass(frozen=True, slots=True)
class _PartitionEntry:
    """Result-side partition record — identity + result, no p-value yet."""

    identifier: tuple[Any, ...]
    expand_over_values: tuple[Any, ...]
    result: EvaluationResult


@dataclass(frozen=True, slots=True)
class _FamilyEntry:
    """Flat record fed to family-function procedures after invariant checks.

    Attributes:
        identifier: ``(factor, *expand_over_values)`` — the hypothesis
            key. With no ``expand_over``, collapses to ``(factor,)``.
        expand_over_values: ``tuple`` in caller-supplied key order;
            empty when ``expand_over`` is empty.
        p_value: ``MetricOutput.metadata['p_value']`` for the resolved
            ``primary`` spec.
        result: Back-reference for survivor rendering; not read by the
            resolution layer itself.
    """

    identifier: tuple[Any, ...]
    expand_over_values: tuple[Any, ...]
    p_value: float
    result: EvaluationResult


def _partition(
    results: Sequence[EvaluationResult],
    *,
    func_name: str,
    expand_over: Sequence[str] = (),
) -> list[_PartitionEntry]:
    """Validate ``expand_over`` keys and partition-key uniqueness.

    Pulled out of ``_resolve_family`` so multi-primary callers run
    the per-result walk once, then attach per-primary p-values via
    ``_attach_p_values``.
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

    entries: list[_PartitionEntry] = []
    seen: dict[tuple[Any, ...], int] = {}
    for idx, result in enumerate(results):
        values = _expand_over_values(result, keys=keys, func_name=func_name)
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
            _PartitionEntry(
                identifier=identifier,
                expand_over_values=values,
                result=result,
            )
        )
    return entries


def _attach_p_values(
    partition: Sequence[_PartitionEntry],
    *,
    func_name: str,
    primary: MetricSpec,
) -> list[_FamilyEntry]:
    """Resolve per-primary p-value for each partition entry."""
    return [
        _FamilyEntry(
            identifier=p.identifier,
            expand_over_values=p.expand_over_values,
            p_value=_resolve_p_value(p.result, primary=primary, func_name=func_name),
            result=p.result,
        )
        for p in partition
    ]


def _resolve_family(
    results: Sequence[EvaluationResult],
    *,
    func_name: str,
    primary: MetricSpec,
    expand_over: Sequence[str] = (),
) -> list[_FamilyEntry]:
    """Single-primary convenience wrapper around ``_partition`` +
    ``_attach_p_values``.

    Steps (raise on failure, in order):

    1. ``expand_over`` names must exist either as a built-in slicing
       field (``forward_periods``) or as a key in every result's
       ``context``. ``factor`` is rejected (it is the hypothesis
       identifier, not a slicing axis).
    2. The partition key ``(factor, *expand_over_values)`` must be
       unique across the input.
    3. Each result must produce the ``primary`` metric's p-value;
       the p must be present and non-NaN.
    """
    partition = _partition(results, func_name=func_name, expand_over=expand_over)
    return _attach_p_values(partition, func_name=func_name, primary=primary)


def _expand_over_values(
    result: EvaluationResult,
    *,
    keys: list[str],
    func_name: str,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for name in keys:
        if name in _BUILTIN_EXPAND_OVER_FIELDS:
            values.append(getattr(result, name))
            continue
        if name not in result.context:
            raise UserInputError(
                func_name=func_name,
                field="expand_over",
                value=name,
                candidates=sorted(result.context)
                or ["<no context keys on this result>"],
                docs_path=f"api/{func_name}#expand_over",
            )
        values.append(result.context[name])
    return tuple(values)


def _resolve_p_value(
    result: EvaluationResult,
    *,
    primary: MetricSpec,
    func_name: str,
) -> float:
    try:
        out = result.metrics.outputs[primary.name]
    except KeyError:
        raise UserInputError(
            func_name=func_name,
            field="primary",
            value=primary.name,
            expected=(
                f"every result to carry the primary metric "
                f"{primary.name!r}; missing on factor={result.factor!r}"
            ),
            candidates=sorted(result.metrics.outputs),
            docs_path=f"api/{func_name}#primary",
        ) from None

    p = out.metadata.get("p_value")
    if p is None:
        raise UserInputError(
            func_name=func_name,
            field="primary",
            value=primary.name,
            expected=(
                f"metadata['p_value'] populated on every result; "
                f"factor={result.factor!r} has no p-value for "
                f"metric {primary.name!r}"
            ),
            docs_path=f"api/{func_name}#primary",
        )

    p_float = float(p)  # type: ignore[arg-type]
    if math.isnan(p_float):
        raise UserInputError(
            func_name=func_name,
            field="primary",
            value=primary.name,
            expected=(
                f"finite p-value on every result; factor={result.factor!r} "
                f"has NaN p for metric {primary.name!r} — drop the result "
                "or pick a different primary"
            ),
            docs_path=f"api/{func_name}#primary",
        )
    return p_float
