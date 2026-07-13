"""Shared family resolution layer for multiple-testing functions.

Every closed-form family function (``bhy`` / ``bhy_hierarchical`` /
``partial_conjunction``) runs through ``_resolve_family`` to turn a
list of :class:`~factrix._results.EvaluationResult` into flat
``_FamilyEntry`` records ready for the procedure-specific step-up math.

The invariants enforced here are the family-layer extension of the
anti-shopping defense. Identity and family partition are separate
concerns and are kept on separate knobs:

* **Identity** — ``(factor, forward_periods, *params)``. Every
  ``EvaluationResult.params`` entry joins the identifier automatically,
  so a swept knob (``base_tf``, ``universe``) never has to be encoded
  into the factor name to stay unique. ``EvaluationResult.metadata`` is
  bookkeeping and never joins the identifier.
* **Family partition** — ``expand_over`` alone, naming ``forward_periods``
  (the lone built-in) or ``params`` keys. It no longer doubles as an
  identity knob, so partitioning is a pure statistical declaration.

The estimator-override hook is
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


def _hypothesis_identity(
    result: EvaluationResult, *, exclude: tuple[str, ...] = ()
) -> tuple[Any, ...]:
    """Hypothesis identity: ``(factor, forward_periods, *sorted(params.items()))``.

    ``exclude`` strips named components — ``partial_conjunction`` passes its
    ``expand_over`` keys so results replicated along the condition axis
    collapse into one aggregation identity while every other swept knob
    keeps identities apart. Param keys ride along with their values so that
    results carrying different key sets cannot collide on values alone.
    """
    parts: list[Any] = [result.factor]
    if "forward_periods" not in exclude:
        parts.append(result.forward_periods)
    parts.extend(sorted((k, v) for k, v in result.params.items() if k not in exclude))
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class _FamilyEntry:
    """Flat record carrying one hypothesis through the family pipeline.

    Spans both stages: ``_partition`` emits entries with ``p_value=None``
    (identity resolved, p-value not yet attached); ``_attach_p_values``
    re-emits them per metric with ``p_value`` populated. Procedures read
    ``p_value`` only after the attach stage, where it is always non-None.

    Attributes:
        identifier: ``(factor, forward_periods, *sorted(params.items()))``
            — the hypothesis key. ``forward_periods`` and every ``params``
            entry always join the identity, independently of ``expand_over``.
            Param keys ride along with their values so that results carrying
            different key sets cannot collide on values alone.
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
                expected="partition key, not the hypothesis identifier 'factor'",
                docs_path=f"api/{func_name}#expand_over",
            )

    _check_param_keys(results, keys=keys, func_name=func_name)

    entries: list[_FamilyEntry] = []
    seen: dict[tuple[Any, ...], int] = {}
    for idx, result in enumerate(results):
        values = _expand_over_values(result, keys=keys)
        identifier = _hypothesis_identity(result)
        if identifier in seen:
            raise UserInputError(
                func_name=func_name,
                field="results",
                value=identifier,
                expected=(
                    "unique (factor, forward_periods, *params) identifier "
                    f"across input; duplicate first seen at index "
                    f"{seen[identifier]}, again at {idx}. Two results that "
                    "differ only in `metadata` are the same hypothesis — "
                    "metadata is bookkeeping and never disambiguates. If the "
                    "repeats are a swept knob (base_tf, universe, ...), stamp "
                    "it on `EvaluationResult.params`; it then joins the "
                    "identifier without partitioning the family"
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

    1. ``expand_over`` names must exist either as a built-in field
       (``forward_periods``) or as a key in every result's ``params``.
       ``factor`` is rejected (it is the hypothesis identifier, not a
       partition key); a ``metadata`` key is rejected (bookkeeping does
       not define a family).
    2. The hypothesis key ``(factor, forward_periods, *params)`` must be
       unique across the input.
    3. Each result must produce the ``metric``'s p-value;
       the p must be present and non-NaN.
    """
    partition = _partition(results, func_name=func_name, expand_over=expand_over)
    return _attach_p_values(partition, func_name=func_name, metric=metric)


def _check_param_keys(
    results: Sequence[EvaluationResult],
    *,
    keys: list[str],
    func_name: str,
) -> None:
    """Raise once listing every ``(factor, missing_key)`` across the input.

    Built-in fields (``forward_periods``) are read off the result and never
    missing; only ``params`` keys are checked. A single failed result does
    not short-circuit the whole screen — integrating results from several
    sources surfaces all gaps in one pass, matching the
    aggregate-then-report idiom of ``evaluate``'s strict / column guards.
    fail-loud is preserved: any gap still raises (silently dropping a result
    would alter family composition and the FDR denominator).

    A key that lives on ``metadata`` instead of ``params`` gets a targeted
    message: it is present in the input, but bookkeeping labels do not
    define a family, so partitioning on one is rejected rather than
    silently honoured.
    """
    param_keys = [k for k in keys if k not in _BUILTIN_EXPAND_OVER_FIELDS]
    if not param_keys:
        return
    missing = [
        (result.factor, name)
        for result in results
        for name in param_keys
        if name not in result.params
    ]
    if not missing:
        return
    missing_keys = sorted({k for _, k in missing})
    available = sorted({k for result in results for k in result.params})
    on_metadata = sorted({k for k in missing_keys for r in results if k in r.metadata})
    detail = "; ".join(f"factor={factor!r} missing {key!r}" for factor, key in missing)
    hint = (
        (
            f" Key(s) {on_metadata!r} live on `metadata`, which is bookkeeping "
            f"and never partitions a family — move them to `params` if they "
            f"are a swept knob."
        )
        if on_metadata
        else ""
    )
    raise UserInputError(
        func_name=func_name,
        field="expand_over",
        value=missing_keys,
        expected=(
            f"expand_over key(s) {missing_keys!r} present in every result's "
            f"params. Missing — {detail}.{hint} Stamp the key on every "
            f"EvaluationResult.params, or drop it from expand_over. "
            f"Param keys seen across the input: {available or ['<none>']!r}"
        ),
        docs_path=f"api/{func_name}#expand_over",
    )


def _expand_over_values(
    result: EvaluationResult,
    *,
    keys: list[str],
) -> tuple[Any, ...]:
    """Read the ``expand_over`` value tuple off one result.

    Presence of every param key is guaranteed by a prior
    ``_check_param_keys`` sweep, so this reads without re-validating.
    """
    return tuple(
        getattr(result, name)
        if name in _BUILTIN_EXPAND_OVER_FIELDS
        else result.params[name]
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
