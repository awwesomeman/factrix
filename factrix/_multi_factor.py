"""Collection-level FDR-control functions (``bhy`` / ``partial_conjunction``
/ ``bhy_hierarchical``).

All three accept ``list[EvaluationResult]`` and a list of primary
:class:`~factrix._metric_index.MetricSpec` records; each function runs
one independent screen per primary and returns a
``dict[primary_name, *Result]`` keyed by ``MetricSpec.name`` (single
primary still returns a dict — no isinstance dispatch on the caller
side).

Family declaration is explicit: the input list *is* the family,
optionally split per-bucket via ``expand_over``. Hypothesis identifier
is ``(factor, *expand_over_values)``. Cell / horizon partitioning is
the caller's responsibility.
"""

from __future__ import annotations

import html
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from factrix._errors import UserInputError
from factrix._family import _attach_p_values, _partition, _resolve_family
from factrix._metric_index import MetricSpec
from factrix.stats.multiple_testing import (
    bhy_adjusted_p,
    partial_conjunction_p,
    simes_p,
)

if TYPE_CHECKING:
    from factrix._results import EvaluationResult


_NOT_METRICSPEC_EXPECTED = (
    "MetricSpec instance (str / Callable not accepted — pick the spec from "
    "fx.metrics.spec_by_name() or the metric module's __metric_specs__ tuple)"
)


def _validate_spec_list(value: Any, *, func_name: str, field: str) -> list[MetricSpec]:
    """Shared validator: ``list[MetricSpec]`` canonical form.

    Used by ``bhy.primary`` / ``compare.metrics`` so both surfaces give
    identical error messages for the same misuse.
    """
    anchor = f"api/{func_name}#{field}"
    if not isinstance(value, list):
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=type(value).__name__,
            expected=(f"list[MetricSpec] (always a list, even for a single {field})"),
            docs_path=anchor,
        )
    if not value:
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=value,
            expected="non-empty list[MetricSpec]",
            docs_path=anchor,
        )
    for i, spec in enumerate(value):
        if not isinstance(spec, MetricSpec):
            raise UserInputError(
                func_name=func_name,
                field=f"{field}[{i}]",
                value=type(spec).__name__,
                expected=_NOT_METRICSPEC_EXPECTED,
                docs_path=anchor,
            )
    return list(value)


def _require_non_empty_results(
    results: Sequence[EvaluationResult], *, func_name: str
) -> None:
    """Shared empty-input guard: every public function raises rather than
    returning an empty container — FDR over an empty candidate set has
    no meaning and ranking an empty leaderboard is undefined.
    """
    if not results:
        raise UserInputError(
            func_name=func_name,
            field="results",
            value=results,
            expected="non-empty list[EvaluationResult]",
            docs_path=f"api/{func_name}#results",
        )


def _render_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return ""
    widths = [
        max(len(col), *(len(row[i]) for row in rows)) for i, col in enumerate(headers)
    ]
    header_line = "  ".join(c.ljust(w) for c, w in zip(headers, widths, strict=True))
    body = "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True))
        for row in rows
    )
    return f"{header_line}\n{body}"


def _render_html(
    caption: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]
) -> str:
    thead = "".join(
        f"<th style='text-align:left'>{html.escape(h)}</th>" for h in headers
    )
    tbody = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        "<table class='factrix-survivors'>"
        f"<caption>{html.escape(caption)}</caption>"
        f"<thead><tr>{thead}</tr></thead>"
        f"<tbody>{tbody}</tbody>"
        "</table>"
    )


@dataclass(frozen=True, slots=True, repr=False)
class _FdrResultBase:
    """Shared fields and render protocol for the FDR survivor result trio.

    Carries the five fields common to :class:`BhyResult`,
    :class:`PartialConjunctionResult`, and :class:`HierarchicalBhyResult`
    plus the ``__len__`` / ``__repr__`` / ``_repr_html_`` machinery.
    Subclasses append their own fields and supply ``_header`` / ``_rows``.

    Common attributes:
        primary_name: ``MetricSpec.name`` of the primary driving the screen;
            also the key under which the record is returned.
        survivors: Surviving :class:`EvaluationResult` records.
        adj_p: BHY-adjusted p-value aligned with ``survivors``.
        q: Nominal FDR target.
        n_tests: Per-bucket / per-identity family size keyed by tuple.
    """

    primary_name: str
    survivors: list[EvaluationResult]
    adj_p: np.ndarray
    q: float
    n_tests: Mapping[tuple[Any, ...], int]

    def __len__(self) -> int:
        return len(self.survivors)

    def _header(self) -> str:
        raise NotImplementedError

    def _rows(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = f"{type(self).__name__}({self._header()})"
        if not self.survivors:
            return head
        headers, rows = self._rows()
        return f"{head}\n{_render_table(headers, rows)}"

    def _repr_html_(self) -> str:
        headers, rows = self._rows()
        return _render_html(f"{type(self).__name__} ({self._header()})", headers, rows)


@dataclass(frozen=True, slots=True, repr=False)
class BhyResult(_FdrResultBase):
    """Result of one BHY step-up for one primary metric.

    Shares ``primary_name`` / ``survivors`` / ``adj_p`` / ``q`` / ``n_tests``
    with :class:`_FdrResultBase`. ``adj_p`` is bucket-local; ``n_tests`` is
    keyed by ``expand_over_values`` tuple (``()`` for single-bucket).

    Attributes:
        expand_over: Keys used to partition the input into independent
            step-up buckets; empty tuple for the single-bucket case.
    """

    expand_over: tuple[str, ...]

    def _header(self) -> str:
        parts = [
            f"primary={self.primary_name}",
            f"n={len(self.survivors)}",
            f"q={self.q:g}",
        ]
        if self.expand_over:
            parts.append(f"expand_over={list(self.expand_over)!r}")
        return ", ".join(parts)

    def _rows(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        headers: tuple[str, ...]
        rows: list[tuple[str, ...]]
        if self.expand_over:
            headers = ("expand_over_values", "factor", "adj_p")
            rows = [
                (
                    repr(tuple(_lookup_expand(r, k) for k in self.expand_over)),
                    r.factor,
                    f"{float(a):.4g}",
                )
                for r, a in zip(self.survivors, self.adj_p, strict=True)
            ]
        else:
            headers = ("factor", "adj_p")
            rows = [
                (r.factor, f"{float(a):.4g}")
                for r, a in zip(self.survivors, self.adj_p, strict=True)
            ]
        return headers, rows


@dataclass(frozen=True, slots=True, repr=False)
class PartialConjunctionResult(_FdrResultBase):
    """Per-identity partial-conjunction survivors for one primary metric.

    Shares ``primary_name`` / ``survivors`` / ``adj_p`` / ``q`` / ``n_tests``
    with :class:`_FdrResultBase`. ``survivors`` is one representative
    :class:`EvaluationResult` per surviving identity; ``adj_p`` is the
    BHY-adjusted PC p-value; ``n_tests`` is the condition count per
    surviving identifier.

    Attributes:
        pc_p: Raw PC p-value aligned with ``survivors``.
        expand_over: Context keys defining the condition axis.
        min_pass: ``k`` in the ``k`` of ``m`` partial conjunction test.
        n_passed_uncorr: Per-survivor count of raw p-values strictly
            below ``q`` (descriptive — not used in inference).
    """

    pc_p: np.ndarray
    expand_over: tuple[str, ...]
    min_pass: int
    n_passed_uncorr: np.ndarray

    def _header(self) -> str:
        return (
            f"primary={self.primary_name}, n={len(self.survivors)}, "
            f"q={self.q:g}, min_pass={self.min_pass}, "
            f"expand_over={list(self.expand_over)!r}"
        )

    def _rows(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        headers: tuple[str, ...] = (
            "factor",
            "pc_p",
            "adj_p",
            "n_tests",
            "n_passed_uncorr",
        )
        rows: list[tuple[str, ...]] = [
            (
                r.factor,
                f"{float(pc):.4g}",
                f"{float(adj):.4g}",
                str(self.n_tests.get((r.factor,), 0)),
                str(int(n)),
            )
            for r, pc, adj, n in zip(
                self.survivors,
                self.pc_p,
                self.adj_p,
                self.n_passed_uncorr,
                strict=True,
            )
        ]
        return headers, rows


@dataclass(frozen=True, slots=True, repr=False)
class HierarchicalBhyResult(_FdrResultBase):
    """Two-stage hierarchical BHY survivors for one primary metric.

    Shares ``primary_name`` / ``survivors`` / ``adj_p`` / ``q`` / ``n_tests``
    with :class:`_FdrResultBase`. ``adj_p`` is
    ``max(outer_adj_p[group], inner_adj_p[i])`` aligned with ``survivors``
    so ``survivor[i] iff adj_p[i] <= q`` holds; ``q`` is shared by both
    layers; ``n_tests`` is the per-group inner family size keyed by
    ``(group_value,)`` — covering *all* input groups, not just survivors.

    Attributes:
        group: Context key naming the group axis.
    """

    group: str

    def _header(self) -> str:
        return (
            f"primary={self.primary_name}, n={len(self.survivors)}, "
            f"q={self.q:g}, group={self.group!r}"
        )

    def _rows(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        headers: tuple[str, ...] = ("group", "factor", "adj_p")
        rows: list[tuple[str, ...]] = [
            (
                repr(_lookup_expand(r, self.group)),
                r.factor,
                f"{float(a):.4g}",
            )
            for r, a in zip(self.survivors, self.adj_p, strict=True)
        ]
        return headers, rows


def _lookup_expand(result: EvaluationResult, key: str) -> Any:
    if key == "forward_periods":
        return result.forward_periods
    return result.context.get(key)


def bhy(
    results: list[EvaluationResult],
    *,
    primary: list[MetricSpec],
    expand_over: tuple[str, ...] = (),
    q: float = 0.05,
) -> dict[str, BhyResult]:
    """Benjamini-Hochberg-Yekutieli step-up FDR, one screen per primary.

    The input list is treated as a single family. When ``expand_over``
    is non-empty, one independent step-up runs per unique tuple of
    ``expand_over`` values (read from ``EvaluationResult.forward_periods``
    for that built-in slicing axis, otherwise from
    ``EvaluationResult.context[k]``). Cell / horizon partitioning is
    the caller's responsibility.

    Args:
        results: :class:`EvaluationResult` records. The full input is
            one family unless ``expand_over`` further partitions it.
        primary: ``list[MetricSpec]`` — element type strictly
            :class:`MetricSpec`. One independent BHY screen runs per
            primary; the return dict is keyed by ``MetricSpec.name``.
            Single-primary callers still receive a one-key dict (no
            isinstance branching downstream).
        expand_over: Tuple of keys whose distinct value tuples split
            the input into independent BHY step-up buckets. Built-in
            field ``"forward_periods"`` is read off the result;
            other keys are looked up on ``result.context``.
        q: Nominal FDR target. Default ``0.05``.

    Returns:
        ``dict[str, BhyResult]`` keyed by ``MetricSpec.name``.

    Raises:
        UserInputError: ``primary`` not a non-empty ``list[MetricSpec]``;
            duplicate ``(factor, *expand_over_values)`` identifier;
            ``expand_over`` key missing from a result's context or
            naming ``'factor'``; primary metric absent from a result's
            outputs or its ``p_value`` missing / NaN.

    Warns:
        RuntimeWarning: Input mixes ``forward_periods`` with
            ``expand_over`` empty (pooling horizons dilutes the
            per-rank threshold). Or most ``expand_over`` buckets are
            singletons (BHY on n=1 provides no FDR correction).
    """
    primary_list = _validate_spec_list(primary, func_name="bhy", field="primary")
    _require_non_empty_results(results, func_name="bhy")
    expand_over_tuple = tuple(expand_over)

    _warn_on_mixed_horizons(results, expand_over=expand_over_tuple)

    partition = _partition(results, func_name="bhy", expand_over=expand_over_tuple)
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, p_entry in enumerate(partition):
        buckets[p_entry.expand_over_values].append(idx)

    singleton = sum(1 for ix in buckets.values() if len(ix) == 1)
    if singleton and len(buckets) > 1:
        warnings.warn(
            f"bhy: {singleton} of {len(buckets)} expand_over buckets "
            "contain a single result — BHY on n=1 is identical to a "
            "raw threshold and provides no FDR correction.",
            RuntimeWarning,
            stacklevel=2,
        )

    n_tests: dict[tuple[Any, ...], int] = {
        bucket_key: len(ix) for bucket_key, ix in buckets.items()
    }

    out: dict[str, BhyResult] = {}
    for spec in primary_list:
        entries = _attach_p_values(partition, func_name="bhy", primary=spec)
        adj_p_all = np.full(len(entries), np.nan, dtype=np.float64)
        for ix in buckets.values():
            p_array = np.array([entries[i].p_value for i in ix], dtype=np.float64)
            adj_p_all[ix] = bhy_adjusted_p(p_array)

        survivor_idxs = np.flatnonzero(adj_p_all <= q)
        out[spec.name] = BhyResult(
            primary_name=spec.name,
            survivors=[entries[i].result for i in survivor_idxs],
            adj_p=adj_p_all[survivor_idxs],
            q=q,
            expand_over=expand_over_tuple,
            n_tests=n_tests,
        )
    return out


def partial_conjunction(
    results: list[EvaluationResult],
    *,
    primary: list[MetricSpec],
    min_pass: int,
    expand_over: tuple[str, ...],
    n_conditions: int | None = None,
    q: float = 0.05,
) -> dict[str, PartialConjunctionResult]:
    """Partial-conjunction screening: identities significant in
    ``min_pass`` of ``m`` conditions, one screen per primary.

    For "factor X is significant in universes A and B" style claims,
    naive intersection-of-survivors does not preserve FDR
    (Benjamini-Bogomolov 2014). The partial conjunction test
    (Benjamini-Heller 2008) combines an identity's per-condition
    p-values into a single PC p-value, then runs BHY across identities.

    PC formula (Bonferroni-style): for ``k = min_pass``,
    ``p_PC = (m - k + 1) * p_((k))`` capped at 1.
    ``k = 1`` reduces to ``m * min(p)`` (union semantics) and is
    forbidden.

    Args:
        results: :class:`EvaluationResult` records. Conditions per
            identity come from ``expand_over``; multiple results
            sharing the hypothesis identifier (``factor``) must differ
            on at least one ``expand_over`` key.
        primary: ``list[MetricSpec]`` — one PC screen runs per primary;
            return dict keyed by ``MetricSpec.name``.
        min_pass: ``k`` in "k of m". Must be ``>= 2``.
        expand_over: Non-empty tuple of context keys (or
            ``"forward_periods"``) defining the condition axis.
        n_conditions: Strict-structure declaration. ``None`` lets ``m`` be
            inferred per identity; an ``int`` requires every identity
            to have exactly that many conditions.
        q: Nominal FDR target for the BHY step-up over PC p-values.

    Returns:
        ``dict[str, PartialConjunctionResult]`` keyed by
        ``MetricSpec.name``.

    Raises:
        UserInputError: ``min_pass < 2``; ``expand_over`` empty;
            ``n_conditions < min_pass``; strict-structure mismatch;
            identity with fewer than ``min_pass`` conditions; any
            ``_resolve_family`` invariant failure.
    """
    primary_list = _validate_spec_list(
        primary, func_name="partial_conjunction", field="primary"
    )
    _require_non_empty_results(results, func_name="partial_conjunction")

    if min_pass < 2:
        expected = "positive integer >= 2"
        if min_pass == 1:
            expected = (
                "min_pass >= 2. min_pass=1 reduces to union semantics "
                "(rejects when any single condition is significant) and "
                "inflates FDR to ~2q under independence; the partial "
                "conjunction surface is contract-bearing only for k >= 2. "
                "The closest available alternative is "
                "bhy(results, expand_over=(...,)), which expands the family "
                "across conditions instead of aggregating (survivor unit "
                "becomes (factor, condition) pair, not factor identifier)"
            )
        raise UserInputError(
            func_name="partial_conjunction",
            field="min_pass",
            value=min_pass,
            expected=expected,
            docs_path="api/partial-conjunction#min-pass-must-be-at-least-2",
        )

    if not expand_over:
        raise UserInputError(
            func_name="partial_conjunction",
            field="expand_over",
            value=expand_over,
            expected=(
                "non-empty tuple of context keys naming the condition "
                "axis (e.g. ('region',) for cross-region replication, "
                "('forward_periods',) for cross-horizon replication). "
                "partial_conjunction is undefined without a condition axis"
            ),
            docs_path="api/partial-conjunction#expand-over-required",
        )

    if n_conditions is not None and n_conditions < min_pass:
        raise UserInputError(
            func_name="partial_conjunction",
            field="n_conditions",
            value=n_conditions,
            expected=(
                f"n_conditions >= min_pass ({min_pass}); requiring more "
                "passes than total conditions is unsatisfiable"
            ),
            docs_path="api/partial-conjunction#n-conditions",
        )

    expand_over_tuple = tuple(expand_over)

    out: dict[str, PartialConjunctionResult] = {}
    for spec in primary_list:
        out[spec.name] = _partial_conjunction_one(
            results,
            primary=spec,
            min_pass=min_pass,
            expand_over=expand_over_tuple,
            n_conditions=n_conditions,
            q=q,
        )
    return out


def _partial_conjunction_one(
    results: Sequence[EvaluationResult],
    *,
    primary: MetricSpec,
    min_pass: int,
    expand_over: tuple[str, ...],
    n_conditions: int | None,
    q: float,
) -> PartialConjunctionResult:
    entries = _resolve_family(
        results,
        func_name="partial_conjunction",
        primary=primary,
        expand_over=expand_over,
    )

    entries_by_factor: dict[str, list[Any]] = defaultdict(list)
    identifiers_ordered: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        factor = entry.result.factor
        entries_by_factor[factor].append(entry)
        if factor not in seen:
            seen.add(factor)
            identifiers_ordered.append(factor)

    pc_p_arr = np.empty(len(identifiers_ordered), dtype=np.float64)
    n_passed_arr = np.empty(len(identifiers_ordered), dtype=np.int64)
    n_tests_per_id: dict[tuple[Any, ...], int] = {}
    rep_results: list[EvaluationResult] = []

    for i, factor in enumerate(identifiers_ordered):
        group = entries_by_factor[factor]
        m = len(group)

        if n_conditions is not None and m != n_conditions:
            raise UserInputError(
                func_name="partial_conjunction",
                field="n_conditions",
                value=n_conditions,
                expected=(
                    f"factor {factor!r} has {m} condition(s) in data but "
                    f"n_conditions={n_conditions} declared (strict structure). "
                    "Pass n_conditions=None for lenient structure, or fix the "
                    f"input so every factor has exactly {n_conditions} "
                    "conditions"
                ),
                docs_path="api/partial-conjunction#strict-vs-lenient",
            )

        if m < min_pass:
            raise UserInputError(
                func_name="partial_conjunction",
                field="results",
                value=factor,
                expected=(
                    f"factor {factor!r} has only {m} condition(s) but "
                    f"min_pass={min_pass} requires at least that many. "
                    "Either drop this factor from the input or lower min_pass"
                ),
                docs_path="api/partial-conjunction#insufficient-conditions",
            )

        ps = np.array([e.p_value for e in group], dtype=np.float64)
        pc_p_arr[i] = partial_conjunction_p(ps, min_pass=min_pass)
        n_passed_arr[i] = int(np.sum(ps < q))
        n_tests_per_id[(factor,)] = m
        rep_results.append(group[0].result)

    if n_conditions is None:
        m_values = set(n_tests_per_id.values())
        if len(m_values) > 1:
            warnings.warn(
                "partial_conjunction: lenient structure (n_conditions=None) "
                f"is running with heterogeneous condition counts "
                f"m={sorted(m_values)} across factors. PC p-values are "
                "valid marginally but the k/m bar differs per factor. "
                "For cross-factor comparability pass n_conditions=<int> "
                "(strict structure) or split the call.",
                RuntimeWarning,
                stacklevel=3,
            )

    adj_p_all = bhy_adjusted_p(pc_p_arr)
    survivor_idx = np.flatnonzero(adj_p_all <= q)
    surviving_factors = [identifiers_ordered[i] for i in survivor_idx]
    return PartialConjunctionResult(
        primary_name=primary.name,
        survivors=[rep_results[i] for i in survivor_idx],
        adj_p=adj_p_all[survivor_idx],
        pc_p=pc_p_arr[survivor_idx],
        q=q,
        expand_over=expand_over,
        min_pass=min_pass,
        n_tests={(f,): n_tests_per_id[(f,)] for f in surviving_factors},
        n_passed_uncorr=n_passed_arr[survivor_idx],
    )


def bhy_hierarchical(
    results: list[EvaluationResult],
    *,
    primary: list[MetricSpec],
    group: str,
    q: float = 0.05,
) -> dict[str, HierarchicalBhyResult]:
    """Yekutieli (2008) two-stage hierarchical BHY, one screen per primary.

    Controls group-level FDR ≤ ``q`` on the outer layer (Simes group
    representative + BHY) and within-group FDR ≤ ``q`` on the inner
    layer (BHY restricted to passing groups). Flat BHY across the
    whole input loses group-level interpretability and pays full
    m-correction even when most groups are dead.

    Args:
        results: :class:`EvaluationResult` records. Each is assigned
            to one group via ``result.context[group]`` (or via
            ``result.forward_periods`` if ``group == "forward_periods"``).
            Within a group, ``factor`` must be unique.
        primary: ``list[MetricSpec]`` — one hierarchical screen per
            primary; return dict keyed by ``MetricSpec.name``.
        group: Single key naming the group axis.
        q: Nominal FDR target shared by both layers.

    Returns:
        ``dict[str, HierarchicalBhyResult]`` keyed by ``MetricSpec.name``.

    Raises:
        UserInputError: ``group == 'factor'`` (it is the identifier);
            only one distinct group value in the input (call ``bhy``
            instead); every result is its own group at ``n >= 3``;
            duplicate ``(factor, group_value)`` identifier; any
            ``_resolve_family`` invariant failure.

    Warns:
        RuntimeWarning: More than half of input groups contain a single
            result — inner BHY on n=1 is a raw cutoff and the outer
            Simes representative equals that single p-value.
    """
    primary_list = _validate_spec_list(
        primary, func_name="bhy_hierarchical", field="primary"
    )
    _require_non_empty_results(results, func_name="bhy_hierarchical")

    out: dict[str, HierarchicalBhyResult] = {}
    for spec in primary_list:
        out[spec.name] = _bhy_hierarchical_one(results, primary=spec, group=group, q=q)
    return out


def _bhy_hierarchical_one(
    results: Sequence[EvaluationResult],
    *,
    primary: MetricSpec,
    group: str,
    q: float,
) -> HierarchicalBhyResult:
    entries = _resolve_family(
        results,
        func_name="bhy_hierarchical",
        primary=primary,
        expand_over=(group,),
    )

    buckets: dict[Any, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        buckets[entry.expand_over_values[0]].append(idx)
    group_keys_ordered = list(
        dict.fromkeys(entry.expand_over_values[0] for entry in entries)
    )

    n_groups = len(group_keys_ordered)
    if n_groups == 1:
        raise UserInputError(
            func_name="bhy_hierarchical",
            field="group",
            value=group,
            expected=(
                "a key with at least 2 distinct values across input "
                f"results; got 1 group ({group_keys_ordered[0]!r}). A "
                "single group reduces the procedure to plain BHY on the "
                "members. Call bhy(results, primary=[...]) directly"
            ),
            docs_path="api/bhy-hierarchical#validation-summary",
        )
    if n_groups == len(entries) and len(entries) >= 3:
        raise UserInputError(
            func_name="bhy_hierarchical",
            field="group",
            value=group,
            expected=(
                f"a key that partitions results into groups of size >= 2; "
                f"got {n_groups} groups across {len(entries)} results "
                "(every result is its own group). Pick a coarser "
                "categorical (family / region / sector) or call bhy() "
                "without grouping"
            ),
            docs_path="api/bhy-hierarchical#validation-summary",
        )

    singletons = sum(1 for ix in buckets.values() if len(ix) == 1)
    if singletons * 2 > n_groups:
        warnings.warn(
            f"bhy_hierarchical: {singletons} of {n_groups} groups "
            "contain a single result — inner BHY on n=1 is a raw cutoff "
            "and the outer Simes representative equals that single "
            "p-value, so those groups get no FDR correction at either "
            "layer.",
            RuntimeWarning,
            stacklevel=3,
        )

    group_simes = np.empty(n_groups, dtype=np.float64)
    inner_adjs: list[np.ndarray] = []
    n_tests: dict[tuple[Any, ...], int] = {}
    for g_idx, gkey in enumerate(group_keys_ordered):
        member_idxs = buckets[gkey]
        member_p = np.array([entries[i].p_value for i in member_idxs], dtype=np.float64)
        group_simes[g_idx] = simes_p(member_p)
        inner_adjs.append(bhy_adjusted_p(member_p))
        n_tests[(gkey,)] = len(member_idxs)

    outer_adj = bhy_adjusted_p(group_simes)

    adj_p_all = np.empty(len(entries), dtype=np.float64)
    for g_idx, gkey in enumerate(group_keys_ordered):
        for j, idx in enumerate(buckets[gkey]):
            adj_p_all[idx] = max(outer_adj[g_idx], inner_adjs[g_idx][j])

    survivor_idxs = np.flatnonzero(adj_p_all <= q)
    return HierarchicalBhyResult(
        primary_name=primary.name,
        survivors=[entries[i].result for i in survivor_idxs],
        adj_p=adj_p_all[survivor_idxs],
        q=q,
        group=group,
        n_tests=n_tests,
    )


def _warn_on_mixed_horizons(
    results: Sequence[EvaluationResult],
    *,
    expand_over: tuple[str, ...],
) -> None:
    if expand_over:
        return
    horizons = {r.forward_periods for r in results}
    if len(horizons) > 1:
        warnings.warn(
            f"bhy: input mixes forward_periods={sorted(horizons)} but "
            "expand_over is empty — different horizons have different "
            "null distributions; pooling them in one step-up dilutes "
            "the per-rank threshold and silently inflates FDR. Either "
            "split the call per horizon, or set "
            "expand_over=('forward_periods',) to declare per-bucket "
            "families.",
            RuntimeWarning,
            stacklevel=3,
        )
