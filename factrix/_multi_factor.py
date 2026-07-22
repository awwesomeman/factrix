"""Collection-level FDR control for per-metric and cross-metric families.

The original ``bhy`` / ``partial_conjunction`` / ``bhy_hierarchical``
procedures run one independent screen per metric and return a
``dict[metric_name, *Result]``. Cross-metric procedures share the
``list[EvaluationResult]`` input but return one result with an explicit
cell-level or factor-level survivor unit.

Family declaration is explicit: the input list *is* the family,
optionally split per-bucket via ``expand_over``. The base hypothesis
identifier is ``(factor, forward_periods, *params)`` — every swept knob on
``EvaluationResult.params`` joins it automatically. ``expand_over`` only
partitions the family; ``metadata`` never touches either.

``bhy_across_metrics`` extends that identity with a metric label;
``partial_conjunction_across_metrics`` treats the declared labels as a fixed
k-of-m condition axis and returns factor-level identities.
"""

from __future__ import annotations

import html
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from factrix._errors import UserInputError, _api_docs_path
from factrix._family import (
    _attach_p_values,
    _hypothesis_identity,
    _partition,
    _resolve_family,
)
from factrix.stats.multiple_testing import (
    bhy_adjusted_p,
    partial_conjunction_p,
    simes_p,
)

if TYPE_CHECKING:
    from factrix._results import EvaluationResult


_NOT_METRICSPEC_EXPECTED = (
    "metric label instance (str / Callable not accepted — pick the spec from "
    "fx.metrics.spec_by_name())"
)


def _validate_metric_list(value: Any, *, func_name: str, field: str) -> list[str]:
    """Shared validator: ``list[str]`` canonical form.

    Used by ``bhy.metrics`` / ``compare.metrics`` so both surfaces give
    identical error messages for the same misuse.
    """
    anchor = _api_docs_path(func_name, field)
    if not isinstance(value, list):
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=type(value).__name__,
            expected=(f"list[str] (always a list, even for a single {field})"),
            docs_path=anchor,
        )
    if not value:
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=value,
            expected="non-empty list[str]",
            docs_path=anchor,
        )
    for i, spec in enumerate(value):
        if not isinstance(spec, str):
            raise UserInputError(
                func_name=func_name,
                field=f"{field}[{i}]",
                value=type(spec).__name__,
                expected="str metric label (e.g. 'ic')",
                docs_path=anchor,
            )
    return list(value)


def _validate_cross_metric_list(value: Any, *, func_name: str) -> list[str]:
    """Validate the ordered, unique metric axis for cross-metric screens."""
    metrics = _validate_metric_list(value, func_name=func_name, field="metrics")
    duplicates = sorted({name for name in metrics if metrics.count(name) > 1})
    if duplicates:
        raise UserInputError(
            func_name=func_name,
            field="metrics",
            value=duplicates,
            expected="unique metric labels; duplicates would repeat one hypothesis",
            docs_path=_api_docs_path(func_name, "metrics"),
        )
    if len(metrics) < 2:
        raise UserInputError(
            func_name=func_name,
            field="metrics",
            value=metrics,
            expected=(
                "at least 2 metric labels for a cross-metric family; "
                "use bhy() for one metric"
            ),
            docs_path=_api_docs_path(func_name, "metrics"),
        )
    return metrics


def _validate_q(value: Any, *, func_name: str) -> float:
    """Validate the nominal FDR target shared by screening procedures."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise UserInputError(
            func_name=func_name,
            field="q",
            value=value,
            expected="float in the open interval (0, 1)",
            docs_path=_api_docs_path(func_name, "q"),
        )
    q = float(value)
    if not np.isfinite(q) or not 0.0 < q < 1.0:
        raise UserInputError(
            func_name=func_name,
            field="q",
            value=value,
            expected="float in the open interval (0, 1)",
            docs_path=_api_docs_path(func_name, "q"),
        )
    return q


def _require_non_empty_results(
    results: Sequence[EvaluationResult], *, func_name: str
) -> None:
    """Shared results-input guard: every public function raises rather than
    returning an empty container — FDR over an empty candidate set has
    no meaning and ranking an empty leaderboard is undefined. Also catches
    the natural ``evaluate`` follow-up mistake of forwarding its ``dict``
    return directly, pointing at ``list(results.values())``.
    """
    if isinstance(results, Mapping):
        raise UserInputError(
            func_name=func_name,
            field="results",
            value=type(results).__name__,
            expected=(
                "list[EvaluationResult], not the dict returned by evaluate() — "
                "pass list(results.values())"
            ),
            docs_path=_api_docs_path(func_name, "results"),
        )
    if not results:
        raise UserInputError(
            func_name=func_name,
            field="results",
            value=results,
            expected="non-empty list[EvaluationResult]",
            docs_path=_api_docs_path(func_name, "results"),
        )
    from factrix._results import EvaluationResult

    for idx, result in enumerate(results):
        if not isinstance(result, EvaluationResult):
            raise UserInputError(
                func_name=func_name,
                field=f"results[{idx}]",
                value=type(result).__name__,
                expected=(
                    "EvaluationResult objects. If this list came from "
                    "evaluate(), pass list(results.values()); do not extend "
                    "a list with the dict itself, because that appends keys"
                ),
                docs_path=_api_docs_path(func_name, "results"),
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


class _ScreenResultMixin:
    """Shared survivor / adjusted-p / repr protocol for FDR screen results.

    Plain (non-dataclass) mixin so every screen result — single-metric or
    cross-metric, keyed by :class:`EvaluationResult` or
    :class:`MetricHypothesis` — shares one ``__len__`` / ``__repr__`` /
    ``_repr_html_`` implementation instead of each concrete result
    reimplementing it. Concrete classes provide ``entries`` / ``adj_p_all`` /
    ``q`` as dataclass fields plus ``_header`` / ``_rows`` for rendering.

    The canonical store is the **full** tested family — ``entries`` with
    ``adj_p_all`` aligned to it — so eliminated entries' adjusted p-values
    survive (a screen of N passing 2 still shows how far the other N-2 sat
    from the threshold). :attr:`survivors` / :attr:`adj_p` are the surviving
    subset, derived on access.
    """

    __slots__ = ()

    entries: Sequence[Any]
    adj_p_all: np.ndarray
    q: float

    @property
    def _survived(self) -> np.ndarray:
        """Boolean mask over ``entries`` — ``adj_p_all <= q``.

        ``NaN <= q`` is ``False``, so a data-shortage entry (dropped before
        the family formed) is correctly reported as not surviving.
        """
        return self.adj_p_all <= self.q

    @property
    def survivors(self) -> list[Any]:
        """The surviving entries, input order."""
        return [e for e, ok in zip(self.entries, self._survived, strict=True) if ok]

    @property
    def adj_p(self) -> np.ndarray:
        """Adjusted p-value for the survivors, aligned with :attr:`survivors`."""
        return self.adj_p_all[self._survived]

    def __len__(self) -> int:
        return int(self._survived.sum())

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
class _FdrResultBase(_ScreenResultMixin):
    """Shared fields for the single-metric FDR survivor result trio.

    Carries the fields common to :class:`BhyResult`,
    :class:`PartialConjunctionResult`, and :class:`HierarchicalBhyResult`.
    Subclasses append their own fields and supply ``_header`` / ``_rows``.

    Common attributes:
        metric_name: ``label`` of the metric driving the screen;
            also the key under which the record is returned.
        entries: Every tested :class:`EvaluationResult`, in input order.
        adj_p_all: BHY-adjusted p-value aligned with ``entries``; ``NaN``
            for an entry dropped before the family formed (data shortage).
        q: Nominal FDR target; must satisfy ``0 < q < 1``.
        n_tests: Per-bucket / per-identity family size keyed by tuple.
    """

    metric_name: str
    entries: list[EvaluationResult]
    adj_p_all: np.ndarray
    q: float
    n_tests: Mapping[tuple[Any, ...], int]

    def to_frame(self) -> pl.DataFrame:
        """Every tested factor with its adjusted p-value and survive flag.

        Columns ``factor`` / ``adj_p`` / ``survived``, one row per tested
        factor (input order) — including the eliminated ones that
        :attr:`survivors` and :attr:`adj_p` drop.
        """
        return pl.DataFrame(
            {
                "factor": [e.factor for e in self.entries],
                "adj_p": self.adj_p_all,
                "survived": self._survived,
            }
        )


@dataclass(frozen=True, slots=True, repr=False)
class BhyResult(_FdrResultBase):
    """Result of one BHY step-up for one metric.

    Shares ``metric_name`` / ``entries`` / ``adj_p_all`` / ``q`` /
    ``n_tests`` with :class:`_FdrResultBase`. ``adj_p_all`` is bucket-local;
    ``n_tests`` is keyed by ``expand_over_values`` tuple (``()`` for
    single-bucket).

    Attributes:
        expand_over: Keys used to partition the input into independent
            step-up buckets; empty tuple for the single-bucket case.
    """

    expand_over: tuple[str, ...]

    def _header(self) -> str:
        parts = [
            f"metric={self.metric_name}",
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


@dataclass(frozen=True, slots=True)
class MetricHypothesis:
    """One traceable ``EvaluationResult`` x metric hypothesis.

    ``is_active=False`` marks an ``insufficient_*`` data-shortage cell. The
    record remains auditable but is excluded from the BHY denominator.
    """

    result: EvaluationResult
    metric_name: str
    p_value: float
    identifier: tuple[Any, ...]
    expand_over_values: tuple[Any, ...]
    is_active: bool


@dataclass(frozen=True, slots=True, repr=False)
class CrossMetricBhyResult(_ScreenResultMixin):
    """One BHY screen over a pooled factor x metric hypothesis family."""

    entries: list[MetricHypothesis]
    adj_p_all: np.ndarray
    q: float
    metrics: tuple[str, ...]
    expand_over: tuple[str, ...]
    n_tests: Mapping[tuple[Any, ...], int]

    def to_frame(self) -> pl.DataFrame:
        """Return every tested cell, including inactive and eliminated rows."""
        return pl.DataFrame(
            {
                "factor": [e.result.factor for e in self.entries],
                "metric": [e.metric_name for e in self.entries],
                "p_value": [e.p_value for e in self.entries],
                "adj_p": self.adj_p_all,
                "survived": self._survived,
                "active": [e.is_active for e in self.entries],
            }
        )

    def _header(self) -> str:
        parts = [
            f"metrics={list(self.metrics)!r}",
            f"n={len(self.survivors)}",
            f"q={self.q:g}",
        ]
        if self.expand_over:
            parts.append(f"expand_over={list(self.expand_over)!r}")
        return ", ".join(parts)

    def _rows(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        if self.expand_over:
            headers: tuple[str, ...] = (
                "expand_over_values",
                "factor",
                "metric",
                "adj_p",
            )
            rows: list[tuple[str, ...]] = [
                (
                    repr(entry.expand_over_values),
                    entry.result.factor,
                    entry.metric_name,
                    f"{float(adj):.4g}",
                )
                for entry, adj in zip(self.survivors, self.adj_p, strict=True)
            ]
            return headers, rows
        headers = ("factor", "metric", "adj_p")
        rows = [
            (entry.result.factor, entry.metric_name, f"{float(adj):.4g}")
            for entry, adj in zip(self.survivors, self.adj_p, strict=True)
        ]
        return headers, rows


@dataclass(frozen=True, slots=True, repr=False)
class PartialConjunctionResult(_FdrResultBase):
    """Per-identity partial-conjunction survivors for one metric.

    Shares ``metric_name`` / ``entries`` / ``adj_p_all`` / ``q`` /
    ``n_tests`` with :class:`_FdrResultBase`. ``entries`` is one
    representative :class:`EvaluationResult` per identity; ``adj_p_all`` is
    the BHY-adjusted PC p-value; ``n_tests`` is the condition count per
    identity, keyed by the identifier with the ``expand_over`` components
    stripped (``factor``, then ``forward_periods`` and ``params`` items not
    named by ``expand_over``).

    Attributes:
        pc_p_all: Raw PC p-value aligned with ``entries``.
        expand_over: ``params`` keys defining the condition axis.
        min_pass: ``k`` in the ``k`` of ``m`` partial conjunction test.
        n_passed_uncorr_all: Per-identity count of raw p-values strictly
            below ``q`` (descriptive — not used in inference), aligned
            with ``entries``.
    """

    pc_p_all: np.ndarray
    expand_over: tuple[str, ...]
    min_pass: int
    n_passed_uncorr_all: np.ndarray

    def _header(self) -> str:
        return (
            f"metric={self.metric_name}, n={len(self.survivors)}, "
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
                str(
                    self.n_tests.get(
                        _hypothesis_identity(r, exclude=self.expand_over), 0
                    )
                ),
                str(int(n)),
            )
            for r, pc, adj, n, ok in zip(
                self.entries,
                self.pc_p_all,
                self.adj_p_all,
                self.n_passed_uncorr_all,
                self._survived,
                strict=True,
            )
            if ok
        ]
        return headers, rows


@dataclass(frozen=True, slots=True, repr=False)
class CrossMetricPartialConjunctionResult(_ScreenResultMixin):
    """Factor-level k-of-m metric confirmation followed by BHY."""

    entries: list[EvaluationResult]
    hypotheses: list[MetricHypothesis]
    adj_p_all: np.ndarray
    pc_p_all: np.ndarray
    q: float
    metrics: tuple[str, ...]
    min_pass: int
    n_tests: Mapping[tuple[Any, ...], int]
    n_active: Mapping[tuple[Any, ...], int]
    n_identities: int
    n_passed_uncorr_all: np.ndarray

    def to_frame(self) -> pl.DataFrame:
        """Return one row per factor identity, including ineligible rows."""
        identities = [_hypothesis_identity(entry) for entry in self.entries]
        return pl.DataFrame(
            {
                "factor": [entry.factor for entry in self.entries],
                "pc_p": self.pc_p_all,
                "adj_p": self.adj_p_all,
                "survived": self._survived,
                "active": [
                    self.n_active[identity] >= self.min_pass for identity in identities
                ],
                "n_tests": [self.n_tests[identity] for identity in identities],
                "n_active": [self.n_active[identity] for identity in identities],
                "n_passed_uncorr": self.n_passed_uncorr_all,
            }
        )

    def _header(self) -> str:
        return (
            f"metrics={list(self.metrics)!r}, n={len(self.survivors)}, "
            f"q={self.q:g}, min_pass={self.min_pass}"
        )

    def _rows(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        identities = [_hypothesis_identity(entry) for entry in self.entries]
        headers: tuple[str, ...] = (
            "factor",
            "pc_p",
            "adj_p",
            "n_tests",
            "n_passed_uncorr",
        )
        rows: list[tuple[str, ...]] = [
            (
                entry.factor,
                f"{float(pc):.4g}",
                f"{float(adj):.4g}",
                str(self.n_tests[identity]),
                str(int(n_passed)),
            )
            for entry, identity, pc, adj, n_passed, ok in zip(
                self.entries,
                identities,
                self.pc_p_all,
                self.adj_p_all,
                self.n_passed_uncorr_all,
                self._survived,
                strict=True,
            )
            if ok
        ]
        return headers, rows


@dataclass(frozen=True, slots=True, repr=False)
class HierarchicalBhyResult(_FdrResultBase):
    """Two-stage hierarchical BHY survivors for one metric.

    Shares ``metric_name`` / ``entries`` / ``adj_p_all`` / ``q`` /
    ``n_tests`` with :class:`_FdrResultBase`. ``adj_p_all`` is
    ``max(outer_adj_p[group], inner_adj_p[i])`` aligned with ``entries``
    so ``entry[i] survives iff adj_p_all[i] <= q``; ``q`` is shared by both
    layers; ``n_tests`` is the per-group inner family size keyed by
    ``(group_value,)`` — covering *all* input groups, not just survivors.

    Attributes:
        group: ``params`` key naming the group axis.
    """

    group: str

    def _header(self) -> str:
        return (
            f"metric={self.metric_name}, n={len(self.survivors)}, "
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
    return result.params.get(key)


def bhy(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    expand_over: tuple[str, ...] = (),
    q: float = 0.05,
) -> dict[str, BhyResult]:
    """Benjamini-Hochberg-Yekutieli step-up FDR, one screen per metric.

    The input list is treated as a single family. When ``expand_over``
    is non-empty, one independent step-up runs per unique tuple of
    ``expand_over`` values (read from ``EvaluationResult.forward_periods``
    for that built-in partition key, otherwise from
    ``EvaluationResult.params[k]``). Pooling horizons in one family is
    appropriate when selection may choose across horizons; partitioning by
    horizon is appropriate only for predeclared, separately reported screens.

    Args:
        results: :class:`EvaluationResult` records. The full input is
            one family unless ``expand_over`` further partitions it.
        metrics: ``list[str]`` — element type strictly
            :class:`metric label`. One independent BHY screen runs per
            metric; the return dict is keyed by ``label``.
            Single-metric callers still receive a one-key dict (no
            isinstance branching downstream).
        expand_over: Tuple of keys whose distinct value tuples split
            the input into independent BHY step-up buckets. Built-in
            field ``"forward_periods"`` is read off the result;
            other keys are looked up on ``result.params``.
        q: Nominal FDR target. Must satisfy ``0 < q < 1``.
            Default ``0.05``.

    Returns:
        ``dict[str, BhyResult]`` keyed by ``label``.

    Raises:
        UserInputError: ``metrics`` not a non-empty ``list[str]``;
            duplicate ``(factor, forward_periods, *params)``
            identifier; ``expand_over`` key missing from a result's
            ``params`` or naming ``'factor'``; metric absent from a result's
            outputs or its ``p_value`` missing / NaN.

    Warns:
        RuntimeWarning: Input pools multiple ``forward_periods`` in the same
            family, reminding callers to match the family to their selection
            rule. Or most ``expand_over`` buckets are
            singletons (BHY on n=1 provides no FDR correction).
    """
    metric_list = _validate_metric_list(metrics, func_name="bhy", field="metrics")
    q_target = _validate_q(q, func_name="bhy")
    _require_non_empty_results(results, func_name="bhy")
    expand_over_tuple = tuple(expand_over)

    _warn_on_mixed_horizons(results, func_name="bhy", expand_over=expand_over_tuple)

    partition = _partition(results, func_name="bhy", expand_over=expand_over_tuple)
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, p_entry in enumerate(partition):
        buckets[p_entry.expand_over_values].append(idx)

    out: dict[str, BhyResult] = {}
    for spec in metric_list:
        entries = _attach_p_values(partition, func_name="bhy", metric=spec)
        active_buckets: dict[tuple[Any, ...], list[int]] = {
            bucket_key: [
                i
                for i in ix
                if not _is_insufficient_short_circuit(entries[i].result, spec)
            ]
            for bucket_key, ix in buckets.items()
        }
        active_buckets = {k: ix for k, ix in active_buckets.items() if ix}
        n_tests = {bucket_key: len(ix) for bucket_key, ix in active_buckets.items()}
        singleton = sum(1 for ix in active_buckets.values() if len(ix) == 1)
        if singleton and len(active_buckets) > 1:
            warnings.warn(
                f"bhy: {singleton} of {len(active_buckets)} expand_over buckets "
                "contain a single result — BHY on n=1 is identical to a "
                "raw threshold and provides no FDR correction.",
                RuntimeWarning,
                stacklevel=2,
            )
        adj_p_all = np.full(len(entries), np.nan, dtype=np.float64)
        for ix in active_buckets.values():
            p_array = np.array([entries[i].p_value for i in ix], dtype=np.float64)
            adj_p_all[ix] = bhy_adjusted_p(p_array)

        out[spec] = BhyResult(
            metric_name=spec,
            entries=[e.result for e in entries],
            adj_p_all=adj_p_all,
            q=q_target,
            expand_over=expand_over_tuple,
            n_tests=n_tests,
        )
    return out


def _is_insufficient_short_circuit(result: EvaluationResult, metric: str) -> bool:
    """Return True when a metric output is a data-shortage placeholder."""
    reason = result.metrics[metric].metadata.get("reason")
    return isinstance(reason, str) and reason.startswith("insufficient_")


def _normalize_metric_hypotheses(
    results: Sequence[EvaluationResult],
    *,
    metrics: Sequence[str],
    func_name: str,
    expand_over: tuple[str, ...] = (),
) -> list[MetricHypothesis]:
    """Flatten results x metrics while preserving result-major order."""
    partition = _partition(results, func_name=func_name, expand_over=expand_over)
    attached = {
        metric: _attach_p_values(partition, func_name=func_name, metric=metric)
        for metric in metrics
    }
    hypotheses: list[MetricHypothesis] = []
    for idx, partition_entry in enumerate(partition):
        for metric in metrics:
            p_value = attached[metric][idx].p_value
            assert p_value is not None  # guaranteed by _attach_p_values
            hypotheses.append(
                MetricHypothesis(
                    result=partition_entry.result,
                    metric_name=metric,
                    p_value=float(p_value),
                    identifier=(*partition_entry.identifier, ("metric", metric)),
                    expand_over_values=partition_entry.expand_over_values,
                    is_active=not _is_insufficient_short_circuit(
                        partition_entry.result, metric
                    ),
                )
            )
    return hypotheses


def bhy_across_metrics(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    expand_over: tuple[str, ...] = (),
    q: float = 0.05,
) -> CrossMetricBhyResult:
    """Pool factor x metric hypotheses into one BHY FDR family.

    The survivor unit is one ``(EvaluationResult, metric label)`` cell. This
    controls hypothesis-level FDR; deduplicating survivors by factor does not
    create a factor-level FDR guarantee.

    Args:
        results: Unique :class:`EvaluationResult` hypotheses.
        metrics: At least two unique inferential metric labels, in the desired
            audit order.
        expand_over: Result fields or ``params`` keys that partition the input
            into separately reported families. Metrics stay pooled inside each
            bucket.
        q: Nominal FDR target in the open interval ``(0, 1)``.

    Returns:
        A :class:`CrossMetricBhyResult` containing every submitted cell.

    Raises:
        UserInputError: Inputs do not define a valid, traceable family or any
            non-short-circuited metric cell has an invalid p-value.
    """
    metric_list = _validate_cross_metric_list(metrics, func_name="bhy_across_metrics")
    q_target = _validate_q(q, func_name="bhy_across_metrics")
    _require_non_empty_results(results, func_name="bhy_across_metrics")
    expand_over_tuple = tuple(expand_over)
    _warn_on_mixed_horizons(
        results,
        func_name="bhy_across_metrics",
        expand_over=expand_over_tuple,
    )

    entries = _normalize_metric_hypotheses(
        results,
        metrics=metric_list,
        func_name="bhy_across_metrics",
        expand_over=expand_over_tuple,
    )
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        if entry.is_active:
            buckets[entry.expand_over_values].append(idx)

    n_tests = {bucket_key: len(ix) for bucket_key, ix in buckets.items()}
    singleton = sum(1 for ix in buckets.values() if len(ix) == 1)
    if singleton and len(buckets) > 1:
        warnings.warn(
            f"bhy_across_metrics: {singleton} of {len(buckets)} expand_over "
            "buckets contain a single active hypothesis; BHY on n=1 is "
            "identical to a raw threshold and provides no FDR correction.",
            RuntimeWarning,
            stacklevel=2,
        )

    adj_p_all = np.full(len(entries), np.nan, dtype=np.float64)
    for ix in buckets.values():
        p_array = np.array([entries[i].p_value for i in ix], dtype=np.float64)
        adj_p_all[ix] = bhy_adjusted_p(p_array)

    return CrossMetricBhyResult(
        entries=entries,
        adj_p_all=adj_p_all,
        q=q_target,
        metrics=tuple(metric_list),
        expand_over=expand_over_tuple,
        n_tests=n_tests,
    )


def partial_conjunction(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    min_pass: int,
    expand_over: tuple[str, ...],
    n_conditions: int | None = None,
    q: float = 0.05,
) -> dict[str, PartialConjunctionResult]:
    """Partial-conjunction screening: identities significant in
    ``min_pass`` of ``m`` conditions, one screen per metric.

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
        results: :class:`EvaluationResult` records. The aggregation
            identity is the hypothesis identifier minus the
            ``expand_over`` components; results sharing an identity form
            that identity's conditions and must differ on at least one
            ``expand_over`` key. Any other swept knob (``params`` key or
            ``forward_periods`` outside ``expand_over``) keeps
            identities apart instead of inflating ``m``.
        metrics: ``list[str]`` — one PC screen runs per metric;
            return dict keyed by ``label``.
        min_pass: ``k`` in "k of m". Must be ``>= 2``.
        expand_over: Non-empty tuple of ``params`` keys (or
            ``"forward_periods"``) defining the condition axis.
        n_conditions: Strict condition-count declaration. ``None`` lets ``m`` be
            inferred per identity; an ``int`` requires every identity
            to have exactly that many conditions.
        q: Nominal FDR target for the BHY step-up over PC p-values. Must
            satisfy ``0 < q < 1``.

    Returns:
        ``dict[str, PartialConjunctionResult]`` keyed by
        ``label``.

    Raises:
        UserInputError: ``min_pass < 2``; ``expand_over`` empty;
            ``n_conditions < min_pass``; condition-count mismatch;
            identity with fewer than ``min_pass`` conditions; any
            ``_resolve_family`` invariant failure.
    """
    metric_list = _validate_metric_list(
        metrics, func_name="partial_conjunction", field="metrics"
    )
    q_target = _validate_q(q, func_name="partial_conjunction")
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
                "non-empty tuple of params keys naming the condition "
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
    for spec in metric_list:
        out[spec] = _partial_conjunction_one(
            results,
            metric=spec,
            min_pass=min_pass,
            expand_over=expand_over_tuple,
            n_conditions=n_conditions,
            q=q_target,
        )
    return out


def _partial_conjunction_one(
    results: Sequence[EvaluationResult],
    *,
    metric: str,
    min_pass: int,
    expand_over: tuple[str, ...],
    n_conditions: int | None,
    q: float,
) -> PartialConjunctionResult:
    entries = _resolve_family(
        results,
        func_name="partial_conjunction",
        metric=metric,
        expand_over=expand_over,
    )

    entries_by_identity: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    identities_ordered: list[tuple[Any, ...]] = []
    for entry in entries:
        identity = _hypothesis_identity(entry.result, exclude=expand_over)
        if identity not in entries_by_identity:
            identities_ordered.append(identity)
        entries_by_identity[identity].append(entry)

    pc_p_arr = np.empty(len(identities_ordered), dtype=np.float64)
    n_passed_arr = np.empty(len(identities_ordered), dtype=np.int64)
    n_tests_per_id: dict[tuple[Any, ...], int] = {}
    rep_results: list[EvaluationResult] = []

    for i, identity in enumerate(identities_ordered):
        group = entries_by_identity[identity]
        m = len(group)

        if n_conditions is not None and m != n_conditions:
            raise UserInputError(
                func_name="partial_conjunction",
                field="n_conditions",
                value=n_conditions,
                expected=(
                    f"identity {identity!r} has {m} condition(s) in data but "
                    f"n_conditions={n_conditions} declared. "
                    "Pass n_conditions=None for inferred condition counts, or fix the "
                    f"input so every identity has exactly {n_conditions} "
                    "conditions"
                ),
                docs_path="api/partial-conjunction#strict-vs-lenient",
            )

        if m < min_pass:
            raise UserInputError(
                func_name="partial_conjunction",
                field="results",
                value=identity,
                expected=(
                    f"identity {identity!r} has only {m} condition(s) but "
                    f"min_pass={min_pass} requires at least that many. "
                    "Either drop this identity from the input or lower min_pass"
                ),
                docs_path="api/partial-conjunction#insufficient-conditions",
            )

        ps = np.array([e.p_value for e in group], dtype=np.float64)
        pc_p_arr[i] = partial_conjunction_p(ps, min_pass=min_pass)
        n_passed_arr[i] = int(np.sum(ps < q))
        n_tests_per_id[identity] = m
        rep_results.append(group[0].result)

    if n_conditions is None:
        m_values = set(n_tests_per_id.values())
        if len(m_values) > 1:
            warnings.warn(
                "partial_conjunction: inferred condition counts (n_conditions=None) "
                f"is running with heterogeneous condition counts "
                f"m={sorted(m_values)} across identities. PC p-values are "
                "valid marginally but the k/m bar differs per identity. "
                "For cross-identity comparability pass n_conditions=<int> "
                "(fixed condition count) or split the call.",
                RuntimeWarning,
                stacklevel=3,
            )

    adj_p_all = bhy_adjusted_p(pc_p_arr)
    return PartialConjunctionResult(
        metric_name=metric,
        entries=rep_results,
        adj_p_all=adj_p_all,
        pc_p_all=pc_p_arr,
        q=q,
        expand_over=expand_over,
        min_pass=min_pass,
        n_tests=n_tests_per_id,
        n_passed_uncorr_all=n_passed_arr,
    )


def partial_conjunction_across_metrics(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    min_pass: int,
    q: float = 0.05,
) -> CrossMetricPartialConjunctionResult:
    """Test whether each factor identity passes at least k of m metrics.

    Metric labels are the predeclared condition axis. For each result, the
    Bonferroni-style partial-conjunction p-value is computed across the fixed
    ``m=len(metrics)`` endpoints, then BHY runs across factor identities.
    ``insufficient_*`` endpoints are conservatively assigned p=1 rather than
    shrinking ``m``; identities with fewer than ``min_pass`` active endpoints
    remain in the audit output but do not enter the outer BHY family.

    Args:
        results: Unique :class:`EvaluationResult` hypotheses. Horizon and all
            ``params`` remain part of each factor identity.
        metrics: At least two unique, predeclared inferential metric labels.
        min_pass: ``k`` in the claim "at least k of m metrics carry signal".
            Must satisfy ``2 <= min_pass <= len(metrics)``.
        q: Nominal FDR target for BHY across factor identities.

    Returns:
        A :class:`CrossMetricPartialConjunctionResult` with factor-level
        survivors and the underlying metric hypotheses retained for audit.

    Raises:
        UserInputError: Inputs, metric p-values, or ``min_pass`` violate the
            declared k-of-m contract.
    """
    metric_list = _validate_cross_metric_list(
        metrics, func_name="partial_conjunction_across_metrics"
    )
    if isinstance(min_pass, bool) or not isinstance(min_pass, Integral):
        raise UserInputError(
            func_name="partial_conjunction_across_metrics",
            field="min_pass",
            value=min_pass,
            expected="integer satisfying 2 <= min_pass <= len(metrics)",
            docs_path="api/partial-conjunction-across-metrics#min-pass",
        )
    min_pass_int = int(min_pass)
    if not 2 <= min_pass_int <= len(metric_list):
        raise UserInputError(
            func_name="partial_conjunction_across_metrics",
            field="min_pass",
            value=min_pass,
            expected=(
                f"integer satisfying 2 <= min_pass <= len(metrics) ({len(metric_list)})"
            ),
            docs_path="api/partial-conjunction-across-metrics#min-pass",
        )

    q_target = _validate_q(q, func_name="partial_conjunction_across_metrics")
    _require_non_empty_results(results, func_name="partial_conjunction_across_metrics")
    _warn_on_mixed_horizons(
        results,
        func_name="partial_conjunction_across_metrics",
        expand_over=(),
        supports_expand_over=False,
    )
    hypotheses = _normalize_metric_hypotheses(
        results,
        metrics=metric_list,
        func_name="partial_conjunction_across_metrics",
    )

    m = len(metric_list)
    identifiers = [_hypothesis_identity(result) for result in results]
    n_tests = {identity: m for identity in identifiers}
    n_active: dict[tuple[Any, ...], int] = {}
    pc_p_all = np.full(len(results), np.nan, dtype=np.float64)
    n_passed = np.zeros(len(results), dtype=np.int64)
    eligible: list[int] = []

    for idx, identity in enumerate(identifiers):
        conditions = hypotheses[idx * m : (idx + 1) * m]
        active_count = sum(condition.is_active for condition in conditions)
        n_active[identity] = active_count
        n_passed[idx] = sum(
            condition.is_active and condition.p_value < q_target
            for condition in conditions
        )
        if active_count < min_pass_int:
            continue
        p_values = np.array(
            [
                condition.p_value if condition.is_active else 1.0
                for condition in conditions
            ],
            dtype=np.float64,
        )
        pc_p_all[idx] = partial_conjunction_p(p_values, min_pass=min_pass_int)
        eligible.append(idx)

    adj_p_all = np.full(len(results), np.nan, dtype=np.float64)
    if eligible:
        adj_p_all[eligible] = bhy_adjusted_p(pc_p_all[eligible])

    return CrossMetricPartialConjunctionResult(
        entries=list(results),
        hypotheses=hypotheses,
        adj_p_all=adj_p_all,
        pc_p_all=pc_p_all,
        q=q_target,
        metrics=tuple(metric_list),
        min_pass=min_pass_int,
        n_tests=n_tests,
        n_active=n_active,
        n_identities=len(eligible),
        n_passed_uncorr_all=n_passed,
    )


def bhy_hierarchical(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    group: str,
    q: float = 0.05,
) -> dict[str, HierarchicalBhyResult]:
    """Yekutieli (2008) two-stage hierarchical BHY, one screen per metric.

    Controls group-level FDR ≤ ``q`` on the outer layer (Simes group
    representative + BHY) and within-group FDR ≤ ``q`` on the inner
    layer (BHY restricted to passing groups). Flat BHY across the
    whole input loses group-level interpretability and pays full
    m-correction even when most groups are dead.

    Args:
        results: :class:`EvaluationResult` records. Each is assigned
            to one group via ``result.params[group]`` (or via
            ``result.forward_periods`` if ``group == "forward_periods"``).
            Within a group, each member is one hypothesis of the inner
            family; the ``(factor, forward_periods, *params)``
            identifier must be unique across the input.
        metrics: ``list[str]`` — one hierarchical screen per
            metric; return dict keyed by ``label``.
        group: Single key naming the group axis.
        q: Nominal FDR target shared by both layers. Must satisfy
            ``0 < q < 1``.

    Returns:
        ``dict[str, HierarchicalBhyResult]`` keyed by ``label``.

    Raises:
        UserInputError: ``group == 'factor'`` (it is the identifier);
            only one distinct group value in the input (call ``bhy``
            instead); every result is its own group at ``n >= 3``;
            duplicate ``(factor, forward_periods, *params)`` identifier; any
            ``_resolve_family`` invariant failure.

    Warns:
        RuntimeWarning: More than half of input groups contain a single
            result — inner BHY on n=1 is a raw cutoff and the outer
            Simes representative equals that single p-value.
    """
    metric_list = _validate_metric_list(
        metrics, func_name="bhy_hierarchical", field="metrics"
    )
    q_target = _validate_q(q, func_name="bhy_hierarchical")
    _require_non_empty_results(results, func_name="bhy_hierarchical")

    out: dict[str, HierarchicalBhyResult] = {}
    for spec in metric_list:
        out[spec] = _bhy_hierarchical_one(results, metric=spec, group=group, q=q_target)
    return out


def _bhy_hierarchical_one(
    results: Sequence[EvaluationResult],
    *,
    metric: str,
    group: str,
    q: float,
) -> HierarchicalBhyResult:
    entries = _resolve_family(
        results,
        func_name="bhy_hierarchical",
        metric=metric,
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
                "members. Call bhy(results, metrics=[...]) directly"
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

    return HierarchicalBhyResult(
        metric_name=metric,
        entries=[e.result for e in entries],
        adj_p_all=adj_p_all,
        q=q,
        group=group,
        n_tests=n_tests,
    )


def _warn_on_mixed_horizons(
    results: Sequence[EvaluationResult],
    *,
    func_name: str,
    expand_over: tuple[str, ...],
    supports_expand_over: bool = True,
) -> None:
    if "forward_periods" in expand_over:
        return
    horizons = {r.forward_periods for r in results}
    if len(horizons) > 1:
        hint = (
            "Use expand_over=('forward_periods',) only when horizon screens "
            "were predeclared and will be selected and reported separately; "
            "it does not control global horizon shopping."
            if supports_expand_over
            else (
                "For predeclared horizon-specific screens, filter the input by "
                "horizon and call the function separately; separate calls do "
                "not control later global horizon shopping."
            )
        )
        warnings.warn(
            f"{func_name}: input mixes forward_periods={sorted(horizons)} but "
            "they are being pooled into one multiple-testing family. This is "
            "the correct choice when the research process may select across "
            f"horizons. {hint}",
            RuntimeWarning,
            stacklevel=3,
        )
