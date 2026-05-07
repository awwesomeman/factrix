"""Shared parser for ``Matrix-row:`` tags in ``factrix/metrics/*.py``.

Single source of truth for "which standalone metrics live in which
cell." Consumed by:

- ``scripts/mkdocs_hooks/gen_metric_matrix.py`` — renders the docs
  matrix in ``docs/reference/_generated_metric_matrix.md``.
- ``factrix.list_metrics`` — runtime API exposing the same data.

Each metric module's docstring carries one or more ``Matrix-row:``
lines with the format::

    Matrix-row: {public_functions} | {cell_scope} | {agg_order} | {inference_se} | {primitives}

``public_functions`` is a comma-separated list. ``cell_scope`` is
either a 4-tuple ``(scope, signal, metric, mode)`` (``*`` denotes
wildcard) or — uniquely for ``factrix.metrics.spanning`` — the
literal label ``factor-return-series consumer (post-PANEL pipeline)``,
which is mapped to ``(INDIVIDUAL, CONTINUOUS, *, *)`` since spanning
metrics consume the spread series produced by the Individual ×
Continuous pipeline.

Stage-1 aggregation helpers that produce intermediate panels / series
(rather than a ``MetricOutput``) are filtered out by ``user_facing``
rows — they are listed in ``Matrix-row:`` for primitive-graph
completeness but are not metrics a user would call directly. The
exclusion set is explicit (not a ``compute_*`` prefix rule) because
``compute_rolling_mean_beta`` is a genuine user-facing metric.
"""

from __future__ import annotations

import ast
import functools
import pathlib
import re
from dataclasses import dataclass

from factrix._axis import FactorScope, Metric, Mode, Signal

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_METRICS_DIR = _REPO_ROOT / "factrix" / "metrics"

_MATRIX_ROW_RE = re.compile(r"^\s*Matrix-row:\s*(.+)$", re.MULTILINE)
_TUPLE_RE = re.compile(r"^\((.*)\)$")

_SPANNING_CELL_LABEL = "factor-return-series consumer (post-PANEL pipeline)"

# Stage-1 helpers: produce intermediate panels / series consumed by the
# user-facing metric in the same module. Listed in ``Matrix-row:`` for
# primitive-graph completeness; excluded from ``user_facing`` rows.
_STAGE1_HELPERS: frozenset[str] = frozenset(
    {
        "compute_caar",
        "compute_event_returns",
        "compute_fm_betas",
        "compute_group_returns",
        "compute_ic",
        "compute_mfe_mae",
        "compute_spread_series",
        "compute_ts_betas",
        # Single-asset fallback wired into the dispatch registry, not a
        # standalone callable (``ts_beta`` itself dispatches to it).
        "ts_beta_single_asset_fallback",
    }
)

# Cross-cutting infrastructure published from ``factrix.metrics`` that
# is **not** a per-(scope, signal) cell metric — e.g. ``by_regime`` is a
# dispatcher that wraps any registered metric. Listed in ``Matrix-row:``
# with ``(*, *, *, *)`` so the primitive graph is complete, but excluded
# from per-cell ``list_metrics`` output and the applicability table
# (which catalogue per-cell metrics, not infra).
_INFRASTRUCTURE: frozenset[str] = frozenset({"by_regime"})
"""Cross-cutting dispatchers published from ``factrix.metrics`` that
are not per-(scope, signal) cell metrics."""


@dataclass(frozen=True, slots=True)
class Cell:
    """Parsed ``(scope, signal, metric, mode)`` cell tuple.

    ``None`` represents the ``*`` wildcard along an axis. ``raw``
    preserves the original string for matrix rendering. ``metric`` and
    ``mode`` are parsed for completeness — :meth:`matches` only filters
    on ``scope`` / ``signal`` (the public ``list_metrics`` axes); the
    matrix renderer reads ``raw`` directly.
    """

    scope: FactorScope | None
    signal: Signal | None
    metric: Metric | None
    mode: Mode | None
    raw: str

    def matches(self, scope: FactorScope, signal: Signal) -> bool:
        """Return True if this cell is applicable to ``(scope, signal)``."""
        return (self.scope is None or self.scope == scope) and (
            self.signal is None or self.signal == signal
        )


@dataclass(frozen=True, slots=True)
class MatrixEntry:
    """One ``Matrix-row:`` tag, un-exploded — drives matrix rendering."""

    module: str
    names: tuple[str, ...]
    cell: Cell
    agg_order: str
    inference_se: str


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One ``(metric_name, module, cell)`` triple parsed from ``Matrix-row:``."""

    name: str
    module: str
    cell: Cell
    agg_order: str
    inference_se: str


def _parse_axis(token: str, enum_cls: type) -> object | None:
    """Map a Matrix-row axis token (``*`` or enum value) to enum or None."""
    token = token.strip()
    if token == "*":
        return None
    try:
        return enum_cls(token.lower())
    except ValueError as exc:
        raise ValueError(
            f"unknown {enum_cls.__name__} token {token!r} in Matrix-row cell"
        ) from exc


def _parse_cell(raw: str) -> Cell:
    raw = raw.strip()
    if raw == _SPANNING_CELL_LABEL:
        return Cell(
            scope=FactorScope.INDIVIDUAL,
            signal=Signal.CONTINUOUS,
            metric=None,
            mode=None,
            raw=raw,
        )
    m = _TUPLE_RE.match(raw)
    if not m:
        raise ValueError(f"unparseable Matrix-row cell: {raw!r}")
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Matrix-row cell tuple has {len(parts)} fields (expected 4): {raw!r}"
        )
    return Cell(
        scope=_parse_axis(parts[0], FactorScope),  # type: ignore[arg-type]
        signal=_parse_axis(parts[1], Signal),  # type: ignore[arg-type]
        metric=_parse_axis(parts[2], Metric),  # type: ignore[arg-type]
        mode=_parse_axis(parts[3], Mode),  # type: ignore[arg-type]
        raw=raw,
    )


def _public_metric_modules() -> list[pathlib.Path]:
    return sorted(p for p in _METRICS_DIR.glob("*.py") if not p.stem.startswith("_"))


def _extract_matrix_rows(path: pathlib.Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    doc = ast.get_docstring(tree) or ""
    return [m.group(1).strip() for m in _MATRIX_ROW_RE.finditer(doc)]


def _parse_module_entries(path: pathlib.Path) -> list[MatrixEntry]:
    entries: list[MatrixEntry] = []
    for raw_value in _extract_matrix_rows(path):
        parts = [p.strip() for p in raw_value.split("|")]
        if len(parts) != 5:
            raise ValueError(
                f"{path.name}: Matrix-row has {len(parts)} pipe-separated"
                f" fields (expected 5): {raw_value!r}"
            )
        names_csv, cell_str, agg_order, inference_se, _primitives = parts
        names = tuple(n for n in (n.strip() for n in names_csv.split(",")) if n)
        entries.append(
            MatrixEntry(
                module=path.stem,
                names=names,
                cell=_parse_cell(cell_str),
                agg_order=agg_order,
                inference_se=inference_se,
            )
        )
    return entries


@functools.cache
def matrix_entries() -> tuple[MatrixEntry, ...]:
    """Return one entry per ``Matrix-row:`` tag (un-exploded by name).

    Sorted by module. Cached — metric module docstrings do not change
    at runtime, so repeated callers (``list_metrics`` in agentic loops)
    avoid re-parsing every public ``factrix/metrics/*.py``.
    """
    out: list[MatrixEntry] = []
    for path in _public_metric_modules():
        out.extend(_parse_module_entries(path))
    out.sort(key=lambda e: e.module)
    return tuple(out)


def all_rows() -> list[MetricRow]:
    """Return every parsed ``Matrix-row:`` row, exploded one per metric name.

    Sorted by ``(module, name)``. Includes stage-1 helpers; for the
    user-facing subset call :func:`user_facing_rows`.
    """
    rows = [
        MetricRow(
            name=name,
            module=entry.module,
            cell=entry.cell,
            agg_order=entry.agg_order,
            inference_se=entry.inference_se,
        )
        for entry in matrix_entries()
        for name in entry.names
    ]
    rows.sort(key=lambda r: (r.module, r.name))
    return rows


def user_facing_rows() -> list[MetricRow]:
    """Return parsed rows excluding stage-1 helpers and cross-cutting infra."""
    excluded = _STAGE1_HELPERS | _INFRASTRUCTURE
    return [r for r in all_rows() if r.name not in excluded]
