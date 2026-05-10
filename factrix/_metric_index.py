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
from typing import Literal

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
_INFRASTRUCTURE: frozenset[str] = frozenset({"by_regime", "by_slice"})
"""Cross-cutting dispatchers published from ``factrix.metrics`` that
are not per-(scope, signal) cell metrics."""

# Scalar-input metrics: pre-aggregated-scalar utilities that consume
# scalars (``gross_spread: float``, ``turnover: float``, ...) rather
# than the standard ``(date-keyed DataFrame, **kwargs) -> MetricOutput``
# panel contract. Not eligible for date-slicing dispatchers like
# ``by_regime``. Hard-coded exception set rather than a Matrix-row tag
# extension — the population is small and stable; YAGNI applies until
# it isn't.
_SCALAR_INPUT_METRICS: frozenset[str] = frozenset({"breakeven_cost", "net_spread"})

# Metrics that ``run_metrics`` cannot auto-discover even though their
# Matrix-row is user-facing. Each entry carries a short reason rendered
# into ``MetricsBundle.skipped`` and into the ``UserInputError`` raised
# when a caller names one explicitly via ``metrics=[...]``. v2 may
# promote this to a Matrix-row tag once the population is stable.
_AUTO_DISCOVER_EXCLUDED: dict[str, str] = {
    # Horizon dispatchers — semantics live at the horizon-loop layer
    # (#186 deprecation in favour of `compare(bundles[])` /
    # `bhy(expand_over=["forward_periods"])`).
    "multi_horizon_ic": (
        "horizon dispatcher; use compare(bundles[]) or "
        "bhy(expand_over=['forward_periods'])"
    ),
    "multi_horizon_hit_rate": (
        "horizon dispatcher; use compare(bundles[]) or "
        "bhy(expand_over=['forward_periods'])"
    ),
    # Stage-1 consumers without v1 dispatch wiring. These metrics are
    # callable directly from ``factrix.metrics`` once the caller has the
    # stage-1 frame; v1 ``run_metrics`` does not yet thread that wiring.
    # Tracked as v1.x follow-up; not blocking the IC-cell stage-1 pattern
    # validated in v1.
    "caar": (
        "consumes caar_df; call factrix.metrics.compute_event_returns + "
        "compute_caar then caar(caar_df) directly"
    ),
    "fama_macbeth": (
        "consumes beta_df; call factrix.metrics.compute_fm_betas then "
        "fama_macbeth(beta_df) directly"
    ),
    "beta_sign_consistency": (
        "consumes beta_df; call factrix.metrics.compute_fm_betas then "
        "beta_sign_consistency(beta_df) directly"
    ),
    "ts_beta": (
        "consumes ts_betas_df; call factrix.metrics.compute_ts_betas then "
        "ts_beta(ts_betas_df) directly"
    ),
    "mean_r_squared": (
        "consumes ts_betas_df; call factrix.metrics.compute_ts_betas then "
        "mean_r_squared(ts_betas_df) directly"
    ),
    "ts_beta_sign_consistency": (
        "consumes ts_betas_df; call factrix.metrics.compute_ts_betas then "
        "ts_beta_sign_consistency(ts_betas_df) directly"
    ),
    "mfe_mae_summary": (
        "consumes mfe_mae_df; call factrix.metrics.compute_mfe_mae then "
        "mfe_mae_summary(mfe_mae_df) directly"
    ),
    # Series-input diagnostics — operate on a (date, value) series, not
    # a panel. Compose explicitly with compute_ic / compute_spread_series.
    "hit_rate": (
        "consumes a (date, value) series; e.g. "
        "hit_rate(compute_ic(panel).rename({'ic': 'value'}))"
    ),
    "multi_split_oos_decay": (
        "consumes a (date, value) series; e.g. "
        "multi_split_oos_decay(compute_spread_series(panel)"
        ".rename({'spread': 'value'}))"
    ),
    "ic_trend": (
        "consumes a (date, value) series; e.g. "
        "ic_trend(compute_ic(panel).rename({'ic': 'value'}))"
    ),
    # Multi-factor spread inputs — require a list/dict of factor spreads,
    # not a single panel.
    "spanning_alpha": (
        "consumes factor_spread; call after compute_spread_series on the "
        "challenger factor"
    ),
    "greedy_forward_selection": (
        "consumes factor_spreads dict; call after compute_spread_series on "
        "each candidate factor"
    ),
}

# Function name → emitted ``MetricOutput.name`` literal, where the two
# diverge. Most metrics emit ``name=<function_name>``; the entries here
# are the historical exceptions and the source of truth for the
# ``MetricOutput.name`` reverse index (#125). Audit against
# ``factrix/metrics/*.py``: every ``MetricOutput(name="X")`` literal in
# a registered metric must either match the function name or appear here.
_EMITTED_NAME_OVERRIDES: dict[str, str] = {
    "clustering_diagnostic": "clustering_hhi",
    "corrado_rank_test": "corrado_rank",
    "fama_macbeth": "fm_beta",
    "multi_split_oos_decay": "oos_decay",
    "pooled_ols": "pooled_beta",
}

# Canonical reverse-index URL convention for ``MetricRow.docs_anchor``
# (#125): docs-root-relative path + mkdocstrings symbol fragment.
# Centralised here so a future docs URL change touches one literal —
# external prose in ``factrix._describe.list_metrics`` and
# ``docs/api/metric-output.md`` cite this constant by name rather
# than restating the string.
DOCS_ANCHOR_FMT: str = "api/metrics/{module}.md#factrix.metrics.{module}.{name}"


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
    """One ``(metric_name, module, cell)`` triple parsed from ``Matrix-row:``.

    ``import_path`` is the fully-qualified submodule (``factrix.metrics.<module>``)
    — every public metric is also re-exported from ``factrix.metrics``,
    so ``from factrix.metrics import <name>`` works regardless. ``input_kind``
    distinguishes the standard date-keyed-DataFrame contract (``"panel"``)
    from pre-aggregated-scalar utilities (``"scalar"``); only ``"panel"``
    metrics are eligible for date-slicing dispatchers like ``by_regime``.
    ``docs_anchor`` is the docs-root-relative path + mkdocstrings symbol
    anchor (``api/metrics/<module>.md#factrix.metrics.<module>.<name>``),
    so a consumer holding the function name can resolve the API page
    without scraping module conventions. ``emitted_name`` is the literal
    string the metric writes into ``MetricOutput.name`` at runtime; for
    most metrics this matches ``name``, but a small set of historical
    exceptions (e.g. ``fama_macbeth`` → ``fm_beta``) emit a different
    label. Consumers holding a ``MetricOutput`` value should look up
    by ``emitted_name``.
    """

    name: str
    module: str
    cell: Cell
    agg_order: str
    inference_se: str
    import_path: str
    input_kind: Literal["panel", "scalar"]
    docs_anchor: str
    emitted_name: str


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


@functools.cache
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
            import_path=f"factrix.metrics.{entry.module}",
            input_kind="scalar" if name in _SCALAR_INPUT_METRICS else "panel",
            docs_anchor=DOCS_ANCHOR_FMT.format(module=entry.module, name=name),
            emitted_name=_EMITTED_NAME_OVERRIDES.get(name, name),
        )
        for entry in matrix_entries()
        for name in entry.names
    ]
    rows.sort(key=lambda r: (r.module, r.name))
    return rows


@functools.cache
def user_facing_rows() -> list[MetricRow]:
    """Return parsed rows excluding stage-1 helpers and cross-cutting infra."""
    excluded = _STAGE1_HELPERS | _INFRASTRUCTURE
    return [r for r in all_rows() if r.name not in excluded]
