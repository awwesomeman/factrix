"""Build-time generator for the standalone-metrics matrix table.

Renders a Markdown table to ``docs/reference/_generated_metric_matrix.md``
from the typed metric specs exposed by :mod:`factrix._metric_index`
(single source of truth, also consumed at runtime by
``factrix.list_metrics``).

One table row per distinct ``(module, cell.raw, aggregation)`` group —
modules whose specs cover multiple cells (e.g. ``tradability`` splitting
its panel-aggregated and rank-autocorrelation metrics across two rows)
surface as multiple rows.

Usage (manual)::

    python scripts/mkdocs_hooks/gen_metric_matrix.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from factrix._metric_index import MetricSpec, public_specs

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUT_FILE = _REPO_ROOT / "docs" / "reference" / "_generated_metric_matrix.md"

_TABLE_HEADER = "| Module | Cell scope | Aggregation order |\n|---|---|---|\n"


@dataclass(frozen=True, slots=True)
class _GroupedRow:
    """One docs-matrix row: a ``(module, cell, aggregation)`` group."""

    module: str
    cell_raw: str
    aggregation: str


def _group_specs(specs: tuple[tuple[str, MetricSpec], ...]) -> list[_GroupedRow]:
    """Collapse per-callable specs into per-(module, cell, aggregation) rows.

    Stable on insertion order so the rendered table follows the
    spec-declaration order in each module.
    """
    seen: dict[tuple[str, str, str], _GroupedRow] = {}
    for stem, spec in specs:
        key = (stem, spec.cell.raw, spec.aggregation.name)
        if key not in seen:
            seen[key] = _GroupedRow(
                module=stem,
                cell_raw=spec.cell.raw,
                aggregation=spec.aggregation.value,
            )
    return list(seen.values())


def _render_row(row: _GroupedRow) -> str:
    module_cell = f"[`metrics.{row.module}`][factrix.metrics.{row.module}]"
    return f"| {module_cell} | `{row.cell_raw}` | {row.aggregation} |\n"


def generate() -> None:
    """Generate ``_generated_metric_matrix.md`` from the metric specs."""
    rows = _group_specs(public_specs())
    lines = [_TABLE_HEADER, *(_render_row(r) for r in rows)]
    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _OUT_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"gen_metric_matrix: wrote {len(rows)} row(s) to {_OUT_FILE}")


def on_pre_build(config: object) -> None:
    """MkDocs hook: regenerate the matrix before every build."""
    generate()


if __name__ == "__main__":
    generate()
