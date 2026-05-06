"""Build-time generator for the standalone-metrics matrix table.

Renders a Markdown table to ``docs/reference/_generated_metric_matrix.md``
from the parsed ``Matrix-row:`` tags exposed by
:mod:`factrix._metric_index` (single source of truth, also consumed at
runtime by ``factrix.list_metrics``).

Usage (manual)::

    python scripts/mkdocs_hooks/gen_metric_matrix.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib

from factrix._metric_index import MatrixEntry, matrix_entries

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUT_FILE = _REPO_ROOT / "docs" / "reference" / "_generated_metric_matrix.md"

_TABLE_HEADER = (
    "| Module | Cell scope | Aggregation order | Inference SE |\n|---|---|---|---|\n"
)


def _render_row(entry: MatrixEntry) -> str:
    module_cell = f"[`metrics.{entry.module}`][factrix.metrics.{entry.module}]"
    return (
        f"| {module_cell} | `{entry.cell.raw}` | "
        f"{entry.agg_order} | {entry.inference_se} |\n"
    )


def generate() -> None:
    """Generate ``_generated_metric_matrix.md`` from module docstrings."""
    entries = matrix_entries()
    lines = [_TABLE_HEADER, *(_render_row(e) for e in entries)]
    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _OUT_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"gen_metric_matrix: wrote {len(entries)} row(s) to {_OUT_FILE}")


def on_pre_build(config: object) -> None:
    """MkDocs hook: regenerate the matrix before every build."""
    generate()


if __name__ == "__main__":
    generate()
