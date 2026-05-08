"""Build-time generator for the cell -> ``evaluate()``-metric reference table.

Renders a Markdown table to
``docs/reference/_generated_evaluate_metric_table.md`` from
``factrix._registry._DISPATCH_REGISTRY`` and the
``factrix._metric_index.user_facing_rows`` import-path index.

Usage (manual)::

    python scripts/mkdocs_hooks/gen_evaluate_metric_table.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib

from factrix._metric_index import user_facing_rows
from factrix._registry import (
    _DISPATCH_REGISTRY,
    _SCOPE_COLLAPSED,
    _DispatchKey,
    _RegistryEntry,
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUT_FILE = _REPO_ROOT / "docs" / "reference" / "_generated_evaluate_metric_table.md"

_TABLE_HEADER = (
    "| Cell `(scope, signal, metric, mode)` | Run by `evaluate()` | "
    "Procedure summary | Literature |\n|---|---|---|---|\n"
)


def _format_cell(key: _DispatchKey) -> str:
    scope = "*" if key.scope is _SCOPE_COLLAPSED else key.scope.name
    metric = key.metric.name if key.metric is not None else "*"
    return f"({scope}, {key.signal.name}, {metric}, {key.mode.name})"


def _render_row(entry: _RegistryEntry, import_path_by_name: dict[str, str]) -> str:
    cell = _format_cell(entry.key)
    name = entry.evaluate_metric_name
    metric = f"[`{name}`][{import_path_by_name[name]}]"
    refs = ", ".join(entry.references) if entry.references else "—"
    return f"| `{cell}` | {metric} | {entry.canonical_use_case} | {refs} |\n"


def generate() -> None:
    """Generate ``_generated_evaluate_metric_table.md`` from the registry."""
    import_path_by_name = {row.name: row.import_path for row in user_facing_rows()}
    entries = list(_DISPATCH_REGISTRY.values())
    lines = [_TABLE_HEADER, *(_render_row(e, import_path_by_name) for e in entries)]
    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _OUT_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"gen_evaluate_metric_table: wrote {len(entries)} row(s) to {_OUT_FILE}")


def on_pre_build(config: object) -> None:
    """MkDocs hook: regenerate the evaluate-metric table before every build."""
    generate()


if __name__ == "__main__":
    generate()
