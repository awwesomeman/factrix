"""Build-time generator for the registry-cell table.

Renders the dispatch cells from ``factrix._registry._DISPATCH_REGISTRY``
into ``docs/development/_generated_registry_cells.md`` so the
architecture page stays in lockstep with code. Mirrors the
``gen_metric_matrix.py`` pattern: code is SSOT, the docs file is a
generated artifact included via ``pymdownx.snippets``.

Usage (manual)::

    python scripts/mkdocs_hooks/gen_registry_cells.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib

from factrix._axis import FactorScope
from factrix._registry import (
    _DISPATCH_REGISTRY,
    _DispatchKey,
    _RegistryEntry,
    _ScopeCollapsedSentinel,
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUT_FILE = _REPO_ROOT / "docs" / "development" / "_generated_registry_cells.md"

_TABLE_HEADER = "| `(scope, signal, metric, mode)` | Procedure class |\n|---|---|\n"


def _scope_name(scope: FactorScope | _ScopeCollapsedSentinel) -> str:
    if isinstance(scope, _ScopeCollapsedSentinel):
        return "_SCOPE_COLLAPSED"
    return scope.name


def _render_row(key: _DispatchKey, entry: _RegistryEntry) -> str:
    scope = _scope_name(key.scope)
    signal = key.signal.name
    metric = key.metric.name if key.metric is not None else "None"
    mode = key.mode.name
    cls = type(entry.procedure).__name__
    return f"| `({scope}, {signal}, {metric}, {mode})` | `{cls}` |\n"


def generate() -> None:
    """Generate ``_generated_registry_cells.md`` from the live registry."""
    rows = [_render_row(k, e) for k, e in _DISPATCH_REGISTRY.items()]
    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _OUT_FILE.write_text(_TABLE_HEADER + "".join(rows), encoding="utf-8")
    print(f"gen_registry_cells: wrote {len(rows)} row(s) to {_OUT_FILE}")


def on_pre_build(config: object) -> None:
    """MkDocs hook: regenerate the table before every build."""
    generate()


if __name__ == "__main__":
    generate()
