"""Build-time generator for the ``MetricOutput.name`` reverse index.

Renders a Markdown table to
``docs/reference/_generated_metric_name_index.md`` mapping every
emitted ``MetricOutput.name`` value back to its source module, import
path, and API-page anchor. Driven by the same ``Matrix-row:`` SSOT as
``gen_metric_matrix.py`` and ``factrix.list_metrics``.

Closes the asymmetry called out in #125: a consumer holding a
``MetricOutput`` value (in a downstream pipeline, an agent loop, a
report renderer) can mechanically resolve ``output.name`` to the
metric's docs page without grepping the codebase or guessing URL
conventions.

Usage (manual)::

    python scripts/mkdocs_hooks/gen_metric_name_index.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib

from factrix._metric_index import (
    MetricSpec,
    docs_anchor_for,
    import_path_for,
    public_specs,
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUT_FILE = _REPO_ROOT / "docs" / "reference" / "_generated_metric_name_index.md"

_TABLE_HEADER = "| `MetricOutput.name` | Source module | API page |\n|---|---|---|\n"


def _render_row(stem: str, spec: MetricSpec) -> str:
    import_path = import_path_for(stem)
    page_link = f"[`{spec.name}`](../{docs_anchor_for(stem, spec.name)})"
    module_link = f"[`{import_path}`][{import_path}]"
    return f"| `{spec.name}` | {module_link} | {page_link} |\n"


def generate() -> None:
    """Generate ``_generated_metric_name_index.md`` from MetricSpec SSOT."""
    rows = sorted(public_specs(), key=lambda pair: pair[1].name)
    lines = [_TABLE_HEADER, *(_render_row(stem, spec) for stem, spec in rows)]
    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _OUT_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"gen_metric_name_index: wrote {len(rows)} row(s) to {_OUT_FILE}")


def on_pre_build(config: object) -> None:
    """MkDocs hook: regenerate the name index before every build."""
    generate()


if __name__ == "__main__":
    generate()
