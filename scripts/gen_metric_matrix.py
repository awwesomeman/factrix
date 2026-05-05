"""Build-time generator for the standalone-metrics matrix table.

Parses every public ``factrix/metrics/*.py`` module, extracts
``Matrix-row:`` lines from each module docstring, and writes a
Markdown table to ``docs/reference/_generated_metric_matrix.md``.

Usage (manual)::

    python scripts/gen_metric_matrix.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import ast
import pathlib
import re

# ---------------------------------------------------------------------------
# Paths (relative to repo root — resolved at call time so the script works
# when invoked from any working directory, including inside mkdocs).
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_METRICS_DIR = _REPO_ROOT / "factrix" / "metrics"
_OUT_FILE = _REPO_ROOT / "docs" / "reference" / "_generated_metric_matrix.md"

_GITHUB_BASE = "https://github.com/awwesomeman/factrix/blob/main/"

_TABLE_HEADER = (
    "| Module | Cell scope | Aggregation order | Inference SE |\n"
    "|---|---|---|---|\n"
)

_MATRIX_ROW_RE = re.compile(r"^\s*Matrix-row:\s*(.+)$", re.MULTILINE)


def _public_metric_modules() -> list[pathlib.Path]:
    """Return sorted list of public metric module paths."""
    return sorted(
        p
        for p in _METRICS_DIR.glob("*.py")
        if not p.stem.startswith("_")
    )


def _extract_matrix_rows(path: pathlib.Path) -> list[str]:
    """Return raw Matrix-row values (one per tag) from a module's docstring."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    doc = ast.get_docstring(tree) or ""
    return [m.group(1).strip() for m in _MATRIX_ROW_RE.finditer(doc)]



def _build_table_row(module_path: pathlib.Path, row_value: str) -> str:
    """Convert one Matrix-row: value into a Markdown table row string.

    Renders 4 user-facing columns (Module | Cell scope | Aggregation order |
    Inference SE); public_functions and primitives are kept in the docstring
    for developers but omitted from the overview table.
    """
    parts = [p.strip() for p in row_value.split("|")]
    if len(parts) != 5:
        raise ValueError(
            f"{module_path.name}: Matrix-row has {len(parts)} pipe-separated"
            f" fields (expected 5): {row_value!r}"
        )
    _public_fns, cell_scope, agg_order, inference_se, _primitives = parts

    stem = module_path.stem
    github_url = _GITHUB_BASE + f"factrix/metrics/{stem}.py"
    module_cell = f"[`metrics/{stem}.py`]({github_url})"

    return (
        f"| {module_cell} | `{cell_scope}` | {agg_order} | {inference_se} |\n"
    )


def generate() -> None:
    """Generate ``_generated_metric_matrix.md`` from module docstrings."""
    lines: list[str] = [_TABLE_HEADER]

    for module_path in _public_metric_modules():
        row_values = _extract_matrix_rows(module_path)
        if not row_values:
            # Skip modules with no Matrix-row tags silently; test coverage
            # in test_docs_matrix.py will catch missing tags.
            continue
        for row_value in row_values:
            lines.append(_build_table_row(module_path, row_value))

    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _OUT_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"gen_metric_matrix: wrote {len(lines) - 1} row(s) to {_OUT_FILE}")


# ---------------------------------------------------------------------------
# MkDocs hook entry point (mkdocs 1.4+)
# ---------------------------------------------------------------------------


def on_pre_build(config: object) -> None:  # noqa: ARG001
    """MkDocs hook: regenerate the matrix before every build."""
    generate()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate()
