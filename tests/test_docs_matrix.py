"""Coverage test: Matrix-row: tags in factrix/metrics/ docstrings.

Validates that:
1. Every public metric module (non-underscore *.py) has at least one
   ``Matrix-row:`` tag in its module-level docstring.
2. Every ``Matrix-row:`` tag has exactly 5 pipe-separated fields
   (public_functions | cell_scope | aggregation_order | inference_se | primitives).
3. ``docs/reference/_generated_metric_matrix.md`` exists and is non-empty
   (only meaningful after a build; skipped if the file is absent).

Decision rationale: ARCHITECTURE.md §Docs SSOT strategy (Option B).
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

METRICS_DIR = pathlib.Path("factrix/metrics")
GENERATED_MATRIX = pathlib.Path("docs/reference/_generated_metric_matrix.md")

_MATRIX_ROW_RE = re.compile(r"^\s*Matrix-row:\s*(.+)$", re.MULTILINE)


def _public_metric_modules() -> set[str]:
    """Return stem names of all public metric modules."""
    return {
        p.stem
        for p in METRICS_DIR.glob("*.py")
        if not p.stem.startswith("_")
    }


def _matrix_rows_for(stem: str) -> list[str]:
    """Return list of raw Matrix-row values found in the module's docstring."""
    path = METRICS_DIR / f"{stem}.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    doc = ast.get_docstring(tree) or ""
    return [m.group(1).strip() for m in _MATRIX_ROW_RE.finditer(doc)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", sorted(_public_metric_modules()))
def test_module_has_matrix_row_tag(stem: str) -> None:
    """Every public metric module must have at least one Matrix-row: tag."""
    rows = _matrix_rows_for(stem)
    assert rows, (
        f"metrics/{stem}.py has no 'Matrix-row:' tag in its module docstring. "
        "Add a Matrix-row: line so the build-time generator can include it."
    )


@pytest.mark.parametrize("stem", sorted(_public_metric_modules()))
def test_matrix_row_has_five_fields(stem: str) -> None:
    """Every Matrix-row: tag must have exactly 5 pipe-separated fields."""
    for row in _matrix_rows_for(stem):
        fields = [f.strip() for f in row.split("|")]
        assert len(fields) == 5, (
            f"metrics/{stem}.py: Matrix-row has {len(fields)} pipe-separated"
            f" fields (expected 5): {row!r}"
        )


def test_generated_matrix_exists_and_nonempty() -> None:
    """Generated matrix file must exist and contain at least one data row."""
    if not GENERATED_MATRIX.exists():
        pytest.skip(
            f"{GENERATED_MATRIX} not found — run "
            "'python scripts/gen_metric_matrix.py' or 'mkdocs build' first."
        )
    lines = [
        ln
        for ln in GENERATED_MATRIX.read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith("|") and not ln.strip().startswith("|---")
    ]
    # At minimum: header row + at least one data row
    assert len(lines) >= 2, (
        f"{GENERATED_MATRIX} appears empty (only {len(lines)} table line(s))."
    )
