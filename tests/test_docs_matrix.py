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
GENERATED_NAME_INDEX = pathlib.Path("docs/reference/_generated_metric_name_index.md")
GENERATED_EVALUATE_METRIC = pathlib.Path(
    "docs/reference/_generated_evaluate_metric_table.md"
)

_MATRIX_ROW_RE = re.compile(r"^\s*Matrix-row:\s*(.+)$", re.MULTILINE)


def _public_metric_modules() -> set[str]:
    """Return stem names of all public metric modules."""
    return {p.stem for p in METRICS_DIR.glob("*.py") if not p.stem.startswith("_")}


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
            "'python scripts/mkdocs_hooks/gen_metric_matrix.py' or 'mkdocs build' first."
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


def test_emitted_name_overrides_match_source() -> None:
    """``_EMITTED_NAME_OVERRIDES`` must reflect every divergent literal.

    AST-grep every ``MetricOutput(name="<lit>")`` in
    ``factrix/metrics/*.py``; for each registered user-facing function
    name, the literal must equal either the function name or the
    override map's value. Catches the case where someone adds a new
    metric that emits a non-matching ``name=`` and forgets to update
    the override map (#125 SSOT contract).
    """
    from factrix._metric_index import _EMITTED_NAME_OVERRIDES, user_facing_rows

    fn_to_emitted: dict[str, str] = {}
    for path in METRICS_DIR.glob("*.py"):
        if path.stem.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for ret in ast.walk(fn):
                if (
                    isinstance(ret, ast.Call)
                    and isinstance(ret.func, ast.Name)
                    and ret.func.id == "MetricOutput"
                ):
                    for kw in ret.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            fn_to_emitted.setdefault(fn.name, kw.value.value)

    user_fns = {r.name for r in user_facing_rows()}
    mismatches: list[str] = []
    for fn, emitted in fn_to_emitted.items():
        if fn not in user_fns or emitted == fn:
            continue
        expected = _EMITTED_NAME_OVERRIDES.get(fn)
        if expected != emitted:
            mismatches.append(
                f"{fn}: emits MetricOutput(name={emitted!r}) but override map says {expected!r}"
            )
    assert not mismatches, (
        "Emitted MetricOutput.name does not match _EMITTED_NAME_OVERRIDES:\n  "
        + "\n  ".join(mismatches)
    )


def test_generated_name_index_matches_renderer() -> None:
    """Generated name-index file must match what the renderer produces.

    Drift guard for #125: catches a stale checked-in file that no longer
    reflects the ``Matrix-row:`` SSOT (e.g. a metric was added but mkdocs
    wasn't rerun before commit).
    """
    if not GENERATED_NAME_INDEX.exists():
        pytest.skip(
            f"{GENERATED_NAME_INDEX} not found — run "
            "'python scripts/mkdocs_hooks/gen_metric_name_index.py' "
            "or 'mkdocs build' first."
        )
    from factrix._metric_index import user_facing_rows
    from scripts.mkdocs_hooks.gen_metric_name_index import (
        _TABLE_HEADER,
        _render_row,
    )

    expected = _TABLE_HEADER + "".join(
        _render_row(r) for r in sorted(user_facing_rows(), key=lambda r: r.emitted_name)
    )
    actual = GENERATED_NAME_INDEX.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{GENERATED_NAME_INDEX} is stale — re-run "
        "'python scripts/mkdocs_hooks/gen_metric_name_index.py' "
        "(or 'mkdocs build') to regenerate."
    )


def test_generated_evaluate_metric_table_matches_renderer() -> None:
    """Generated evaluate-metric table must match what the renderer produces.

    Drift guard for #144: catches a stale checked-in file that no longer
    reflects ``_DISPATCH_REGISTRY`` (e.g. a cell was added but mkdocs
    wasn't rerun before commit).
    """
    if not GENERATED_EVALUATE_METRIC.exists():
        pytest.skip(
            f"{GENERATED_EVALUATE_METRIC} not found — run "
            "'python scripts/mkdocs_hooks/gen_evaluate_metric_table.py' "
            "or 'mkdocs build' first."
        )
    from factrix._metric_index import user_facing_rows
    from factrix._registry import _DISPATCH_REGISTRY
    from scripts.mkdocs_hooks.gen_evaluate_metric_table import (
        _TABLE_HEADER,
        _render_row,
    )

    import_path_by_name = {r.name: r.import_path for r in user_facing_rows()}
    expected = _TABLE_HEADER + "".join(
        _render_row(e, import_path_by_name) for e in _DISPATCH_REGISTRY.values()
    )
    actual = GENERATED_EVALUATE_METRIC.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{GENERATED_EVALUATE_METRIC} is stale — re-run "
        "'python scripts/mkdocs_hooks/gen_evaluate_metric_table.py' "
        "(or 'mkdocs build') to regenerate."
    )
