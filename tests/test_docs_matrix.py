"""Coverage tests: ``__metric_specs__`` tuples in ``factrix/metrics/`` modules.

Validates that:
1. Every public metric module (non-underscore ``*.py``) declares a
   non-empty module-level ``__metric_specs__`` tuple of
   :class:`~factrix._metric_index.MetricSpec` instances.
2. ``docs/reference/_generated_metric_matrix.md`` exists and is
   non-empty (only meaningful after a build; skipped if the file is
   absent).
3. Each spec's resolved ``emitted_name`` matches the literal in
   ``MetricOutput(name=...)`` inside the declaring module.
4. The generated docs-name-index file matches the live renderer
   output (drift guard for #125).
5. The generated evaluate-metric table file matches the live renderer
   output (drift guard for #144).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

METRICS_DIR = pathlib.Path("factrix/metrics")
GENERATED_MATRIX = pathlib.Path("docs/reference/_generated_metric_matrix.md")
GENERATED_NAME_INDEX = pathlib.Path("docs/reference/_generated_metric_name_index.md")
GENERATED_EVALUATE_METRIC = pathlib.Path(
    "docs/reference/_generated_evaluate_metric_table.md"
)


def _public_metric_modules() -> set[str]:
    """Return stem names of all public metric modules."""
    return {p.stem for p in METRICS_DIR.glob("*.py") if not p.stem.startswith("_")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", sorted(_public_metric_modules()))
def test_module_declares_metric_specs(stem: str) -> None:
    """Every public metric module must declare a non-empty ``__metric_specs__``."""
    from factrix._metric_index import MetricSpec, module_specs

    specs = module_specs(stem)
    assert specs, (
        f"metrics/{stem}.py has no '__metric_specs__' tuple at module "
        f"scope. Declare one MetricSpec per public callable."
    )
    for spec in specs:
        assert isinstance(spec, MetricSpec), (
            f"metrics/{stem}.py: every __metric_specs__ entry must be a "
            f"MetricSpec instance; got {type(spec).__name__}"
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


def test_emitted_name_matches_metric_output_literal() -> None:
    """Every ``MetricOutput(name=...)`` literal must match its spec's resolved
    ``emitted_name``.

    AST-greps every ``MetricOutput(name="<lit>")`` call in
    ``factrix/metrics/*.py``; for each registered user-facing callable
    the literal must equal the spec's resolved ``emitted_name`` (i.e.
    ``spec.emitted_name or spec.name``). Catches the case where a
    metric is added that emits a non-matching ``name=`` and the spec
    author forgets to set ``emitted_name``.
    """
    from factrix._metric_index import emitted_name_of, public_specs

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

    expected_by_name = {spec.name: emitted_name_of(spec) for _, spec in public_specs()}
    mismatches: list[str] = []
    for fn, emitted in fn_to_emitted.items():
        if fn not in expected_by_name:
            continue
        expected = expected_by_name[fn]
        if expected != emitted:
            mismatches.append(
                f"{fn}: emits MetricOutput(name={emitted!r}) but spec "
                f"resolves emitted_name={expected!r}"
            )
    assert not mismatches, (
        "MetricOutput.name does not match MetricSpec.emitted_name:\n  "
        + "\n  ".join(mismatches)
    )


def test_generated_name_index_matches_renderer() -> None:
    """Generated name-index file must match what the renderer produces.

    Drift guard for #125: catches a stale checked-in file that no longer
    reflects the spec SSOT (e.g. a metric was added but mkdocs wasn't
    rerun before commit).
    """
    if not GENERATED_NAME_INDEX.exists():
        pytest.skip(
            f"{GENERATED_NAME_INDEX} not found — run "
            "'python scripts/mkdocs_hooks/gen_metric_name_index.py' "
            "or 'mkdocs build' first."
        )
    from factrix._metric_index import emitted_name_of, public_specs
    from scripts.mkdocs_hooks.gen_metric_name_index import (
        _TABLE_HEADER,
        _render_row,
    )

    expected = _TABLE_HEADER + "".join(
        _render_row(stem, spec)
        for stem, spec in sorted(
            public_specs(), key=lambda pair: emitted_name_of(pair[1])
        )
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
    from factrix._metric_index import import_path_for, public_specs
    from factrix._registry import _DISPATCH_REGISTRY
    from scripts.mkdocs_hooks.gen_evaluate_metric_table import (
        _TABLE_HEADER,
        _render_row,
    )

    import_path_by_name = {
        spec.name: import_path_for(stem) for stem, spec in public_specs()
    }
    expected = _TABLE_HEADER + "".join(
        _render_row(e, import_path_by_name) for e in _DISPATCH_REGISTRY.values()
    )
    actual = GENERATED_EVALUATE_METRIC.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{GENERATED_EVALUATE_METRIC} is stale — re-run "
        "'python scripts/mkdocs_hooks/gen_evaluate_metric_table.py' "
        "(or 'mkdocs build') to regenerate."
    )
