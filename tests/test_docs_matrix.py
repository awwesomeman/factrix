"""Docs and metric-registry consistency tests."""

from __future__ import annotations

import ast
import pathlib

import pytest

METRICS_DIR = pathlib.Path("factrix/metrics")
GENERATED_MATRIX = pathlib.Path("docs/reference/_generated_metric_matrix.md")
GENERATED_NAME_INDEX = pathlib.Path("docs/reference/_generated_metric_name_index.md")


def _public_metric_modules() -> set[str]:
    """Return stem names of all public metric modules."""
    return {p.stem for p in METRICS_DIR.glob("*.py") if not p.stem.startswith("_")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", sorted(_public_metric_modules()))
def test_module_registers_metric_specs(stem: str) -> None:
    """Every public metric module must register at least one ``@metric`` class."""
    from factrix._metric_index import MetricSpec, module_specs

    specs = module_specs(stem)
    assert specs, (
        f"metrics/{stem}.py registers no @metric classes. Decorate each "
        f"public callable with @metric."
    )
    for spec in specs:
        assert isinstance(spec, MetricSpec), (
            f"metrics/{stem}.py: every registered spec must be a "
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


def test_metric_output_name_matches_spec_name() -> None:
    """Every ``MetricResult(name=...)`` literal must match its spec's ``name``.

    AST-greps every ``MetricResult(name="<lit>")`` call in
    ``factrix/metrics/*.py``; for each registered user-facing callable
    the literal must equal ``spec.name``. Catches the case where a
    metric is added that emits a non-matching ``name=``.
    """
    from factrix._metric_index import public_specs

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
                    and ret.func.id == "MetricResult"
                ):
                    for kw in ret.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            fn_to_emitted.setdefault(fn.name, kw.value.value)

    registered = {spec.name for _, spec in public_specs()}
    mismatches: list[str] = []
    for fn, emitted in fn_to_emitted.items():
        if fn not in registered:
            continue
        if fn != emitted:
            mismatches.append(
                f"{fn}: emits MetricResult(name={emitted!r}) but spec name is {fn!r}"
            )
    assert not mismatches, (
        "MetricResult.name does not match MetricSpec.name:\n  "
        + "\n  ".join(mismatches)
    )


def test_generated_name_index_matches_renderer() -> None:
    """Generated name-index file must match what the renderer produces.

    Drift guard: catches a stale checked-in file that no longer
    reflects the spec SSOT (e.g. a metric was added but mkdocs wasn't
    rerun before commit).
    """
    if not GENERATED_NAME_INDEX.exists():
        pytest.skip(
            f"{GENERATED_NAME_INDEX} not found — run "
            "'python scripts/mkdocs_hooks/gen_metric_name_index.py' "
            "or 'mkdocs build' first."
        )
    from factrix._metric_index import public_specs
    from scripts.mkdocs_hooks.gen_metric_name_index import (
        _TABLE_HEADER,
        _render_row,
    )

    expected = _TABLE_HEADER + "".join(
        _render_row(stem, spec)
        for stem, spec in sorted(public_specs(), key=lambda pair: pair[1].name)
    )
    actual = GENERATED_NAME_INDEX.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{GENERATED_NAME_INDEX} is stale — re-run "
        "'python scripts/mkdocs_hooks/gen_metric_name_index.py' "
        "(or 'mkdocs build') to regenerate."
    )
