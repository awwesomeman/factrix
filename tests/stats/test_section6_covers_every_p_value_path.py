"""Coverage guard: every metric that publishes a ``p_value`` is named in §6.

factrix's standing rule is that only *measured* inference paths are
exposed. §6 of ``docs/reference/statistical-methods.md`` is where a path's
size measurement — or an explicit statement that it has none yet — lives,
so a metric that can return a non-``None`` ``p_value`` and is absent from
§6 is an undocumented inference path, not merely a documentation gap.

The metric side is derived from source rather than from a hand-kept list,
so a new p-emitting metric fails this test until §6 accounts for it. For
each registered metric the scan walks its own function body for a
``MetricResult(...)`` whose ``p_value`` keyword is anything other than the
literal ``None``; a metric that constructs no ``MetricResult`` itself
(``quantile_spread``, ``greedy_forward_selection``) delegates to a
same-module helper, so the scan follows intra-module calls transitively
rather than falling back to the whole file — a module-wide fallback would
attribute a sibling's p-value to a descriptive metric.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
from factrix._metric_index import public_specs

METRICS_DIR = pathlib.Path("factrix/metrics")
DOCS_PAGE = pathlib.Path("docs/reference/statistical-methods.md")


def _section_6() -> str:
    text = DOCS_PAGE.read_text(encoding="utf-8")
    match = re.search(r"^## 6\..*?(?=^## 7\.)", text, re.M | re.S)
    assert match, "section 6 not found — has the page been renumbered?"
    return match.group(0)


def _module_functions(stem: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse((METRICS_DIR / f"{stem}.py").read_text(encoding="utf-8"))
    return {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _constructs_p_value(node: ast.AST) -> bool:
    """A ``MetricResult(p_value=...)`` with anything but the literal None."""
    for call in ast.walk(node):
        if not (
            isinstance(call, ast.Call)
            and getattr(call.func, "id", None) == "MetricResult"
        ):
            continue
        for kw in call.keywords:
            if kw.arg != "p_value":
                continue
            if not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                return True
    return False


def _emits_p_value(name: str, fns: dict[str, ast.FunctionDef]) -> bool:
    """Follow intra-module delegation until a p-emitting construction turns up."""
    seen: set[str] = set()
    stack = [name]
    while stack:
        current = stack.pop()
        if current in seen or current not in fns:
            continue
        seen.add(current)
        node = fns[current]
        if _constructs_p_value(node):
            return True
        for call in ast.walk(node):
            callee = (
                getattr(call.func, "id", None) if isinstance(call, ast.Call) else None
            )
            if callee in fns:
                stack.append(callee)
    return False


def _p_value_metrics() -> list[str]:
    by_module: dict[str, list[str]] = {}
    for stem, spec in public_specs():
        by_module.setdefault(stem, []).append(spec.name)
    emitting: list[str] = []
    for stem, names in sorted(by_module.items()):
        fns = _module_functions(stem)
        for name in names:
            assert name in fns, f"{stem}.{name} has no function definition"
            if _emits_p_value(name, fns):
                emitting.append(name)
    return sorted(set(emitting))


def test_the_scan_finds_a_plausible_number_of_p_value_paths():
    """Guards the guard: an AST scan that silently matches nothing would
    make the coverage assertion below vacuously true."""
    found = _p_value_metrics()
    assert len(found) >= 15, found
    # Spot-checks in both directions, so a broken scan cannot pass.
    for name in ("ic", "fm_beta", "monotonicity", "directional_hit_rate"):
        assert name in found, f"{name} publishes a p_value but the scan missed it"
    for name in ("top_concentration", "clustering_hhi", "oos_decay", "mfe_mae"):
        assert name not in found, f"{name} is descriptive but the scan claims a p_value"


@pytest.mark.parametrize("metric_name", _p_value_metrics())
def test_section_6_names_every_p_value_metric(metric_name):
    assert f"`{metric_name}`" in _section_6(), (
        f"{metric_name} publishes a p_value but §6 of "
        f"{DOCS_PAGE} does not name it. Every exposed inference path needs "
        "a size measurement there, or an explicit statement that it has "
        "none yet."
    )
