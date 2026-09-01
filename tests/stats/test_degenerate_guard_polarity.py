"""Regression lint for NaN-safe inference degeneracy guards.

``nan < EPSILON`` is false, so a negative-polarity threshold cannot by itself
withhold inference. FX007 records the existing sites instead of forcing a
high-risk bulk rewrite, but rejects any new bare comparison. A site leaves the
baseline when it adopts ``_degenerate_t_input`` or an explicit non-finite guard.
"""

from __future__ import annotations

import ast
import pathlib
import re
from collections import Counter

import pytest
from factrix._stats.core import _degenerate_t_input
from factrix._types import EPSILON

PACKAGE_DIR = pathlib.Path("factrix")

_QUANTITY_TOKEN = re.compile(
    r"(^|_)(se\d*|std|sigma\d*|var|denom|sxx|scale|gamma)(_|$)"
)

# Line-independent baseline of pre-existing negative-polarity guards. Counts
# distinguish repeated expressions in one module without pinning source lines.
_FX007_BASELINE = Counter(
    {
        ("factrix/_ols.py", "sigma2 < EPSILON"): 1,
        ("factrix/_stats/bootstrap.py", "gamma_0 < EPSILON"): 1,
        ("factrix/_stats/core.py", "denom < EPSILON"): 1,
        ("factrix/_stats/core.py", "var_s <= EPSILON"): 1,
        ("factrix/_stats/diagnostics.py", "denom < EPSILON"): 1,
        ("factrix/_stats/diagnostics.py", "var < EPSILON"): 1,
        ("factrix/_stats/hac.py", "denom < EPSILON"): 1,
        ("factrix/_stats/hac.py", "se < EPSILON"): 2,
        ("factrix/_stats/ols.py", "se < EPSILON"): 1,
        ("factrix/_stats/ols.py", "sxx < EPSILON"): 2,
        ("factrix/_stats/unit_root.py", "se < EPSILON"): 1,
        ("factrix/_stats/unit_root.py", "sigma2 < EPSILON"): 1,
        ("factrix/metrics/_helpers.py", "denom < EPSILON"): 1,
        ("factrix/metrics/common_beta.py", "float(np.std(x)) < EPSILON"): 1,
        ("factrix/metrics/common_beta.py", "var_total <= EPSILON * EPSILON"): 1,
        ("factrix/metrics/corrado_rank.py", "std_u < EPSILON"): 1,
        ("factrix/metrics/fm_beta.py", "sigma2_f < EPSILON"): 1,
        ("factrix/metrics/ic.py", "std_ic < EPSILON"): 1,
        ("factrix/metrics/predictive_beta.py", "x_std < EPSILON"): 1,
        ("factrix/slicing/period_inference.py", "se < EPSILON"): 1,
        ("factrix/slicing/period_inference.py", "se2 <= EPSILON"): 1,
    }
)


def _contains_epsilon(node: ast.AST) -> bool:
    return any(
        isinstance(part, ast.Name) and part.id == "EPSILON" for part in ast.walk(node)
    )


def _is_inference_quantity(node: ast.AST) -> bool:
    for part in ast.walk(node):
        if isinstance(part, ast.Name) and _QUANTITY_TOKEN.search(part.id.lower()):
            return True
        if isinstance(part, ast.Attribute) and part.attr in {"std", "var"}:
            return True
    return False


def _finite_guard_target(node: ast.AST) -> str | None:
    if not isinstance(node, ast.UnaryOp) or not isinstance(node.op, ast.Not):
        return None
    call = node.operand
    if not isinstance(call, ast.Call) or len(call.args) != 1:
        return None
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "isfinite":
        return None
    return ast.dump(call.args[0], include_attributes=False)


def _has_finite_or_guard(compare: ast.Compare, parents: dict[ast.AST, ast.AST]) -> bool:
    target = ast.dump(compare.left, include_attributes=False)
    node: ast.AST = compare
    while node in parents:
        node = parents[node]
        if (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and target
            in {
                guarded
                for value in node.values
                if (guarded := _finite_guard_target(value)) is not None
            }
        ):
            return True
        if isinstance(node, ast.stmt):
            break
    return False


def _bare_guards(source: str, path: str) -> Counter[tuple[str, str]]:
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    guards: Counter[tuple[str, str]] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or _has_finite_or_guard(node, parents):
            continue
        operands = [node.left, *node.comparators]
        for left, operator, right in zip(
            operands[:-1], node.ops, operands[1:], strict=True
        ):
            if (
                isinstance(operator, ast.Lt | ast.LtE)
                and _is_inference_quantity(left)
                and _contains_epsilon(right)
            ):
                symbol = "<" if isinstance(operator, ast.Lt) else "<="
                guards[
                    (path, f"{ast.unparse(left)} {symbol} {ast.unparse(right)}")
                ] += 1
    return guards


def _package_guards() -> Counter[tuple[str, str]]:
    guards: Counter[tuple[str, str]] = Counter()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        guards.update(_bare_guards(path.read_text(encoding="utf-8"), path.as_posix()))
    return guards


def test_no_new_bare_degenerate_threshold_guards() -> None:
    """Lint FX007: additions must use the canonical or explicit safe guard."""
    actual = _package_guards()
    unexpected = actual - _FX007_BASELINE
    stale = _FX007_BASELINE - actual
    assert not unexpected and not stale, (
        "FX007: a negative-polarity inference threshold is not NaN-safe. "
        "Use _degenerate_t_input for t-statistic inputs or add an explicit "
        "`not np.isfinite(x) or ...` guard. Remove migrated sites from the "
        f"baseline.\nunexpected: {unexpected}\nstale: {stale}"
    )


def test_fx007_distinguishes_safe_guard_polarities() -> None:
    """Guard the lint against accepting a new bare comparison or safe forms."""
    source = """
def checks(se):
    if se < EPSILON:
        return 1
    if not np.isfinite(se) or se < EPSILON:
        return 2
    if se > EPSILON:
        return 3
"""
    assert _bare_guards(source, "example.py") == Counter(
        {("example.py", "se < EPSILON"): 1}
    )


@pytest.mark.parametrize(
    ("std", "n"),
    [
        (float("nan"), 1),
        (float("inf"), 1),
        (float("-inf"), 1),
        (0.0, 1),
        (EPSILON, 1),
        (1.0, 0),
    ],
)
def test_canonical_t_guard_rejects_degenerate_inputs(std: float, n: int) -> None:
    assert _degenerate_t_input(std, n) is True


def test_canonical_t_guard_accepts_finite_positive_input() -> None:
    assert _degenerate_t_input(EPSILON * 2.0, 1) is False
