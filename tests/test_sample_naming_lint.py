"""FX003 lint: module-level sample-size constants follow the axis-token grammar.

AST-scans every ``factrix/**/*.py`` for module-level ``MIN_*`` /
``DEFAULT_MIN_*`` constant assignments and asserts each name matches
``MIN_[<DOMAIN>_]<AXIS>[_<TIER>]`` (the axis token is mandatory). Companion to
the ``compute_* <-> PIPELINE`` naming lint (FX001/FX002) in
``test_metric_index.py``; the grammar itself is documented in
``docs/development/architecture.md`` (Sample guards).

Scope is deliberately the module-level ``MIN_*`` constant — an open set anyone
can extend in any module, with no other guard. The closed ``SampleThreshold``
field set (backed by ``_AXES``) and metadata keys (guarded by
``test_docs_stat_keys_by_metric``) are covered elsewhere and intentionally not
re-linted here.
"""

from __future__ import annotations

import ast
import pathlib
import re

PACKAGE_DIR = pathlib.Path("factrix")

# A name that looks like a sample-size floor constant — what the lint inspects.
_CANDIDATE = re.compile(r"^(DEFAULT_)?MIN_[A-Z0-9_]+$")

# The axis-token grammar it must follow: optional ``DEFAULT_`` prefix, optional
# ``<DOMAIN>_`` prefix(es), a mandatory axis token, optional ``_HARD`` / ``_WARN``.
_VALID = re.compile(
    r"^(DEFAULT_)?MIN_([A-Z0-9]+_)*(PERIODS|ASSETS|EVENTS|PAIRS)(_HARD|_WARN)?$"
)


def _module_level_names(path: pathlib.Path) -> set[str]:
    """Names bound by a top-level ``Assign`` / ``AnnAssign`` in one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_sample_constant_naming_lint() -> None:
    """Lint FX003: every ``MIN_*`` constant carries an axis token."""
    violations: list[str] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        for name in _module_level_names(path):
            if _CANDIDATE.match(name) and not _VALID.match(name):
                violations.append(f"{path}: {name}")
    assert not violations, (
        "FX003: sample-size constants must match "
        "MIN_[<DOMAIN>_]<AXIS>[_<TIER>] with AXIS in "
        "{PERIODS, ASSETS, EVENTS, PAIRS}:\n  " + "\n  ".join(violations)
    )
