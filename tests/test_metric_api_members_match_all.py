"""Invariant: docs/api/metrics/<mod>.md ``members:`` list == module ``__all__``.

Keeps the mkdocstrings curated render surface in lockstep with each metric
module's declared public surface. Adding a new public symbol to a metric
module means updating only ``__all__`` — this test catches forgetting to
update the docs page (and vice versa).

mkdocstrings-python does not auto-follow ``__all__`` when ``members:`` is
omitted (its default filter is ``["!^_"]``, surface-by-prefix only), so
the docs page must spell out the list. SSOT is each module's ``__all__``
in teaching order; the docs ``members:`` is a derived mirror with the
same ordering, validated here.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent

# Derived from the package, not hand-listed: a hand-written list silently
# stops guarding any module added after it was written (``k_spread`` and
# ``predictive_beta`` went unchecked that way). Every public metric module
# must have a docs page, so a missing page fails loudly below.
METRIC_MODULES = sorted(
    p.stem
    for p in (_REPO / "factrix" / "metrics").glob("*.py")
    if not p.stem.startswith("_")
)
assert METRIC_MODULES, "no public metric modules found under factrix/metrics/"

_MEMBERS_BLOCK = re.compile(r"members:\n((?: +- \w+\n)+)")


def _parse_members(md_path: Path) -> list[str]:
    m = _MEMBERS_BLOCK.search(md_path.read_text(encoding="utf-8"))
    if m is None:
        return []
    return re.findall(r"- (\w+)", m.group(1))


@pytest.mark.parametrize("module_name", METRIC_MODULES)
def test_docs_members_matches_module_all(module_name: str) -> None:
    module = importlib.import_module(f"factrix.metrics.{module_name}")
    declared = list(module.__all__)
    docs_path = (
        Path(__file__).parent.parent / "docs" / "api" / "metrics" / f"{module_name}.md"
    )
    rendered = _parse_members(docs_path)
    assert rendered == declared, (
        f"docs/api/metrics/{module_name}.md members: list out of sync with "
        f"factrix.metrics.{module_name}.__all__. "
        f"docs={rendered!r} vs __all__={declared!r}"
    )
