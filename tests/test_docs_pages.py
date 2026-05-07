"""Validates every ``docs/**/*.md`` page against the live public surface.

Catches the drift class flagged by PR #97: a stale ``factrix.X.Y``
reference or a ``from factrix.subpkg import X`` that survived a rename.
The validator from #90 only inspected ``factrix/llms-full.txt``; this
sibling extends the same snapshot approach to every authored docs page.

``docs/plans/**`` is excluded — those pages are intentionally fossilised
historical planning artifacts (also excluded from the published site
via ``mkdocs.yml`` ``exclude_docs``).
"""

from __future__ import annotations

import pathlib

import pytest

from tests._doc_validation import (
    import_resolves,
    imports,
    referenced_chains,
    resolves,
)

DOCS_ROOT = pathlib.Path("docs")
EXCLUDED_PREFIXES = (DOCS_ROOT / "plans",)


def _page_paths() -> list[pathlib.Path]:
    return sorted(
        p
        for p in DOCS_ROOT.rglob("*.md")
        if not any(p.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES)
    )


@pytest.mark.parametrize("path", _page_paths(), ids=lambda p: str(p))
def test_page_references_resolve(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    failures = sorted(
        ".".join(chain) for chain in referenced_chains(text) if not resolves(chain)
    )
    assert not failures, (
        f"Unresolvable factrix.* references in {path}:\n  "
        + "\n  ".join(f"factrix.{f}" for f in failures)
    )


@pytest.mark.parametrize("path", _page_paths(), ids=lambda p: str(p))
def test_page_imports_resolve(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    failures = [
        f"{module}.{name}" if name else f"{module} (module not importable)"
        for module, name in imports(text)
        if not import_resolves(module, name)
    ]
    assert not failures, f"Imports in {path} that do not resolve:\n  " + "\n  ".join(
        failures
    )
