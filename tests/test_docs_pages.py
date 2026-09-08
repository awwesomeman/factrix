"""Validates every ``docs/**/*.md`` page against the live public surface.

Catches drift in documentation: stale ``factrix.X.Y``
references or ``from factrix.subpkg import X`` that survived a rename.
This check validates every authored docs page.

``docs/plans/**`` is excluded — those pages are intentionally fossilised
historical planning artifacts (also excluded from the published site
via ``mkdocs.yml`` ``exclude_docs``).
"""

from __future__ import annotations

import pathlib

import pytest

from tests._doc_validation import (
    docs_page_paths,
    import_resolves,
    imports,
    referenced_chains,
    resolves,
)


def test_non_symbol_reference_directive_is_exact() -> None:
    text = """
<!-- factrix-doc-non-symbol: factrix.tracker.tag -->
`factrix.tracker.tag` is a storage key, not a Python symbol.
`factrix.tracker.tag.child` is not covered by the exact declaration.
`factrix.renamed_symbol` is an undeclared API reference.
"""

    failures = {chain for chain in referenced_chains(text) if not resolves(chain)}

    assert failures == {("renamed_symbol",), ("tracker", "tag", "child")}


@pytest.mark.parametrize("path", docs_page_paths(), ids=lambda p: str(p))
def test_page_references_resolve(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    failures = sorted(
        ".".join(chain) for chain in referenced_chains(text) if not resolves(chain)
    )
    assert not failures, (
        f"Unresolvable factrix.* references in {path}:\n  "
        + "\n  ".join(f"factrix.{f}" for f in failures)
    )


@pytest.mark.parametrize("path", docs_page_paths(), ids=lambda p: str(p))
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
