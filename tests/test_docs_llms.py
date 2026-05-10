"""Validates ``factrix/llms-full.txt`` and ``factrix/llms.txt`` against the
live public API surface.

**Approach: snapshot, not generation.** ``llms-full.txt`` is heavily
curated narrative + tables + worked examples. Generating it from
docstrings would lose editorial control over what an LLM agent sees.
Instead, this test parses the existing file for symbol references and
verifies that every reference still resolves and that every public
``__all__`` symbol gets at least one mention.

Three checks:

1. Every ``factrix.X.Y`` / ``fx.X.Y`` attribute chain in ``llms-full.txt``
   walks from the ``factrix`` package without raising ``AttributeError``.
2. Every ``from factrix... import NAME`` statement imports a name that
   actually exists at that module path.
3. Every name in ``factrix.__all__`` appears at least once in
   ``llms-full.txt`` — keeps the LLM reference in lockstep when the
   public surface widens.

Bare references like ``StatCode.MEAN`` (no ``fx.`` / ``factrix.``
prefix) are intentionally not validated — too many false positives
against ``profile.X`` style attribute talk in prose.

Sibling test ``test_docs_pages.py`` runs the resolution checks
against every ``docs/**/*.md`` page; this file remains specific to
the curated llms snapshot (because of the ``__all__``-coverage check).
"""

from __future__ import annotations

import pathlib
import re

import factrix

from tests._doc_validation import (
    import_resolves,
    imports,
    referenced_chains,
    resolves,
)

LLMS_FULL = pathlib.Path("factrix/llms-full.txt")
LLMS_INDEX = pathlib.Path("factrix/llms.txt")


def test_every_referenced_symbol_resolves() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    failures = sorted(
        ".".join(chain) for chain in referenced_chains(text) if not resolves(chain)
    )
    assert not failures, (
        "Unresolvable factrix.* references in llms-full.txt:\n  "
        + "\n  ".join(f"factrix.{f}" for f in failures)
    )


def test_every_imported_name_resolves() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    failures = [
        f"{module}.{name}" if name else f"{module} (module not importable)"
        for module, name in imports(text)
        if not import_resolves(module, name)
    ]
    assert not failures, (
        "Imports in llms-full.txt that do not resolve:\n  " + "\n  ".join(failures)
    )


def test_every_public_symbol_mentioned_in_llms_full() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    # Word-boundary match: prevents `Metric` from being silently satisfied
    # by `MetricOutput` (and similarly for other short prefix names).
    missing = [
        name
        for name in sorted(factrix.__all__)
        if not re.search(rf"\b{re.escape(name)}\b", text)
    ]
    assert not missing, (
        "Public symbols in factrix.__all__ never mentioned in llms-full.txt:\n  "
        + "\n  ".join(missing)
        + "\nAdd at least one mention so LLM agents do not miss them."
    )


def test_llms_index_exists_and_nonempty() -> None:
    text = LLMS_INDEX.read_text(encoding="utf-8")
    assert text.strip(), f"{LLMS_INDEX} is empty"
    assert "factrix" in text.lower()
