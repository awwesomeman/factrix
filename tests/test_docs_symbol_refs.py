"""Validates that ``factrix.X.Y`` symbol references inside the package resolve.

Docstrings and comments in ``factrix/**/*.py`` name symbols the same way
docs pages do, and nothing checked them. The layers that exist cover
neighbouring surfaces:

* ``ruff``'s ``D`` rules check docstring *structure*, not the truth of
  what a docstring says.
* ``mkdocs build --strict`` + autorefs resolve cross-references, but only
  for symbols mkdocstrings actually renders. Every ``::: factrix.X`` block
  in ``docs/api/**`` targets a public symbol (sole exception:
  ``factrix._axis.SpecRole``), so mkdocstrings never opens
  ``factrix/_stats/**``, ``factrix/metrics/_helpers.py`` or
  ``factrix/_multi_factor.py`` — their symbol references are plain text
  nobody validates.
* ``test_docs_pages.py`` and ``test_docs_llms.py`` run this same check, but
  over ``docs/**/*.md`` and the llms snapshots respectively.
* ``test_docs_bibliography.py`` walks ``factrix/**/*.py`` already, but for
  *citation anchors* — the other half of the same problem.

This file closes the remaining gap: symbol chains in package source.
Roughly 178 of the package's distinct ``factrix.X.Y`` chains live in
modules that are never rendered.

**Why mechanical rather than another manual pass.** #321 was a manual
citation-accuracy review whose stated scope included ``factrix/**/*.py``
References blocks. A misattribution of the circular block bootstrap to
Künsch (1989) sat inside that scope and survived it, only surfacing in the
v0.20.0 pre-release review. Manual passes miss; a test does not.
"""

from __future__ import annotations

import pathlib

import pytest

from tests._doc_validation import (
    is_logger_name,
    logger_namespaces,
    referenced_chains,
    resolves,
)

PACKAGE_ROOT = pathlib.Path("factrix")
SOURCE_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))
LOGGER_NAMESPACES = logger_namespaces(SOURCE_FILES)


def test_source_files_found():
    """Guard against a silently empty sweep (a bad glob passes everything)."""
    assert len(SOURCE_FILES) > 50


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p))
def test_symbol_references_resolve(path: pathlib.Path):
    """Every ``factrix.X.Y`` chain in the file resolves against the package."""
    unresolved = sorted(
        ".".join(chain)
        for chain in referenced_chains(path.read_text(encoding="utf-8"))
        if not is_logger_name(chain, LOGGER_NAMESPACES) and not resolves(chain)
    )
    assert not unresolved, (
        f"{path}: docstring/comment references that do not resolve: "
        f"{unresolved}. Either the symbol moved and the prose is stale, or "
        f"the name is a logging-hierarchy node (add it via getLogger / a "
        f"*_LOGGER_NAME constant so logger_namespaces() picks it up)."
    )


class TestLoggerNamespaceExemption:
    """The exemption must be narrow enough to still catch real breakage."""

    def test_discovered_from_source_not_hardcoded(self):
        """Every known logger namespace comes out of the sweep."""
        assert {"dag", "evaluation", "metrics"} <= LOGGER_NAMESPACES
        # f-string form is truncated at the placeholder and matches by prefix.
        assert "metric." in LOGGER_NAMESPACES

    def test_prefix_form_does_not_swallow_siblings(self):
        """``metric.`` must not exempt ``metrics.<anything>``."""
        assert is_logger_name(("metric", "ic"), LOGGER_NAMESPACES)
        # The scanner stops at the f-string brace, so the bare stem appears too.
        assert is_logger_name(("metric",), LOGGER_NAMESPACES)
        assert not is_logger_name(("metrics", "nonexistent"), LOGGER_NAMESPACES)

    def test_a_real_broken_chain_is_not_exempt(self):
        """The two residues this test was written to catch."""
        for chain in (("_describe", "list_metrics"), ("metrics", "spec_by_name")):
            assert not is_logger_name(chain, LOGGER_NAMESPACES)
            assert not resolves(chain)
