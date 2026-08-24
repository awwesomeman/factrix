"""Validates that every bibliography citation resolves to a real anchor.

Citations are written ``[Author (Year)][author-year]`` in both docstrings
and authored docs pages, and resolve through mkdocs-autorefs to an
explicit ``[](){ #author-year }`` anchor in
``docs/reference/bibliography.md``.

**Why this is not redundant with the docs build.** CI runs ``mkdocs build
--strict`` on every change under ``factrix/**`` or ``docs/**``, and
autorefs turns an unresolved cross-reference into a warning that
``--strict`` escalates to a build failure. That already covers citations
in *rendered* symbols. It does not cover citations in modules mkdocstrings
never renders: every ``::: factrix.X`` block in ``docs/api/**`` targets a
public symbol (the sole private exception is ``factrix._axis.SpecRole``),
so the citations in ``factrix/_stats/**`` and ``factrix/metrics/_helpers.py``
are plain text nobody validates. This test is the only check that reaches
them, and it gives the whole surface sub-second local feedback instead of
requiring a docs-extra install.

Three checks:

1. Every citation in ``factrix/**/*.py`` and ``docs/**/*.md`` names an
   anchor defined in ``bibliography.md``.
2. Every anchor in ``bibliography.md`` ends in a 4-digit year.
3. No year-bearing anchor is defined outside ``bibliography.md``.

Checks 2 and 3 exist to keep check 1 honest rather than to police style.
``CITATION_RE`` identifies a citation by the trailing year in its slug —
that is what makes it immune to subscript chains like ``values[i][0]``.
An anchor named without a year would silently fall outside the pattern,
and a year-bearing anchor defined on some other page would make
``bibliography.md`` the wrong authority to check against. Both conventions
hold today; if either is deliberately changed, check 1 needs rethinking,
so failing loudly here is the point.

The reverse direction — anchors that nothing cites — is deliberately not
asserted. Around 29 entries are referenced only from prose, which is a
legitimate way to carry a reference.

``docs/plans/**`` is excluded for the same reason as in
``test_docs_pages.py``: fossilised planning artifacts, also excluded from
the published site via ``mkdocs.yml`` ``exclude_docs``.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tests._doc_validation import anchors, citations

BIBLIOGRAPHY = pathlib.Path("docs/reference/bibliography.md")
DOCS_ROOT = pathlib.Path("docs")
PACKAGE_ROOT = pathlib.Path("factrix")
EXCLUDED_PREFIXES = (DOCS_ROOT / "plans",)
YEAR_SUFFIX_RE = re.compile(r"-\d{4}[a-z]?$")


def _citing_paths() -> list[pathlib.Path]:
    """Docs pages plus package sources — everywhere citations are authored."""
    pages = (
        p
        for p in DOCS_ROOT.rglob("*.md")
        if not any(p.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES)
        and p != BIBLIOGRAPHY
    )
    return sorted([*pages, *PACKAGE_ROOT.rglob("*.py")])


def _bibliography_anchors() -> set[str]:
    return anchors(BIBLIOGRAPHY.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", _citing_paths(), ids=lambda p: str(p))
def test_citations_resolve_to_a_bibliography_anchor(path: pathlib.Path) -> None:
    defined = _bibliography_anchors()
    dangling = sorted(citations(path.read_text(encoding="utf-8")) - defined)
    assert not dangling, (
        f"{path} cites bibliography anchors that do not exist: "
        f"{dangling}. Add the entry to {BIBLIOGRAPHY} with an explicit "
        f"'[](){{ #slug }}' anchor, or fix the slug in the citation."
    )


def test_bibliography_anchors_end_in_a_year() -> None:
    offenders = sorted(
        a for a in _bibliography_anchors() if not YEAR_SUFFIX_RE.search(a)
    )
    assert not offenders, (
        f"{BIBLIOGRAPHY} defines anchors without a trailing 4-digit year: "
        f"{offenders}. Citations are detected by that year, so an anchor "
        f"without one is invisible to this test. Rename it to "
        f"'author-YYYY' (with an optional 'a'/'b' suffix for same-year "
        f"works), or widen CITATION_RE in tests/_doc_validation.py."
    )


def test_year_bearing_anchors_live_only_in_the_bibliography() -> None:
    strays = {}
    for page in DOCS_ROOT.rglob("*.md"):
        if page == BIBLIOGRAPHY or any(
            page.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES
        ):
            continue
        found = sorted(
            a
            for a in anchors(page.read_text(encoding="utf-8"))
            if YEAR_SUFFIX_RE.search(a)
        )
        if found:
            strays[str(page)] = found
    assert not strays, (
        f"year-bearing anchors defined outside {BIBLIOGRAPHY}: {strays}. "
        f"Citations are checked against the bibliography alone, so a "
        f"reference anchor elsewhere would be reported as dangling. Move "
        f"the entry into the bibliography."
    )
