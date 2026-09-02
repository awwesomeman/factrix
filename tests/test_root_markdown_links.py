"""Relative links in the repo-root markdown files resolve.

``mkdocs.yml`` sets ``validation.links.not_found`` and ``.anchors`` to ``warn``,
which ``--strict`` turns into errors, so every link inside ``docs/**`` is
already checked by the build. The repo-root files are outside it: ``README.md``,
``CONTRIBUTING.md``, ``CLAUDE.md`` and ``CHANGELOG.md`` are never built, so
nothing notices when a target moves.

``CLAUDE.md``'s two links carry anchors, which is the failure that stays
silent longest: renaming ``## Testing rules`` or ``## Period grid, not
calendar`` leaves the link pointing at a file that still exists, at a section
that no longer does.

External links are not checked here — that is the ``link-check`` workflow's
job, and reaching the network from the unit suite would make it flaky for a
reason unrelated to the repo.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(".")

#: Inline markdown links: ``[text](target)`` or ``[text](target "title")``.
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: ATX headings. Setext headings are not used in this repo's markdown; the
#: population assertions below fail loudly if that stops being true and a
#: link starts pointing at one.
#:
#: Applied line by line rather than with ``re.MULTILINE`` over the whole file,
#: because a ``#`` inside a fenced block is a comment in the fenced language,
#: not a heading. ``contributing.md`` carries ``# edit and test`` in a shell
#: fence, which a whole-file scan turns into a phantom anchor that a broken
#: link can then resolve against.
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")

#: Opening or closing line of a fenced code block.
_FENCE = re.compile(r"^\s*(?:```|~~~)")

_EXTERNAL = ("http://", "https://", "mailto:", "#")


def _root_markdown() -> list[pathlib.Path]:
    return sorted(REPO_ROOT.glob("*.md"))


def _relative_links() -> list[tuple[pathlib.Path, str]]:
    """``(source file, target)`` for every non-external inline link."""
    found: list[tuple[pathlib.Path, str]] = []
    for path in _root_markdown():
        for match in _LINK.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1)
            if not target.startswith(_EXTERNAL):
                found.append((path, target))
    return found


def _slug(heading: str) -> str:
    """GitHub's heading slug: lowercase, punctuation dropped, spaces hyphened.

    ``_`` is **kept**. It is a word character, and GitHub slugs the *rendered*
    heading, where an underscore inside backticks survives as code. Stripping
    it alongside the formatting characters made 11 of ``architecture.md``'s 43
    anchors wrong — in a repo whose section titles are snake_case API names,
    ``expand_over-semantics`` became ``expandover-semantics``, so a correct
    link would have been rejected. Underscore as markdown *emphasis* would
    slug differently, but a sweep of every heading under ``docs/`` and the
    repo root finds 196 snake_case uses and zero emphatic ones.
    """
    text = heading.strip().lower()
    text = re.sub(r"[`*]", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text).strip("-")


def _headings(text: str) -> list[str]:
    """ATX headings outside fenced code blocks."""
    found: list[str] = []
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = _HEADING.match(line)
        if match:
            found.append(match.group(1))
    return found


def _anchors(path: pathlib.Path) -> set[str]:
    return {_slug(h) for h in _headings(path.read_text(encoding="utf-8"))}


def test_the_scan_finds_the_links_it_is_meant_to_check() -> None:
    """Without this the whole module passes by matching nothing.

    Every assertion below is a loop over what the regex found, so a regex that
    stops matching turns the file green rather than red — the failure mode
    that makes a link checker worthless. Assert the scan has a population, and
    that it still reaches every root file that carries a relative link.
    """
    assert _root_markdown(), "no markdown files at the repo root"

    links = _relative_links()
    assert links, (
        "no relative links found in the repo-root markdown — either they were "
        "all removed or _LINK stopped matching the syntax in use."
    )

    sources = {path.name for path, _ in links}
    assert {"README.md", "CONTRIBUTING.md", "CLAUDE.md"} <= sources, (
        f"the scan reached only {sorted(sources)}; a file that carries "
        "relative links is no longer being read."
    )


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        # Underscore survives: it is a word character, and snake_case API
        # names are what this repo's section titles are built from.
        ("`expand_over` semantics", "expand_over-semantics"),
        # ...including next to a formatting character that does come out.
        (
            "Naming: `data` (DataFrame) vs `df_*` (degrees of freedom)",
            "naming-data-dataframe-vs-df_-degrees-of-freedom",
        ),
        # Emphasis markers are formatting, not content.
        ("**Bold** and *italic*", "bold-and-italic"),
        # Punctuation dropped, whitespace runs collapse to one hyphen.
        ("Period grid, not calendar", "period-grid-not-calendar"),
    ],
)
def test_slug_follows_githubs_rule(heading: str, expected: str) -> None:
    """The only coverage of the underscore rule, not a guard for a guard.

    Neither anchored link in the repo root contains an underscore
    (``period-grid-not-calendar``, ``testing-rules``), so with these cases
    removed, reintroducing the ``[`*_]`` strip that this file shipped with
    passes every remaining test. That makes this the regression test for the
    defect, which is ordinary practice rather than extra machinery.

    GitHub's rule is the target because these files are read on GitHub;
    ``docs/**`` stays on mkdocs' rule via ``--strict``, and the two differing
    is by design.
    """
    assert _slug(heading) == expected


def test_headings_inside_code_fences_are_not_anchors() -> None:
    """A ``#`` in a fenced block is a comment, not a heading.

    Reachable today: ``contributing.md`` has ``# edit and test`` inside a
    shell fence. Counting it would let a link to ``#edit-and-test`` resolve
    against an anchor GitHub never creates — a false pass, the direction that
    stays silent.
    """
    text = "\n".join(
        [
            "# Real heading",
            "",
            "```bash",
            "# edit and test",
            "```",
            "",
            "## Another real one",
            "",
            "~~~python",
            "# not a heading either",
            "~~~",
        ]
    )
    assert _headings(text) == ["Real heading", "Another real one"]


def test_at_least_one_link_is_anchored() -> None:
    """The anchor branch must actually run; it is the risky half."""
    anchored = [(path, target) for path, target in _relative_links() if "#" in target]
    assert anchored, (
        "no anchored relative links found, so test_relative_links_resolve "
        "never exercises its anchor check."
    )


@pytest.mark.parametrize(
    ("source", "target"),
    _relative_links(),
    ids=lambda value: str(value).replace("/", "-"),
)
def test_relative_links_resolve(source: pathlib.Path, target: str) -> None:
    path_part, _, anchor = target.partition("#")
    resolved = (source.parent / path_part).resolve()

    assert resolved.exists(), f"{source}: '{target}' points at a missing path"

    if not anchor:
        return

    assert resolved.suffix == ".md", (
        f"{source}: '{target}' anchors into a non-markdown file"
    )
    available = _anchors(resolved)
    assert anchor in available, (
        f"{source}: '{target}' names an anchor that no heading in "
        f"{path_part} produces. Closest available: "
        f"{sorted(a for a in available if a.split('-')[0] == anchor.split('-')[0])}"
    )
