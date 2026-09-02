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
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)

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
    """GitHub's heading slug: lowercase, punctuation dropped, spaces hyphened."""
    text = heading.strip().lower()
    text = re.sub(r"[`*_]", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text).strip("-")


def _anchors(path: pathlib.Path) -> set[str]:
    return {
        _slug(heading) for heading in _HEADING.findall(path.read_text(encoding="utf-8"))
    }


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
