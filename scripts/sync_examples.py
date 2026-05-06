"""Build-time sync of ``examples/*.ipynb`` into ``docs/examples/`` + index.

The repo-root ``examples/`` directory is the SSOT for runnable demo
notebooks. ``docs/examples/`` is a build-time artifact that mkdocs-jupyter
renders into the docs site — it is regenerated before every build and is
not tracked by git.

Two responsibilities:

1. Copy every ``*.ipynb`` from ``examples/`` to ``docs/examples/``.
2. Generate ``docs/examples/index.md`` from each notebook — H1 in the
   first markdown cell becomes the index entry title; the first bullet
   under ``## Use this when`` becomes the one-line teaser.

The index is generated rather than hand-maintained so adding a new
recipe cannot drift from the listing.

Usage (manual)::

    python scripts/sync_examples.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
from dataclasses import dataclass

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SRC_DIR = _REPO_ROOT / "examples"
_DST_DIR = _REPO_ROOT / "docs" / "examples"


@dataclass(frozen=True)
class _RecipeMeta:
    """Index metadata extracted from a recipe notebook."""

    filename: str
    title: str
    teaser: str


def _markdown_sources(nb_path: pathlib.Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        out.append(src if isinstance(src, str) else "".join(src))
    return out


def _extract_h1(text: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError("missing H1 in first markdown cell")
    return match.group(1).strip()


def _extract_use_when_bullet(sources: list[str]) -> str:
    """First bullet under ``## Use this when``, joined across wrap lines.

    A markdown bullet may wrap onto continuation lines indented under the
    same item. We capture until the next bullet (``\\n-``) or blank line.
    """
    heading = re.compile(r"##\s+Use this when\s*\n+")
    bullet = re.compile(
        r"^-\s+(.+?)(?=\n\s*-\s|\n\s*\n|\Z)",
        flags=re.DOTALL | re.MULTILINE,
    )
    for text in sources:
        h = heading.search(text)
        if not h:
            continue
        b = bullet.search(text, h.end())
        if b:
            # Collapse internal newlines + leading whitespace into single space
            raw = b.group(1)
            return re.sub(r"\s*\n\s*", " ", raw).strip()
    raise ValueError("no '## Use this when' bullet found")


def _collect(nb_path: pathlib.Path) -> _RecipeMeta:
    sources = _markdown_sources(nb_path)
    if not sources:
        raise ValueError(f"{nb_path}: no markdown cells")
    return _RecipeMeta(
        filename=nb_path.name,
        title=_extract_h1(sources[0]),
        teaser=_extract_use_when_bullet(sources),
    )


_INDEX_HEADER = """# Examples

Hand-edited Jupyter notebooks demonstrating one factrix research scenario
each. The source `.ipynb` files live in the
[`examples/`](https://github.com/awwesomeman/factrix/tree/main/examples)
directory of the repository — clone and run them locally to experiment,
or read the rendered output below.

## Available recipes

"""


def _render_index(metas: list[_RecipeMeta]) -> str:
    lines = [_INDEX_HEADER]
    for meta in sorted(metas, key=lambda m: m.filename):
        lines.append(f"- [{meta.title}]({meta.filename}) — {meta.teaser}\n")
    return "".join(lines)


def sync() -> None:
    """Copy notebooks and regenerate ``docs/examples/index.md``."""
    _DST_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(_SRC_DIR.glob("*.ipynb"))
    metas: list[_RecipeMeta] = []
    for src in notebooks:
        dst = _DST_DIR / src.name
        shutil.copy2(src, dst)
        metas.append(_collect(src))

    (_DST_DIR / "index.md").write_text(_render_index(metas), encoding="utf-8")
    print(f"sync_examples: synced {len(notebooks)} notebook(s) + index to {_DST_DIR}")


def on_pre_build(config: object) -> None:
    """MkDocs hook: re-sync example notebooks before every build."""
    sync()


if __name__ == "__main__":
    sync()
