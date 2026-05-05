"""Build-time sync of ``factrix/llms*.txt`` into ``docs/`` site root.

The package directory ``factrix/`` is the SSOT for ``llms.txt`` and
``llms-full.txt`` — this lets the wheel ship them under
``site-packages/factrix/`` so agents grepping installed packages find
them locally, while this hook mirrors the same content into the docs
site so they also deploy at the public URL root.

Usage (manual)::

    python scripts/sync_llms_txt.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SRC_DIR = _REPO_ROOT / "factrix"
_DST_DIR = _REPO_ROOT / "docs"
_FILES = ("llms.txt", "llms-full.txt")

_BANNER = (
    "<!-- GENERATED FILE — DO NOT EDIT.\n"
    "     SSOT lives at factrix/{name}; regenerated on every mkdocs build\n"
    "     by scripts/sync_llms_txt.py. Edits here will be overwritten. -->\n"
    "\n"
)


def sync() -> None:
    """Copy ``factrix/llms*.txt`` into ``docs/`` with a generated banner."""
    for name in _FILES:
        src = _SRC_DIR / name
        dst = _DST_DIR / name
        body = src.read_text(encoding="utf-8")
        dst.write_text(_BANNER.format(name=name) + body, encoding="utf-8")
        print(f"sync_llms_txt: wrote {dst.relative_to(_REPO_ROOT)}")


def on_pre_build(config):  # noqa: ANN001, ARG001 — mkdocs hook signature
    """MkDocs ``on_pre_build`` hook — runs before each build."""
    sync()


if __name__ == "__main__":
    sync()
