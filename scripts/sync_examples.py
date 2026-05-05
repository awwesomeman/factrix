"""Build-time sync of ``examples/*.ipynb`` into ``docs/examples/``.

The repo-root ``examples/`` directory is the SSOT for runnable demo
notebooks. ``docs/examples/`` is a build-time artifact that mkdocs-jupyter
renders into the docs site — it is regenerated before every build and is
not tracked by git.

Usage (manual)::

    python scripts/sync_examples.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib
import shutil

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SRC_DIR = _REPO_ROOT / "examples"
_DST_DIR = _REPO_ROOT / "docs" / "examples"


def sync() -> None:
    """Copy every ``*.ipynb`` from ``examples/`` to ``docs/examples/``."""
    _DST_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(_SRC_DIR.glob("*.ipynb"))
    for src in notebooks:
        dst = _DST_DIR / src.name
        shutil.copy2(src, dst)
    print(f"sync_examples: copied {len(notebooks)} notebook(s) to {_DST_DIR}")


def on_pre_build(config: object) -> None:
    """MkDocs hook: re-sync example notebooks before every build."""
    sync()


if __name__ == "__main__":
    sync()
