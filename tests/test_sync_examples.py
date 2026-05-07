"""Unit tests for ``scripts/mkdocs_hooks/sync_examples.py``.

Targets the prune behaviour added in #99: a notebook removed from
``examples/`` should disappear from ``docs/examples/`` on the next
build, not linger as a "not in nav" warning forever.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from scripts.mkdocs_hooks import sync_examples

_MINIMAL_NB = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Title\n", "\n", "## Use this when\n", "\n", "- teaser\n"],
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _write_nb(path: pathlib.Path, title: str = "Title") -> None:
    nb = json.loads(json.dumps(_MINIMAL_NB))
    nb["cells"][0]["source"] = [
        f"# {title}\n",
        "\n",
        "## Use this when\n",
        "\n",
        "- teaser line\n",
    ]
    path.write_text(json.dumps(nb), encoding="utf-8")


@pytest.fixture
def isolated_dirs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pathlib.Path, pathlib.Path]:
    src = tmp_path / "examples"
    dst = tmp_path / "docs" / "examples"
    src.mkdir(parents=True)
    dst.mkdir(parents=True)
    monkeypatch.setattr(sync_examples, "_SRC_DIR", src)
    monkeypatch.setattr(sync_examples, "_DST_DIR", dst)
    return src, dst


def test_sync_prunes_stale_destination_notebook(
    isolated_dirs: tuple[pathlib.Path, pathlib.Path],
) -> None:
    src, dst = isolated_dirs
    _write_nb(src / "current.ipynb", title="Current")
    _write_nb(dst / "current.ipynb", title="Current")
    _write_nb(dst / "retired.ipynb", title="Retired")

    sync_examples.sync()

    assert (dst / "current.ipynb").exists()
    assert not (dst / "retired.ipynb").exists()


def test_sync_does_not_touch_files_outside_dst(
    isolated_dirs: tuple[pathlib.Path, pathlib.Path],
) -> None:
    src, dst = isolated_dirs
    _write_nb(src / "a.ipynb", title="A")
    sibling = dst.parent / "sibling.ipynb"
    sibling.write_text("untouched", encoding="utf-8")

    sync_examples.sync()

    assert sibling.read_text(encoding="utf-8") == "untouched"


def test_sync_leaves_non_notebook_files_alone(
    isolated_dirs: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """The prune glob is ``*.ipynb`` only; ``index.md`` etc. survive."""
    src, dst = isolated_dirs
    _write_nb(src / "a.ipynb", title="A")
    keepme = dst / "keepme.txt"
    keepme.write_text("not a notebook", encoding="utf-8")

    sync_examples.sync()

    assert keepme.exists()


def test_sync_handles_rename_in_one_pass(
    isolated_dirs: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """Renamed notebook (old name pruned + new name copied) is handled
    in a single ``sync()`` call — the prune-before-copy ordering."""
    src, dst = isolated_dirs
    _write_nb(src / "renamed.ipynb", title="Renamed")
    _write_nb(dst / "old_name.ipynb", title="Renamed")

    sync_examples.sync()

    assert (dst / "renamed.ipynb").exists()
    assert not (dst / "old_name.ipynb").exists()
