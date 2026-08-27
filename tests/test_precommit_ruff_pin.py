"""Guard the ruff version sync between pyproject and the pre-commit config.

``pyproject.toml``'s ``dev`` extra is the source of truth for the ruff
version; ``.pre-commit-config.yaml`` pins the same version as the ``rev``
of the ruff-pre-commit repo. ``pre-commit autoupdate`` moves only the
latter, so this test fails when a bump lands on one side alone.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"


def _pyproject_ruff_pin() -> str:
    match = re.search(
        r'^\s*"ruff==(?P<version>[^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M
    )
    assert match is not None, (
        "pyproject.toml declares no pinned 'ruff==' dev dependency"
    )
    return match.group("version")


def _precommit_ruff_rev() -> str:
    text = PRECOMMIT_CONFIG.read_text(encoding="utf-8")
    match = re.search(
        r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v?(?P<version>\S+)",
        text,
    )
    assert match is not None, ".pre-commit-config.yaml pins no ruff-pre-commit rev"
    return match.group("version")


def test_precommit_ruff_rev_matches_pyproject_pin() -> None:
    pinned = _pyproject_ruff_pin()
    rev = _precommit_ruff_rev()
    assert rev == pinned, (
        f"ruff version drift: pyproject pins {pinned!r} but "
        f".pre-commit-config.yaml pins rev {rev!r}. The pyproject pin is the "
        "source of truth — bump both together."
    )
